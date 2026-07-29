#!/usr/bin/env python3
"""List in-progress and queued ScaleTrain jobs for a team, with GPU counts.

The ScaleTrain CLI (`scale-train list jobs`) is per-user and interactive-only,
so this queries Kubernetes directly using the `team=<name>` labels that
ScaleTrain stamps on its jobs/pods. EKS cluster only (kubectl's current
context); GKE jobs won't appear.

Usage: python3 list_team_jobs.py [team]   # default: egp
"""
import collections
import json
import subprocess
import sys

team = sys.argv[1] if len(sys.argv) > 1 else "egp"


def kubectl(kind):
    out = subprocess.check_output(
        ["kubectl", "get", kind, "-n", "train", "-l", f"team={team}", "-o", "json"]
    )
    return json.loads(out)["items"]


def pod_gpus(spec):
    return sum(
        int(c["resources"].get("requests", {}).get("nvidia.com/gpu", 0))
        for c in spec["containers"]
    )


pods_by_id = collections.defaultdict(list)
for p in kubectl("pods"):
    pods_by_id[p["metadata"]["labels"].get("scaletrain/job_id", "?")].append(p)

running, queued = [], []
for j in kubectl("jobs"):
    m, s, spec = j["metadata"], j.get("status", {}), j["spec"]
    if any(
        c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
        for c in s.get("conditions", [])
    ):
        continue
    lab = m["labels"]
    name = lab.get("scaletrain/job_name", "?")
    user = lab.get("user", "?")
    jid = lab.get("scaletrain/job_id", "?")
    gpp = pod_gpus(spec["template"]["spec"])
    row = (name, user, jid, m["creationTimestamp"])
    if spec.get("suspend"):
        queued.append(row + (gpp, "suspended (Kueue, not admitted)"))
        continue
    run_gpus = sum(
        pod_gpus(p["spec"]) for p in pods_by_id[jid] if p["status"]["phase"] == "Running"
    )
    if run_gpus:
        running.append(row + (run_gpus,))
    else:
        queued.append(row + (gpp, "admitted, pod Pending/unschedulable"))

print(f"=== IN PROGRESS on {team} ({len(running)} jobs) ===")
for name, user, jid, ts, g in sorted(running, key=lambda r: (r[1], r[0])):
    print(f"{name:34s} {user:18s} {jid}  gpus={g:3d}  since={ts}")
print(f"TOTAL RUNNING GPUs: {sum(r[4] for r in running)}\n")

print(f"=== QUEUED on {team} ({len(queued)} jobs) ===")
for name, user, jid, ts, g, why in sorted(queued, key=lambda r: (r[1], r[0], r[3])):
    print(f"{name:34s} {user:18s} {jid}  gpus_requested={g:3d}  {why}  created={ts}")
print(f"TOTAL QUEUED GPU REQUESTS: {sum(r[4] for r in queued)}")
