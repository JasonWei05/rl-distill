#!/usr/bin/env python3
"""Mirror JWei05 Hugging Face model repos to S3 (copy only; nothing is deleted from HF).

For each Gemma repo under the author: snapshot_download into a /tmp NVMe staging dir (never EFS), `aws s3 sync` to
S3_ROOT/<repo>/ (repo-relative layout, so `hf_exports`-style consumers can point at the S3 prefix), verify every HF file
appears on S3 with the same byte size, write _MIRROR_COMPLETE.json (file list, sizes, HF revision) and remove the staging
copy. Resumable: repos with a complete marker are skipped; partial syncs are resumed by `aws s3 sync`. Repos are processed
in the order given (--order-file) or largest-last by default so the not-yet-backed-up ones land first.

    setsid nohup python rl-distill-scripts/scale_train/mirror_hf_to_s3.py > /tmp/hf_mirror/mirror.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

S3_ROOT = "s3://scale-ml/genai/rl-distill/hf-mirror/JWei05"
AUTHOR = "JWei05"


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def s3_listing(prefix: str) -> dict[str, int]:
    out = subprocess.run(["aws", "s3", "ls", "--recursive", prefix + "/"], capture_output=True, text=True)
    sizes: dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            sizes[parts[3][len(prefix.split("/", 3)[3]) + 1:]] = int(parts[2])
    return sizes


def mirror_one(api: HfApi, repo_id: str, staging_root: Path, token: str | None) -> None:
    name = repo_id.split("/", 1)[1]
    dest = f"{S3_ROOT}/{name}"
    marker = f"{dest}/_MIRROR_COMPLETE.json"
    if subprocess.run(["aws", "s3", "ls", marker], capture_output=True).returncode == 0:
        print(f"{ts()} SKIP {repo_id} (complete marker present)", flush=True)
        return
    info = api.model_info(repo_id, files_metadata=True, token=token)
    expected = {s.rfilename: s.size or 0 for s in info.siblings}
    total_gb = sum(expected.values()) / 1e9
    print(f"{ts()} START {repo_id} files={len(expected)} size={total_gb:.1f}GB rev={info.sha[:8]}", flush=True)
    free_gb = shutil.disk_usage(staging_root).free / 1e9
    if free_gb < total_gb + 200:
        raise RuntimeError(f"staging has {free_gb:.0f} GB free, need {total_gb:.0f}+200")
    local = staging_root / name
    snapshot_download(repo_id, revision=info.sha, local_dir=local, token=token, max_workers=16)
    subprocess.run(["aws", "s3", "sync", "--only-show-errors", "--exclude", ".cache/*", str(local), dest], check=True)
    on_s3 = s3_listing(dest)
    missing = {f: sz for f, sz in expected.items() if on_s3.get(f) != sz}
    if missing:
        raise RuntimeError(f"{repo_id}: {len(missing)} files missing/size-mismatch on S3, e.g. {list(missing)[:3]}")
    receipt = {"repo_id": repo_id, "hf_revision": info.sha, "files": expected, "total_bytes": sum(expected.values()), "mirrored_at": ts(), "s3_prefix": dest}
    receipt_path = local / "_MIRROR_COMPLETE.json"
    receipt_path.write_text(json.dumps(receipt, indent=1))
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(receipt_path), marker], check=True)
    shutil.rmtree(local)
    print(f"{ts()} DONE {repo_id} {total_gb:.1f}GB -> {dest}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=Path("/tmp/hf_mirror/staging"))
    parser.add_argument("--filter", default="gemma", help="case-insensitive substring a repo id must contain")
    parser.add_argument("--order-file", type=Path, default=None, help="one repo id per line; listed repos go first in that order")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these repo ids")
    args = parser.parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repos = [m.id for m in api.list_models(author=AUTHOR, limit=1000) if args.filter.lower() in m.id.lower()]
    if args.only:
        repos = [r for r in repos if r in set(args.only)]
    sizes = {}
    for r in repos:
        sizes[r] = sum((s.size or 0) for s in api.model_info(r, files_metadata=True, token=token).siblings)
    first = [line.strip() for line in args.order_file.read_text().splitlines() if line.strip()] if args.order_file else []
    rest = sorted((r for r in repos if r not in first), key=sizes.get)
    ordered = [r for r in first if r in sizes] + rest
    print(f"{ts()} PLAN {len(ordered)} repos, {sum(sizes.values())/1e12:.2f} TB", flush=True)
    failures = []
    for repo_id in ordered:
        for attempt in range(1, 4):
            try:
                mirror_one(api, repo_id, args.staging, token)
                break
            except Exception as exc:  # noqa: BLE001 - keep the batch going, report at the end
                print(f"{ts()} FAIL {repo_id} attempt {attempt}: {exc}", flush=True)
                time.sleep(60)
        else:
            failures.append(repo_id)
    print(f"{ts()} ALL_DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
