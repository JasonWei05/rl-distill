#!/usr/bin/env python3
"""Re-publish a finished run's best_hf at its best SAVED checkpoint, with a fresh, verified completion marker.

Selection: the step with the highest W&B validation metric among steps that have a complete permanent checkpoint
(<ckpt>/global_step_N/_REMOTE_COMPLETE.json) with an HF export, optionally capped by --max-step; earliest step on ties.
This replaces the run's own choice when that choice is not the true best (early stopping off -> final step; min_delta early
stopping -> a "best" that lags the max by < min_delta) and does not depend on pod logs (which miss steps across outages).

Publish order is crash-safe: the old best_hf/_REMOTE_COMPLETE.json is deleted FIRST (a crash mid-way leaves best_hf
"incomplete", never "complete but stale"), then the old best_hf objects, then the HF export is server-side copied, then
run_outcome.json is rewritten (the run-file's original is kept once as run_outcome.runfile.json), then the size manifest is
written and checked with full_checkpoint_s3.check_best_hf_export -- the same check the run-file preflight uses. The outcome is
also copied to best_hf/_RUN_OUTCOME.json (run_qwen3_opd_math_suite_watcher.sh keys on it). stop_qwen3_opd_at_step.sh passes
--stop-reason / --final-step for a run it cut short (no run-file outcome exists yet).
  python republish_best_hf.py --artifact-s3 s3://.../qwen3-opd/<spec>-<tag> --ckpt-s3 s3://.../qwen3-opd-full-checkpoints/<spec>-<tag> \
      --wandb-run rl-distill/DAPO/<run id> [--max-step 400] [--dry-run]
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_checkpoint_s3 as fcs  # noqa: E402

METRIC = "val-core/math/acc/mean@16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-s3", required=True, help="run artifact root holding best_hf/ and run_outcome.json")
    ap.add_argument("--ckpt-s3", required=True, help="full-checkpoint root holding global_step_N/")
    ap.add_argument("--wandb-run", required=True, help="entity/project/run_id")
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--stop-reason", default=None, help="override the outcome's stop_reason (manual stops)")
    ap.add_argument("--final-step", type=int, default=None, help="override the outcome's final_step (manual stops)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import wandb

    h = wandb.Api(timeout=180).run(a.wandb_run).history(keys=[a.metric], samples=100000, pandas=True)
    vals = {int(s): float(v) for s, v in zip(h["_step"], h[a.metric]) if s > 0 and (a.max_step is None or s <= a.max_step)}
    if not vals:
        sys.exit(f"FATAL: no {a.metric} values in {a.wandb_run}")

    art_root = fcs._normalize_s3_uri(a.artifact_s3)
    ckpt_root = fcs._normalize_s3_uri(a.ckpt_s3)
    art_bucket, art_prefix = fcs._split_s3_uri(art_root)
    ckpt_bucket, ckpt_prefix = fcs._split_s3_uri(ckpt_root)
    client = fcs._s3_client()

    def hf_files(step):
        if fcs._read_json_object(client, ckpt_bucket, f"{ckpt_prefix}/global_step_{step}/_REMOTE_COMPLETE.json", missing_ok=True) is None:
            return None
        src = f"{ckpt_prefix}/global_step_{step}/actor/huggingface/"
        files = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=ckpt_bucket, Prefix=src):
            files += [{"path": o["Key"][len(src):], "size": int(o["Size"])} for o in page.get("Contents", []) if not o["Key"].endswith("/")]
        return files if any(f["path"].endswith(".safetensors") for f in files) else None

    best = None
    for step in sorted(vals, key=lambda s: (-vals[s], s)):   # highest metric first, earliest on ties
        files = hf_files(step)
        if files:
            best = (step, vals[step], sorted(files, key=lambda f: f["path"]))
            break
    if best is None:
        sys.exit("FATAL: no validated step has a complete checkpoint with an HF export")
    step, score, files = best
    true_max_step = max(vals, key=lambda s: (vals[s], -s))
    print(f"best saved step {step} ({a.metric}={score:.4f}); true max {vals[true_max_step]:.4f} @ {true_max_step}; "
          f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB")

    old_outcome = fcs._read_json_object(client, art_bucket, f"{art_prefix}/{fcs._RUN_OUTCOME_NAME}", missing_ok=True) or {}
    if old_outcome.get("best_step") == step and fcs.check_best_hf_export(art_root) is not None:
        print("best_hf already holds this step and verifies -> nothing to do"); return
    if a.dry_run:
        print(f"DRY RUN: would replace best_hf (currently best_step={old_outcome.get('best_step')}) with global_step_{step}"); return

    # 1. invalidate first
    client.delete_object(Bucket=art_bucket, Key=f"{art_prefix}/{fcs._BEST_HF_PREFIX}/{fcs._BEST_HF_COMPLETE_NAME}")
    # 2. clear the old export
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=art_bucket, Prefix=f"{art_prefix}/{fcs._BEST_HF_PREFIX}/"):
        for o in page.get("Contents", []):
            client.delete_object(Bucket=art_bucket, Key=o["Key"])
    # 3. server-side copy of the chosen export
    src = f"{ckpt_prefix}/global_step_{step}/actor/huggingface/"
    for f in files:
        client.copy({"Bucket": ckpt_bucket, "Key": src + f["path"]}, art_bucket, f"{art_prefix}/{fcs._BEST_HF_PREFIX}/{f['path']}")
    # 4. outcome (keep the run-file's original once)
    if old_outcome and fcs._read_json_object(client, art_bucket, f"{art_prefix}/run_outcome.runfile.json", missing_ok=True) is None:
        client.put_object(Bucket=art_bucket, Key=f"{art_prefix}/run_outcome.runfile.json",
                          Body=(json.dumps(old_outcome, sort_keys=True, indent=2) + "\n").encode(), ContentType="application/json")
    outcome = dict(old_outcome)
    outcome.update({"best_step": step, "best_score": score, "early_stopping_metric": a.metric,
                    "best_hf_source": f"global_step_{step}/actor/huggingface",
                    "best_selection": "max validation over saved checkpoints (republish_best_hf.py)",
                    "republished_at": datetime.now(timezone.utc).isoformat()})
    outcome.setdefault("status", "complete")
    outcome.setdefault("final_step", max(vals))
    if a.stop_reason is not None:
        outcome["stop_reason"] = a.stop_reason
    if a.final_step is not None:
        outcome["final_step"] = a.final_step
    body = (json.dumps(outcome, sort_keys=True, indent=2) + "\n").encode()
    client.put_object(Bucket=art_bucket, Key=f"{art_prefix}/{fcs._RUN_OUTCOME_NAME}", Body=body, ContentType="application/json")
    client.put_object(Bucket=art_bucket, Key=f"{art_prefix}/{fcs._BEST_HF_PREFIX}/_RUN_OUTCOME.json", Body=body,
                      ContentType="application/json")
    # 5. completion marker, then verify with the run-file's own check
    manifest = {"schema_version": 1, "protocol": fcs._BEST_HF_PROTOCOL, "status": "complete", "checkpoint_s3_uri": art_root,
                "best_step": step, "final_step": int(outcome["final_step"]), "stop_reason": outcome.get("stop_reason"),
                "file_count": len(files), "total_bytes": sum(f["size"] for f in files), "files": files, "run_outcome": outcome}
    client.put_object(Bucket=art_bucket, Key=f"{art_prefix}/{fcs._BEST_HF_PREFIX}/{fcs._BEST_HF_COMPLETE_NAME}",
                      Body=(json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(), ContentType="application/json")
    verified = fcs.check_best_hf_export(art_root)
    print(f"BEST_HF_REPUBLISHED best_step={step} score={score:.4f} files={verified['file_count']} -> {art_root}/best_hf (verified)")


if __name__ == "__main__":
    main()
