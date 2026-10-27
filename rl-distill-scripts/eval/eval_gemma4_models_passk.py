#!/usr/bin/env python3
"""x32 band-validation pass@k (+ own-sample NLL) for a fixed list of Gemma 4 models: untrained bases and S3 HF exports.

Same protocol and grader as eval_student_checkpoints_passk.py (12-shot prompt, temp 1, 8192 tokens, strict last-\\boxed{}; the
id_<band> validation set, 300 q x 32 samples), whose materialize -> eval_math_passk path it reuses. That controller only knows Hub
step exports. Two differences:
  * models are given explicitly:  --model TAG=base:e4b  |  --model TAG=s3:<checkpoint root>/global_step_<N>  (an SFT-trainer
    permanent checkpoint with huggingface/ + _REMOTE_COMPLETE.json; architecture from --s3-arch)
  * --predictive-topk-width 128 (the width the x32 protocol pins; eval_math_passk accepts only that or 0 = no logprobs) records per
    token the top-128 predictive entropy (+ tail residual) and the sampled token's logprob, so every trace row carries
    token_entropy_{sum,count} and sampled_token_logprob_{sum,count}: entropy per token and own-sample NLL per token
Each (model, band) result goes to <s3-root>/<tag>__x32_<band>/ (metrics.json, traces/); finished ones are skipped.
    python eval_gemma4_models_passk.py --gpus 0,1,2,3 --parallelism dp --bands easy,medium,hard --s3-root s3://... \\
        --model e4b_base=base:e4b --model e4bp_s250=s3:s3://.../12b-base-all-to-e4b-.../global_step_250 --s3-arch e4b
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student_checkpoints_passk as ctl  # noqa: E402


def registry_entry(tag: str, spec: str, s3_arch: str) -> dict:
    kind, _, value = spec.partition(":")
    if kind == "base":
        architecture, repo, revision = ctl.ARCH[value]
        return {"tag": tag, "display_name": f"{architecture} base", "category": "base", "architecture": architecture, "trained_on": None,
                "math_datasets": ["id_easy", "id_medium", "id_hard", "math500", "gsm8k"],
                "source": {"type": "hf_snapshot", "repo_id": repo, "revision": revision}}
    if kind == "s3":
        root = value.rstrip("/")
        step = int(root.rsplit("global_step_", 1)[1])
        architecture, repo, revision = ctl.ARCH[s3_arch]
        # trained_on is registry metadata only (the band-union students were trained on all three bands)
        return {"tag": tag, "display_name": f"{architecture} distilled, {root.split('/')[-2]} step {step}", "category": "distilled",
                "architecture": architecture, "trained_on": "medium", "math_datasets": ["id_easy", "id_medium", "id_hard", "math500", "gsm8k"],
                "source": {"type": "s3_hf_export", "uri": f"{root}/huggingface/", "completion_uri": f"{root}/_REMOTE_COMPLETE.json",
                           "expected_global_step": step, "base_repo": repo, "base_revision": revision}}
    raise SystemExit(f"model spec must be base:<arch> or s3:<uri>, got {spec!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="TAG=base:<arch> | TAG=s3:<.../global_step_N>")
    ap.add_argument("--s3-arch", choices=list(ctl.ARCH), default="e4b", help="architecture of the s3: models")
    ap.add_argument("--bands", default="easy,medium,hard")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--parallelism", choices=["tp", "dp"], default="dp")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--s3-root", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--predictive-topk-width", type=int, default=128)
    ap.add_argument("--keep-materialized", action="store_true")
    args = ap.parse_args()
    args.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    i = ctl.SAMPLING.index("--predictive_topk_width")
    ctl.SAMPLING[i + 1] = str(args.predictive_topk_width)
    failed = []
    for item in args.model:
        tag, _, spec = item.partition("=")
        for band in [b.strip() for b in args.bands.split(",") if b.strip()]:
            run_tag = f"{tag}__x32_{band}"
            if subprocess.run([ctl.AWS, "s3", "ls", f"{args.s3_root.rstrip('/')}/{run_tag}/metrics.json"], capture_output=True).returncode == 0:
                print(f"[{run_tag}] already on S3 -> skip", flush=True)
                continue
            step_root = args.out_root / run_tag
            step_root.mkdir(parents=True, exist_ok=True)
            entry = registry_entry(run_tag, spec, args.s3_arch)
            registry = {"schema_version": 1, "protocol": "gemma4_rl_distill_eval_sources_v1", "study": "gemma4-base-all-predistill",
                        "models": [entry]}
            (step_root / "source_registry.json").write_text(json.dumps(registry, indent=2))
            if not ctl._evaluate_tag(run_tag, band, args, step_root):
                failed.append(run_tag)
                # the shard logs live on the pod's local disk only: put their tails into the job log, then stop
                for log in sorted(step_root.glob("dp*/eval.log")) + [step_root / "eval.log"]:
                    if log.exists():
                        print(f"----- tail of {log}", flush=True)
                        print("\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
                print(f"EVAL_LIST_DONE failed={failed} (stopped at the first failure)", flush=True)
                return 1
    print(f"EVAL_LIST_DONE failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
