#!/usr/bin/env python3
"""Evaluate every pushed checkpoint of distilled students with the 32-sample validation protocol and plot pass@k.

For each ``--repo JWei05/Distill-gemma4-e4b-base-<band>-to-<student>-base`` the script lists the Hub tree,
and for every ``step_NNNNNN/`` export not yet evaluated it (1) pins the repo's current ``main`` commit,
(2) materializes the checkpoint through ``data/materialize_gemma4_eval_models.py`` (identity SHA, base
metadata fill-in), (3) runs ``eval_math_passk.py`` on the band's 300-question validation set with the
``gemma4_rl_distill_math_eval_v2_x32`` manifest (32 samples/q, no logprobs), and (4) re-plots the pass@k
curves of all evaluated steps against the E4B-base reference trace. ``--poll-minutes 0`` runs once;
otherwise it keeps polling (checkpoints appear every 250 training steps).

    python rl-distill-scripts/eval_student_checkpoints_passk.py --gpu 6 --poll-minutes 10 \
        --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --repo JWei05/Distill-gemma4-e4b-base-hard-to-12b-base
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "rl-distill-scripts"
REPO_NAME = re.compile(r"^JWei05/(?:Distill-gemma4|gemma4-distill-v2)-e4b-base-(?P<band>easy|medium|hard)-to-(?P<student>12b|26b|e4b|e2b)-base$")
ARCH = {"12b": ("gemma-4-12B", "google/gemma-4-12B", "023679ed352de9bb66cc873c9009ce3482585c08"),
        "26b": ("gemma-4-26B-A4B", "google/gemma-4-26B-A4B", "24548b62aa021d562695c04aaf7758a1ea47990b"),
        "e4b": ("gemma-4-E4B", "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"),
        "e2b": ("gemma-4-E2B", "google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")}
SAMPLING = ["--temperature", "1.0", "--top_k", "-1", "--top_p", "1.0", "--max_tokens", "8192", "--max_prompt_tokens", "4096",
            "--max_model_len", "12288", "--predictive_topk_width", "0", "--request_batch_size", "2048", "--questions_per_batch", "64",
            "--subset_strategy", "monte_carlo", "--monte_carlo_resamples", "4096", "--ks", "1", "2", "4", "8", "16", "32"]


def sh(cmd: list[str], env: dict | None = None, log: Path | None = None) -> int:
    print("+", " ".join(cmd)[:300], flush=True)
    with (log.open("a") if log else open(os.devnull, "w")) as handle:
        proc = subprocess.run(cmd, env=env, stdout=handle if log else None, stderr=subprocess.STDOUT if log else None, text=True)
    return proc.returncode


def evaluate_step(api: HfApi, repo: str, step: str, band: str, student: str, args, out_root: Path) -> bool:
    tag = f"{repo.split('/')[-1]}__{step}"
    step_root = out_root / tag
    if (step_root / "metrics.json").exists():
        return True
    step_root.mkdir(parents=True, exist_ok=True)
    commit = api.list_repo_commits(repo)[0].commit_id
    architecture, meta_repo, meta_rev = ARCH[student]
    registry = {"schema_version": 1, "protocol": "gemma4_rl_distill_eval_sources_v1", "study": "gemma4-e4b-base-control",
                "models": [{"tag": tag, "display_name": f"{repo} {step}", "category": "distilled", "architecture": architecture,
                            "trained_on": band, "math_datasets": [f"id_{band}"],
                            "source": {"type": "hf_subfolder", "repo_id": repo, "revision": commit, "subfolder": step,
                                       "metadata_repo": meta_repo, "metadata_revision": meta_rev}}]}
    (step_root / "source_registry.json").write_text(json.dumps(registry, indent=2))
    log = step_root / "eval.log"
    if sh([sys.executable, str(SCRIPTS / "data/materialize_gemma4_eval_models.py"), "--source-registry", str(step_root / "source_registry.json"),
           "--output-root", str(step_root / "models"), "--models", tag, "--execute"], log=log) != 0:
        print(f"[{tag}] materialize FAILED (see {log})", flush=True); return False
    resolved = json.loads((step_root / "models" / "resolved_model_registry.json").read_text())
    models = resolved["models"] if isinstance(resolved, dict) and "models" in resolved else resolved
    entry = next(m for m in (models if isinstance(models, list) else models.values()) if m["tag"] == tag)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), VERL_MATH_VERIFY_STRICT_BOXED="1", VERL_MATH_VERIFY_TIMEOUT="30", VERL_MATH_SYMPY_TIMEOUT="5.0")
    cmd = [sys.executable, str(SCRIPTS / "eval_math_passk.py"), "--model", entry["model"], "--expected_model_identity_sha256", entry["expected_model_identity_sha256"],
           "--tag", tag, "--chat_template", str(SCRIPTS / "data/gemma3_it_fewshot_math.jinja"), "--datasets", str(args.manifest.parent / f"id_{band}.parquet"),
           "--dataset_manifest", str(args.manifest), "--out", str(step_root / "metrics.json"), "--trace_dir", str(step_root / "traces"),
           "--tensor_parallel_size", "1", "--gpu_memory_utilization", str(args.gpu_memory_utilization), *SAMPLING]
    if sh(cmd, env=env, log=log) != 0 or not (step_root / "metrics.json").exists():
        print(f"[{tag}] eval FAILED (see {log})", flush=True); return False
    metrics = json.loads((step_root / "metrics.json").read_text())["results"][f"id_{band}"]
    print(f"[{tag}] id_{band}: mean@32={metrics['mean@k']} pass@32={metrics['pass@k']} maj@32={metrics['maj@k']}", flush=True)
    return True


def plot_repo(repo: str, band: str, out_root: Path, reference_root: Path, figures: Path) -> None:
    short = repo.split("/")[-1]
    traces = [f"E4B base (teacher)={reference_root / f'id_{band}' / 'traces' / f'base_e4b__id_{band}.jsonl'}"]
    for step_dir in sorted(out_root.glob(f"{short}__step_*")):
        trace = step_dir / "traces" / f"{step_dir.name}__id_{band}.jsonl"
        if trace.exists():
            traces.append(f"{short.replace('Distill-gemma4-', '')} {step_dir.name.split('__')[-1]}={trace}")
    if len(traces) < 2 or not Path(traces[0].split("=", 1)[1]).exists():
        return
    out = figures / f"passk_{short.replace('Distill-gemma4-', '').replace('gemma4-distill-v2-', '')}_val32.png"
    sh([sys.executable, str(SCRIPTS / "plot_passk_from_traces.py"), "--out", str(out), "--title",
        f"{short}: validation pass@k (32 samples/q, 300 q, {band} band) vs the E4B base teacher", *sum((["--trace", t] for t in traces), [])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--poll-minutes", type=float, default=10.0, help="0 = single pass")
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/gemma4_e4b_val32/data/math_eval_manifest.json"))
    parser.add_argument("--reference-root", type=Path, default=Path("/tmp/gemma4_e4b_val32"), help="E4B base x32 results (id_<band>/traces)")
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/gemma4_e4b_val32/students"))
    parser.add_argument("--figures", type=Path, default=SCRIPTS / "figures")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    api = HfApi()
    while True:
        for repo in args.repo:
            m = REPO_NAME.match(repo)
            if not m:
                raise SystemExit(f"repo name not recognised: {repo}")
            band, student = m["band"], m["student"]
            try:
                steps = sorted(e.path for e in api.list_repo_tree(repo, revision="main") if e.path.startswith("step_"))
            except RepositoryNotFoundError:
                print(f"[{repo}] not on the Hub yet", flush=True); continue
            new = [s for s in steps if not (args.out_root / f"{repo.split('/')[-1]}__{s}" / "metrics.json").exists()]
            print(f"[{repo}] steps={steps} new={new} {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
            for step in new:
                if evaluate_step(api, repo, step, band, student, args, args.out_root):
                    plot_repo(repo, band, args.out_root, args.reference_root, args.figures)
        if args.poll_minutes <= 0:
            return 0
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
