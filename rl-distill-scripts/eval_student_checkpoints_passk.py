#!/usr/bin/env python3
"""Evaluate every pushed checkpoint of distilled students with the 32-sample validation protocol and plot pass@k.

For each ``--repo JWei05/Distill-gemma4-<teacher>-<band>-to-<student>-base`` the script lists the Hub tree,
and for every ``step_NNNNNN/`` export not yet evaluated it (1) pins the repo's current ``main`` commit,
(2) materializes the checkpoint through ``data/materialize_gemma4_eval_models.py`` (identity SHA, base
metadata fill-in), (3) runs ``eval_math_passk.py`` on the band's 300-question validation set with the
``gemma4_rl_distill_math_eval_v2_x32`` manifest (32 samples/q, no logprobs) and (4) re-plots the pass@k
curves of all evaluated steps against the E4B-base reference trace.

GPU layout (``--gpus 0,1``): ``--parallelism tp`` runs one vLLM instance tensor-parallel over the GPUs
(26B-A4B); ``--parallelism dp`` runs one single-GPU instance per GPU, each on an interleaved shard of the
questions, then merges the shard traces and re-aggregates them with ``--resume_traces`` (12B). The merge is
exact: sampling seeds derive from (dataset name, question id, sample index), not from row order, and the
shard manifests keep the protocol's fixed 32 samples/question.

ScaleTrain pod (``scale_train/run_gemma4_student_ckpt_passk_st.sh``): with ``--s3-root`` every finished
step is uploaded (metrics, traces, log -- not the materialized weights, which are deleted after the eval);
the process exits once ``--final-step`` is evaluated for every repo, or after ``--max-idle-hours`` without
a new checkpoint. Plots need the E4B-base reference traces, so they are drawn on the box that holds them:
``--plot-from-s3`` (no GPU) syncs the finished steps down from ``--s3-root`` and re-plots the figures.

    # pod
    python rl-distill-scripts/eval_student_checkpoints_passk.py --gpus 0,1 --parallelism dp --poll-minutes 10 \
        --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --final-step 1000 \
        --s3-root s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1
    # local plots
    python rl-distill-scripts/eval_student_checkpoints_passk.py --plot-from-s3 --poll-minutes 10 \
        --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --repo JWei05/Distill-gemma4-e4b-base-medium-to-26b-base \
        --s3-root s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "rl-distill-scripts"
REPO_NAME = re.compile(
    r"^JWei05/(?:Distill-gemma4|gemma4-distill-v2)-(?P<teacher>26b|12b|e4b|e2b|e4b-base)-(?P<band>easy|medium|hard)"
    r"-to-(?P<student>12b|26b|e4b|e2b)-base$"
)
ARCH = {"12b": ("gemma-4-12B", "google/gemma-4-12B", "023679ed352de9bb66cc873c9009ce3482585c08"),
        "26b": ("gemma-4-26B-A4B", "google/gemma-4-26B-A4B", "24548b62aa021d562695c04aaf7758a1ea47990b"),
        "e4b": ("gemma-4-E4B", "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"),
        "e2b": ("gemma-4-E2B", "google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f")}
SAMPLING = ["--temperature", "1.0", "--top_k", "-1", "--top_p", "1.0", "--max_tokens", "8192", "--max_prompt_tokens", "4096",
            "--max_model_len", "12288", "--predictive_topk_width", "0", "--request_batch_size", "2048", "--questions_per_batch", "64",
            "--subset_strategy", "monte_carlo", "--monte_carlo_resamples", "4096", "--ks", "1", "2", "4", "8", "16", "32"]
GRADER_ENV = {"VERL_MATH_VERIFY_STRICT_BOXED": "1", "VERL_MATH_VERIFY_TIMEOUT": "30", "VERL_MATH_SYMPY_TIMEOUT": "5.0"}
AWS = shutil.which("aws") or str(REPO_ROOT / ".venv/bin/aws")


def step_tag(repo: str, step: str) -> str:
    """Registry-safe tag (lowercase letters, digits, underscores): distill_gemma4_e4b_base_medium_to_12b_base__step_000250."""
    return f"{re.sub(r'[^a-z0-9_]', '_', repo.split('/')[-1].lower())}__{step}"


def sh(cmd: list[str], env: dict | None = None, log: Path | None = None) -> int:
    print("+", " ".join(cmd)[:300], flush=True)
    with (log.open("a") if log else open(os.devnull, "w")) as handle:
        proc = subprocess.run(cmd, env=env, stdout=handle if log else None, stderr=subprocess.STDOUT if log else None, text=True)
    return proc.returncode


def eval_cmd(entry: dict, tag: str, dataset: Path, manifest: Path, out: Path, trace_dir: Path, tp: int, args) -> list[str]:
    return [sys.executable, str(SCRIPTS / "eval_math_passk.py"), "--model", entry["model"],
            "--expected_model_identity_sha256", entry["expected_model_identity_sha256"], "--tag", tag,
            "--chat_template", str(SCRIPTS / "data/gemma3_it_fewshot_math.jinja"), "--datasets", str(dataset),
            "--dataset_manifest", str(manifest), "--out", str(out), "--trace_dir", str(trace_dir),
            "--tensor_parallel_size", str(tp), "--gpu_memory_utilization", str(args.gpu_memory_utilization), *SAMPLING]


def shard_dataset(manifest_path: Path, band: str, step_root: Path, shards: int) -> list[tuple[Path, Path]]:
    """Interleaved question shards of id_<band> with a per-shard protocol manifest (same dataset name -> same seeds)."""
    import pandas as pd

    manifest = json.loads(manifest_path.read_text())
    entry = next(d for d in manifest["datasets"] if d["name"] == f"id_{band}")
    frame = pd.read_parquet(manifest_path.parent / f"id_{band}.parquet")
    if len(frame) != int(entry["unique_questions"]):
        raise SystemExit(f"id_{band}: {len(frame)} rows but {entry['unique_questions']} unique questions; cannot shard by row")
    out = []
    for k in range(shards):
        shard_dir = step_root / f"dp{k}" / "data"
        shard_dir.mkdir(parents=True, exist_ok=True)
        parquet = shard_dir / f"id_{band}.parquet"
        shard = frame.iloc[k::shards].reset_index(drop=True)
        shard.to_parquet(parquet, index=False)
        shard_entry = dict(entry, output_path=str(parquet), output_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
                           unique_questions=len(shard), total_requests=len(shard) * int(entry["samples_per_question"]))
        shard_manifest = shard_dir / "math_eval_manifest.json"
        shard_manifest.write_text(json.dumps(dict(manifest, datasets=[shard_entry]), indent=2))
        out.append((parquet, shard_manifest))
    return out


def evaluate_step(api: HfApi, repo: str, step: str, band: str, student: str, args, out_root: Path) -> bool:
    tag = step_tag(repo, step)
    step_root = out_root / tag
    if (step_root / "metrics.json").exists():
        return True
    step_root.mkdir(parents=True, exist_ok=True)
    commit = api.list_repo_commits(repo)[0].commit_id
    architecture, meta_repo, meta_rev = ARCH[student]
    registry = {"schema_version": 1, "protocol": "gemma4_rl_distill_eval_sources_v1", "study": "gemma4-e4b-base-control",
                "models": [{"tag": tag, "display_name": f"{repo} {step}", "category": "distilled", "architecture": architecture,
                            # the registry validator wants the own band plus MATH500/GSM8K listed; only id_<band> is run here
                            "trained_on": band, "math_datasets": [f"id_{band}", "math500", "gsm8k"],
                            "source": {"type": "hf_subfolder", "repo_id": repo, "revision": commit, "subfolder": step,
                                       "metadata_repo": meta_repo, "metadata_revision": meta_rev}}]}
    (step_root / "source_registry.json").write_text(json.dumps(registry, indent=2))
    return _evaluate_tag(tag, band, args, step_root)


def _evaluate_tag(tag: str, band: str, args, step_root: Path) -> bool:
    log = step_root / "eval.log"
    if sh([sys.executable, str(SCRIPTS / "data/materialize_gemma4_eval_models.py"), "--source-registry", str(step_root / "source_registry.json"),
           "--output-root", str(step_root / "models"), "--models", tag, "--execute"], log=log) != 0:
        print(f"[{tag}] materialize FAILED (see {log})", flush=True)
        return False
    resolved = json.loads((step_root / "models" / "resolved_model_registry.json").read_text())
    models = resolved["models"] if isinstance(resolved, dict) and "models" in resolved else resolved
    entry = next(m for m in (models if isinstance(models, list) else models.values()) if m["tag"] == tag)
    dataset = args.manifest.parent / f"id_{band}.parquet"
    ok = False
    if args.parallelism == "tp":
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(args.gpus), **GRADER_ENV)
        ok = sh(eval_cmd(entry, tag, dataset, args.manifest, step_root / "metrics.json", step_root / "traces", len(args.gpus), args),
                env=env, log=log) == 0
    else:
        procs = []
        for k, (shard, shard_manifest) in enumerate(shard_dataset(args.manifest, band, step_root, len(args.gpus))):
            shard_root = step_root / f"dp{k}"
            cmd = eval_cmd(entry, tag, shard, shard_manifest, shard_root / "metrics.json", shard_root / "traces", 1, args)
            print("+", " ".join(cmd)[:300], flush=True)
            handle = (shard_root / "eval.log").open("a")
            procs.append((handle, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[k], **GRADER_ENV),
                                                   stdout=handle, stderr=subprocess.STDOUT, text=True)))
        codes = [proc.wait() for _, proc in procs]
        for handle, _ in procs:
            handle.close()
        if all(code == 0 for code in codes):
            merged = step_root / "traces" / f"{tag}__id_{band}.jsonl"
            merged.parent.mkdir(exist_ok=True)
            with merged.open("w", encoding="utf-8") as out:
                for k in range(len(args.gpus)):
                    out.write((step_root / f"dp{k}" / "traces" / f"{tag}__id_{band}.jsonl").read_text(encoding="utf-8"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[0], **GRADER_ENV)
            ok = sh(eval_cmd(entry, tag, dataset, args.manifest, step_root / "metrics.json", step_root / "traces", 1, args)
                    + ["--resume_traces"], env=env, log=log) == 0   # re-aggregates the merged traces; no model load
        else:
            print(f"[{tag}] shard evals exited {codes} (see {step_root}/dp*/eval.log)", flush=True)
    if not ok or not (step_root / "metrics.json").exists():
        print(f"[{tag}] eval FAILED (see {log})", flush=True)
        return False
    metrics = json.loads((step_root / "metrics.json").read_text())["results"][f"id_{band}"]
    print(f"[{tag}] id_{band}: mean@32={metrics['mean@k']} pass@32={metrics['pass@k']} maj@32={metrics['maj@k']}", flush=True)
    if not args.keep_materialized:
        shutil.rmtree(step_root / "models", ignore_errors=True)
    if args.s3_root:
        if sh([AWS, "s3", "sync", str(step_root), f"{args.s3_root.rstrip('/')}/{tag}/", "--only-show-errors",
               "--exclude", "models/*", "--exclude", "dp*/models/*", "--exclude", "dp*/data/*"]) != 0:
            print(f"[{tag}] S3 upload FAILED (kept locally; retried on the next pass)", flush=True)
            (step_root / "metrics.json").rename(step_root / "metrics.unsynced.json")
            return False
    return True


def base_tag(student: str, band: str) -> str:
    return f"base_{student}__x32_{band}"


def evaluate_base(student: str, band: str, args, out_root: Path) -> bool:
    """Evaluate the untrained student base (pinned snapshot) with the same x32 protocol -> reference curve."""
    tag = base_tag(student, band)
    step_root = out_root / tag
    if (step_root / "metrics.json").exists():
        return True
    step_root.mkdir(parents=True, exist_ok=True)
    architecture, repo, revision = ARCH[student]
    registry = {"schema_version": 1, "protocol": "gemma4_rl_distill_eval_sources_v1", "study": "gemma4-e4b-base-control",
                "models": [{"tag": tag, "display_name": f"{architecture} base (no distillation)", "category": "base", "architecture": architecture,
                            "trained_on": None, "math_datasets": ["id_easy", "id_medium", "id_hard", "math500", "gsm8k"],
                            "source": {"type": "hf_snapshot", "repo_id": repo, "revision": revision}}]}
    (step_root / "source_registry.json").write_text(json.dumps(registry, indent=2))
    return _evaluate_tag(tag, band, args, step_root)


def sync_from_s3(args) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    # --delete: a result the submitter parked under _superseded/ (export re-pushed by a relaunched run) disappears locally too
    sh([AWS, "s3", "sync", args.s3_root.rstrip("/") + "/", str(args.out_root) + "/", "--only-show-errors", "--delete",
        "--exclude", "*/eval.log", "--exclude", "*/dp*/*", "--exclude", "_superseded/*"])


def plot_repo(repo: str, band: str, out_root: Path, reference_root: Path, figures: Path) -> None:
    short = repo.split("/")[-1]
    traces = [f"E4B base (teacher)={reference_root / f'id_{band}' / 'traces' / f'base_e4b__id_{band}.jsonl'}"]
    student = REPO_NAME.match(repo)["student"]
    base_trace = out_root / base_tag(student, band) / "traces" / f"{base_tag(student, band)}__id_{band}.jsonl"
    if base_trace.exists():
        traces.append(f"{student.upper()} base (no distillation)={base_trace}")
    for step_dir in sorted(out_root.glob(step_tag(repo, "step_*"))):
        trace = step_dir / "traces" / f"{step_dir.name}__id_{band}.jsonl"
        if trace.exists() and (step_dir / "metrics.json").exists():
            traces.append(f"{short.replace('Distill-gemma4-', '')} {step_dir.name.split('__')[-1]}={trace}")
    if len(traces) < 2 or not Path(traces[0].split("=", 1)[1]).exists():
        return
    out = figures / f"passk_{short.replace('Distill-gemma4-', '').replace('gemma4-distill-v2-', '')}_val32.png"
    sh([sys.executable, str(SCRIPTS / "plot_passk_from_traces.py"), "--out", str(out), "--title",
        f"{short}: validation pass@k (32 samples/q, 300 q, {band} band) vs the E4B base teacher", *sum((["--trace", t] for t in traces), [])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids for the eval (e.g. 0,1); required unless --plot-from-s3")
    parser.add_argument("--parallelism", choices=["tp", "dp"], default="tp",
                        help="tp: one vLLM instance tensor-parallel over --gpus (26B-A4B); dp: one instance per GPU on question shards (12B)")
    parser.add_argument("--poll-minutes", type=float, default=10.0, help="0 = single pass")
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/gemma4_e4b_val32/data/math_eval_manifest.json"))
    parser.add_argument("--reference-root", type=Path, default=Path("/tmp/gemma4_e4b_val32"), help="E4B base x32 results (id_<band>/traces)")
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/gemma4_e4b_val32/students"))
    parser.add_argument("--figures", type=Path, default=SCRIPTS / "figures")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--s3-root", default=None, help="upload every finished step to <s3-root>/<repo>__<step>/ (pod) / source of --plot-from-s3")
    parser.add_argument("--plot-from-s3", action="store_true", help="no GPU: sync finished steps from --s3-root and re-plot")
    parser.add_argument("--no-plot", action="store_true", help="evaluate only (pods have no reference traces)")
    parser.add_argument("--keep-materialized", action="store_true", help="keep the materialized checkpoint weights after the eval")
    parser.add_argument("--base", choices=list(ARCH), default=None,
                        help="instead of Hub exports, evaluate this untrained base with the same protocol (band from the single --repo)")
    parser.add_argument("--step", default=None, help="evaluate only this export (e.g. step_000250) of the single --repo; "
                        "exit 1 if it is not on the Hub (one ScaleTrain job per checkpoint)")
    parser.add_argument("--final-step", type=int, default=None, help="exit once step_<N> is evaluated for every repo")
    parser.add_argument("--max-idle-hours", type=float, default=None, help="exit after this long without a new checkpoint")
    args = parser.parse_args()
    if args.plot_from_s3 and not args.s3_root:
        parser.error("--plot-from-s3 requires --s3-root")
    if not args.plot_from_s3:
        if not args.gpus:
            parser.error("--gpus is required for evaluation")
        args.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if args.parallelism == "dp" and len(args.gpus) < 2:
            parser.error("--parallelism dp needs at least two GPUs")
    if args.step and (len(args.repo) != 1 or args.plot_from_s3):
        parser.error("--step takes exactly one --repo and no --plot-from-s3")
    if args.base:
        m = REPO_NAME.match(args.repo[0])
        if len(args.repo) != 1 or not m or args.plot_from_s3:
            parser.error("--base takes exactly one --repo (for the band) and no --plot-from-s3")
        ok = evaluate_base(args.base, m["band"], args, args.out_root)
        print(f"base {args.base}: {'evaluated' if ok else 'FAILED'}", flush=True)
        return 0 if ok else 1
    api = HfApi()
    last_progress = time.time()
    while True:
        progressed, reached_final = False, bool(args.final_step)
        if args.plot_from_s3:
            before = {p.name for p in args.out_root.glob("*__step_*") if (p / "metrics.json").exists()}
            sync_from_s3(args)
        for repo in args.repo:
            m = REPO_NAME.match(repo)
            if not m:
                raise SystemExit(f"repo name not recognised: {repo}")
            band, student = m["band"], m["student"]
            if args.plot_from_s3:
                done = sorted(p.name.split("__")[-1] for p in args.out_root.glob(step_tag(repo, "step_*")) if (p / "metrics.json").exists())
                print(f"[{repo}] synced steps={done} {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
                if any(step_tag(repo, s) not in before for s in done):
                    progressed = True
                    plot_repo(repo, band, args.out_root, args.reference_root, args.figures)
            else:
                try:
                    steps = sorted(e.path for e in api.list_repo_tree(repo, revision="main") if e.path.startswith("step_"))
                except RepositoryNotFoundError:
                    print(f"[{repo}] not on the Hub yet", flush=True)
                    steps = []
                if args.step:
                    if args.step not in steps:
                        raise SystemExit(f"{repo}: {args.step} is not on the Hub (have {steps})")
                    steps = [args.step]
                new = [s for s in steps if not (args.out_root / step_tag(repo, s) / "metrics.json").exists()]
                print(f"[{repo}] steps={steps} new={new} {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
                for step in new:
                    if evaluate_step(api, repo, step, band, student, args, args.out_root):
                        progressed = True
                        if not args.no_plot:
                            plot_repo(repo, band, args.out_root, args.reference_root, args.figures)
            if args.final_step and not (args.out_root / step_tag(repo, f"step_{args.final_step:06d}") / "metrics.json").exists():
                reached_final = False
        if progressed:
            last_progress = time.time()
        if reached_final:
            print(f"final step {args.final_step} evaluated for every repo; done", flush=True)
            return 0
        if args.step:
            done = (args.out_root / step_tag(args.repo[0], args.step) / "metrics.json").exists()
            print(f"{args.step}: {'evaluated' if done else 'FAILED'}", flush=True)
            return 0 if done else 1
        if args.poll_minutes <= 0:
            return 0
        if args.max_idle_hours and time.time() - last_progress > args.max_idle_hours * 3600:
            print(f"no new checkpoint for {args.max_idle_hours} h; exiting", flush=True)
            return 0
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
