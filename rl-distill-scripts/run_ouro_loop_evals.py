#!/usr/bin/env python3
"""Local Ouro looped-LM math evals: {1.4B, 2.6B} x {1, 4} recurrent steps x {12-shot, 0-shot}.

Each (config, dataset) pair is one eval_math_passk.py process on the three 300-question DeepScaleR
band validation sets, 16 samples/question, the standard math protocol (temp 1.0, top_p 1.0, 8192 new
tokens, boxed-only math_verify). Loop count is set through vLLM ``hf_overrides={"total_ut_steps": N}``;
4 is Ouro's shipped (intended) setting, early_exit_threshold=1.0 = always run all steps. Prompts are the
repo's 12-shot math examples rendered in Ouro's native ChatML (data/ouro_chatml_{fewshot,0shot}_math.jinja).

Scheduler: every GPU has two half-slots; a 4-step task takes a whole GPU (KV cache is 4x per token),
a 1-step task takes one half. Heaviest tasks start first; finished (config, dataset) metrics are skipped
on restart, and a restarted driver adopts still-running tasks from their ``.running`` markers.

Each task streams finished samples to ``<tag>/stream/``; ``--report`` grades them incrementally (same
boxed-only ``math_verify`` as the final metrics) so unfinished tasks show a live mean accuracy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "1p4b": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
    "2p6b": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
}
PROMPTS = {
    "fewshot12": "rl-distill-scripts/data/ouro_chatml_fewshot_math.jinja",
    "0shot": "rl-distill-scripts/data/ouro_chatml_0shot_math.jinja",
}
LOOPS = (4, 1)
DATASETS = ("id_easy", "id_medium", "id_hard")
CHATML_STOP_TOKEN_IDS = ("2", "1")  # <|im_end|>, <|im_start|>
KV_GIB = {1: 28, 2: 62}  # by half-slots used


def build_tasks(data_dir: Path, out_root: Path) -> list[dict]:
    tasks = []
    for size, loops, prompt, dataset in itertools.product(("2p6b", "1p4b"), LOOPS, PROMPTS, DATASETS):
        tag = f"ouro{size}_ut{loops}_{prompt}"
        repo, revision = MODELS[size]
        units = 2 if loops == 4 else 1
        metrics = out_root / tag / f"metrics_{dataset}.json"
        command = [
            sys.executable, "rl-distill-scripts/eval_math_passk.py",
            "--model", repo, "--model_revision", revision,
            "--tag", tag, "--chat_template", PROMPTS[prompt],
            "--datasets", str(data_dir / f"{dataset}.parquet"),
            "--samples_per_question", "16", "--allow_nonstandard_sample_count",
            "--ks", "1", "2", "4", "8", "16",
            "--subset_strategy", "monte_carlo", "--monte_carlo_resamples", "4096",
            "--out", str(metrics), "--trace_dir", str(out_root / tag / "traces"),
            "--stream_dir", str(out_root / tag / "stream"),
            "--predictive_topk_width", "0",
            "--questions_per_batch", "300", "--request_batch_size", "4800",
            "--kv_cache_memory_gib", str(KV_GIB[units]),
            "--stop_token_ids", *CHATML_STOP_TOKEN_IDS,
            "--hf_overrides", f'{{"total_ut_steps": {loops}}}',
        ]
        tasks.append({"name": f"{tag}__{dataset}", "units": units, "metrics": metrics, "command": command})
    return tasks


def graded_stream(stream: Path) -> list[dict]:
    """Grade the samples a running task has streamed so far; grades are cached next to the stream."""
    sys.path.insert(0, str(REPO))
    from verl.utils.reward_score import math_verify

    cache = stream.with_suffix(".graded.jsonl")
    graded = [json.loads(line) for line in cache.open()] if cache.exists() else []
    with stream.open() as handle, cache.open("a") as out:
        for index, line in enumerate(handle):
            if index < len(graded) or not line.endswith("\n"):
                continue
            sample = json.loads(line)
            row = {
                "question_id": sample["question_id"],
                "acc": float(math_verify.compute_score(sample["response_text"], sample["gold"])) > 0.5,
                "response_length": sample["response_length"],
            }
            out.write(json.dumps(row) + "\n")
            graded.append(row)
    return graded


def report(out_root: Path) -> str:
    """Markdown table: mean@16 / pass@16 per band, plus response length and 8192-token truncation rate.

    Unfinished tasks show ``*live mean acc (graded samples/4800)*`` from their streamed samples.
    """
    lines = [
        "| Model | Loops | Prompt | id_easy | id_medium | id_hard | mean resp len | truncated % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size, loops, prompt in itertools.product(("1p4b", "2p6b"), (1, 4), PROMPTS):
        tag = f"ouro{size}_ut{loops}_{prompt}"
        cells, lengths = [], []
        for dataset in DATASETS:
            metrics = out_root / tag / f"metrics_{dataset}.json"
            stream = out_root / tag / "stream" / f"{tag}__{dataset}.stream.jsonl"
            if not metrics.exists():
                graded = graded_stream(stream) if stream.exists() else []
                if graded:
                    live_acc = 100 * sum(row["acc"] for row in graded) / len(graded)
                    cells.append(f"*{live_acc:.1f} ({len(graded)}/4800)*")
                    lengths += [row["response_length"] for row in graded]
                else:
                    cells.append("—")
                continue
            result = json.loads(metrics.read_text())["results"][dataset]
            cells.append(f"{result['mean@k']:.1f} / {result['pass@k']:.1f}")
            traces = out_root / tag / "traces" / f"{tag}__{dataset}.jsonl"
            lengths += [json.loads(line)["response_length"] for line in traces.open()]
        length_cells = (
            [f"{sum(lengths) / len(lengths):.0f}", f"{100 * sum(n >= 8192 for n in lengths) / len(lengths):.1f}"]
            if lengths
            else ["—", "—"]
        )
        model = {"1p4b": "Ouro-1.4B", "2p6b": "Ouro-2.6B"}[size]
        loop_label = f"{loops} (intended)" if loops == 4 else str(loops)
        prompt_label = {"fewshot12": "12-shot", "0shot": "0-shot"}[prompt]
        lines.append(f"| {model} | {loop_label} | {prompt_label} | " + " | ".join(cells + length_cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/ouro_eval/data"))
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/ouro_eval/results"))
    parser.add_argument("--report", action="store_true", help="print the results table and exit")
    parser.add_argument("--doc", type=Path, help="with --report: also write the table between the OURO_RESULTS_TABLE markers")
    args = parser.parse_args()
    if args.report:
        table = report(args.out_root)
        print(table)
        if args.doc:
            text = args.doc.read_text()
            head, rest = text.split("<!-- OURO_RESULTS_TABLE -->\n", 1)
            tail = rest.split("<!-- /OURO_RESULTS_TABLE -->", 1)[1]
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            body = f"_Updated {stamp}. Cells: mean@16 / pass@16 (%); italics = live mean accuracy of samples graded so far._\n\n"
            args.doc.write_text(f"{head}<!-- OURO_RESULTS_TABLE -->\n{body}{table}\n<!-- /OURO_RESULTS_TABLE -->{tail}")
        return

    pending = [t for t in build_tasks(args.data_dir, args.out_root) if not t["metrics"].exists()]
    free = {gpu: 2 for gpu in args.gpus}
    running: list[tuple[subprocess.Popen | int, dict, int]] = []  # Popen, or the pid of an adopted task
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for task in list(pending):
        marker = task["metrics"].with_suffix(".running")
        if not marker.exists():
            continue
        state = json.loads(marker.read_text())
        if Path(f"/proc/{state['pid']}").exists():
            pending.remove(task)
            free[state["gpu"]] -= task["units"]
            running.append((state["pid"], task, state["gpu"]))
            print(f"{time.strftime('%FT%TZ', time.gmtime())} ADOPT {task['name']} gpu{state['gpu']} pid={state['pid']}", flush=True)
    print(f"{time.strftime('%FT%TZ', time.gmtime())} PLAN {len(pending)} tasks on GPUs {args.gpus}", flush=True)
    failed = []
    while pending or running:
        for proc, task, gpu in list(running):
            if isinstance(proc, int):
                if Path(f"/proc/{proc}").exists():
                    continue
                returncode = 0 if task["metrics"].exists() else None
            elif (returncode := proc.poll()) is None:
                continue
            running.remove((proc, task, gpu))
            free[gpu] += task["units"]
            task["metrics"].with_suffix(".running").unlink(missing_ok=True)
            status = "DONE" if returncode == 0 and task["metrics"].exists() else f"FAIL rc={returncode}"
            if status != "DONE":
                failed.append(task["name"])
            print(f"{time.strftime('%FT%TZ', time.gmtime())} {status} {task['name']} gpu{gpu}", flush=True)
        for task in list(pending):
            gpu = next((g for g in args.gpus if free[g] >= task["units"]), None)
            if gpu is None:
                continue
            free[gpu] -= task["units"]
            pending.remove(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log = open(log_dir / f"{task['name']}.log", "w")
            proc = subprocess.Popen(task["command"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            task["metrics"].parent.mkdir(parents=True, exist_ok=True)
            task["metrics"].with_suffix(".running").write_text(json.dumps({"pid": proc.pid, "gpu": gpu}))
            running.append((proc, task, gpu))
            print(f"{time.strftime('%FT%TZ', time.gmtime())} START {task['name']} gpu{gpu} units={task['units']}", flush=True)
        time.sleep(10)
    print(f"{time.strftime('%FT%TZ', time.gmtime())} ALL_DONE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
