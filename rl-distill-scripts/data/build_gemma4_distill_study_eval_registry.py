#!/usr/bin/env python3
"""Build the eval-source registry for the Gemma 4 distillation study.

Roster = the two untrained base students (E2B/E4B, pinned), the six small RL teachers (e2b/e4b x
easy/medium/hard, pinned to immutable Hub commits) and every distilled student found on the Hub.
``--no-bases`` drops the bases. Distilled repos are
discovered by name (the two node prefixes ``Distill-gemma4-`` and ``gemma4-distill-v2-``), keep
only repos that already hold a ``step_NNNNNN/`` export (the model is pushed at the final step),
take the latest step, and pin the repo's current ``main`` commit so a later push cannot change
what was evaluated. Re-run to pick up newly finished distillations; existing pins are kept.

Every model evaluates all three 300-question bands (its own band is the in-distribution number,
the others report cross-band transfer) plus MATH500 and GSM8K.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "rl-distill-scripts/config/gemma4_distill_study_eval_sources.json"
HF_API = "https://huggingface.co/api"
MATH_DATASETS = ["id_easy", "id_medium", "id_hard", "math500", "gsm8k"]

BASES = {  # architecture -> (repo, pinned revision) — same pins as the earlier eval registry
    "gemma-4-E2B": ("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"),
    "gemma-4-E4B": ("google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"),
}
# The small RL teachers: W&B val mean@16 peaks, pinned to the Hub commit that holds that step
# (identical to the pins in scale_train/run_gemma4_bestckpt_trace_collection.sh).
RL_TEACHERS = [
    ("e2b", "easy", 130, "JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-easy-seed42-local2gpu", "c82460136fb16c36ff91dd2a489fba9332b52432"),
    ("e2b", "medium", 240, "JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-medium-seed42-local2gpu", "497e7964f98bb6d825e4c7fcc95d75fa928a6d83"),
    ("e2b", "hard", 190, "JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-hard-seed42-local2gpu", "59762d43bf94b6938d2e560e1fdbfdbf0d3f9e4c"),
    ("e4b", "easy", 100, "JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5", "345beec132e3e38922b21fd7b648d002dc4d333d"),
    ("e4b", "medium", 90, "JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5", "5f90d25e193d5071d74975793c20d4ac10fc733f"),
    ("e4b", "hard", 120, "JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-hard-seed42-26b-bands-es5", "627bd9d825ffdab1552fb3bbc1af410a8d2ac0a1"),
]
DISTILLED_NAME = re.compile(
    r"^JWei05/(?:Distill-gemma4|gemma4-distill-v2)-(?P<teacher>26b|12b|e4b|e2b)-(?P<band>easy|medium|hard)-to-(?P<student>e4b|e2b)-base$"
)
ARCH = {"e2b": "gemma-4-E2B", "e4b": "gemma-4-E4B"}


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token or None


def _get(session: requests.Session, url: str) -> Any:
    for attempt in range(4):
        response = session.get(url, timeout=60)
        if response.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"rate limited by the Hub: {url}")


def _source(repo: str, revision: str, subfolder: str, architecture: str) -> dict[str, Any]:
    metadata_repo, metadata_revision = BASES[architecture]
    return {
        "type": "hf_subfolder",
        "repo_id": repo,
        "revision": revision,
        "subfolder": subfolder,
        "metadata_repo": metadata_repo,
        "metadata_revision": metadata_revision,
    }


def discover_distilled(session: requests.Session, existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    models = []
    listing = _get(session, f"{HF_API}/models?author=JWei05&limit=500")
    for item in sorted(listing, key=lambda m: m["id"]):
        match = DISTILLED_NAME.match(item["id"])
        if not match:
            continue
        teacher, band, student = match["teacher"], match["band"], match["student"]
        tag = f"distill_{teacher}_{band}_to_{student}"
        if tag in existing:  # keep the earlier pin; re-pinning would silently change the evaluated weights
            models.append(existing[tag])
            continue
        time.sleep(0.5)
        tree = _get(session, f"{HF_API}/models/{item['id']}/tree/main")
        steps = sorted(entry["path"] for entry in tree if entry["path"].startswith("step_"))
        if not steps:
            print(f"  skip {item['id']}: no step_* export yet (run still in progress)")
            continue
        commit = _get(session, f"{HF_API}/models/{item['id']}/commits/main")[0]["id"]
        models.append(
            {
                "tag": tag,
                "display_name": f"Distilled {student.upper()} base <- {teacher} RL {band} teacher ({steps[-1]})",
                "category": "distilled",
                "architecture": ARCH[student],
                "trained_on": band,
                "math_datasets": list(MATH_DATASETS),
                "source": _source(item["id"], commit, steps[-1], ARCH[student]),
            }
        )
        print(f"  pinned {item['id']} {steps[-1]} @ {commit[:12]}")
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-discovery", action="store_true", help="do not query the Hub for distilled models")
    parser.add_argument("--no-bases", action="store_true", help="do not roster the two untrained base models")
    args = parser.parse_args()

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        existing = {m["tag"]: m for m in json.loads(args.output.read_text())["models"] if m["category"] == "distilled"}

    models: list[dict[str, Any]] = []
    for architecture, (repo, revision) in ([] if args.no_bases else BASES.items()):
        student = architecture.split("-")[-1].lower()
        models.append(
            {
                "tag": f"base_{student}",
                "display_name": f"Gemma 4 {architecture.split('-')[-1]} PT base",
                "category": "base",
                "architecture": architecture,
                "trained_on": None,
                "math_datasets": list(MATH_DATASETS),
                "source": {"type": "hf_snapshot", "repo_id": repo, "revision": revision},
            }
        )
    for student, band, step, repo, revision in RL_TEACHERS:
        models.append(
            {
                "tag": f"rl_{student}_{band}",
                "display_name": f"{student.upper()} RL {band} best (step {step})",
                "category": "rl",
                "architecture": ARCH[student],
                "trained_on": band,
                "math_datasets": list(MATH_DATASETS),
                "source": _source(repo, revision, f"step_{step:06d}", ARCH[student]),
            }
        )
    if not args.skip_discovery:
        session = requests.Session()
        token = _token()
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        models.extend(discover_distilled(session, existing))
    else:
        models.extend(existing.values())

    payload = {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_eval_sources_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "gemma4-distill-study",
        "models": models,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    counts = {c: sum(m["category"] == c for m in models) for c in ("base", "rl", "distilled")}
    print(f"wrote {args.output} with {len(models)} models: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
