#!/usr/bin/env python3
# Copyright 2026 rl-distill contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Upload completed Gemma 4 trace bundles to Hugging Face dataset repos.

One repo per trace spec, ``<repo-base>-<spec>`` (default ``JWei05/gemma4-bestckpt-traces-topk128-v2-<spec>``),
holding exactly what ``scale_train/run_gemma4_distill_one.sh`` needs to consume the bundle without the
generating node: ``train/`` and ``validation/`` (shards, manifests, run configs), the ``source/`` prompt
rosters, ``dataset_index.json``, ``COMPLETE.json`` and ``logs/final-validation.log``.

The generator leaves hidden ``.<shard>.lock`` files next to every shard (atomic-publish locks) and a
``.cache/`` directory when the bundle itself came from the Hub; both are excluded. Any ``.lock`` file
already present in the repo from an earlier upload is deleted first.

    python rl-distill-scripts/data/upload_gemma4_trace_bundle_hf.py 26b-hard e2b-hard \\
        --root /tmp/gemma4_bestckpt_traces_v2 [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ALLOW_PATTERNS = (
    "train/*",
    "validation/*",
    "source/*",
    "dataset_index.json",
    "COMPLETE.json",
    "logs/final-validation.log",
)
IGNORE_PATTERNS = ("*.lock", ".*", "**/.*", ".cache/*", "logs/*worker*")


def bundle_is_complete(root: Path, expected_rows: int | None) -> dict:
    """Return the dataset index of a COMPLETE bundle, raising if it is not one."""
    if not (root / "COMPLETE.json").exists():
        raise SystemExit(f"{root}: no COMPLETE.json (final validation has not passed)")
    index = json.loads((root / "dataset_index.json").read_text(encoding="utf-8"))
    if expected_rows is not None and index["total_rows"] != expected_rows:
        raise SystemExit(f"{root}: dataset_index reports {index['total_rows']} rows, expected {expected_rows}")
    if not any((root / "source").glob("*.parquet")):
        raise SystemExit(f"{root}: source/ has no roster parquet; the view builder needs it")
    return index


def upload_bundle(api, *, root: Path, repo: str, private: bool, index: dict, dry_run: bool) -> dict:
    from huggingface_hub import CommitOperationDelete

    if dry_run:
        return {"repo": repo, "dry_run": True}
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    stale = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".lock")]
    if stale:
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
            commit_message=f"remove {len(stale)} stray .lock files",
        )
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(root),
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
        commit_message=(
            f"{root.name} top-128 trace bundle ({index['total_rows']} rows, "
            f"{index['total_response_tokens']} response tokens)"
        ),
    )
    info = api.dataset_info(repo, files_metadata=True)
    files = [s.rfilename for s in info.siblings]
    return {
        "repo": repo,
        "sha": info.sha,
        "train_parquet": sum(1 for f in files if f.startswith("train/") and f.endswith(".parquet")),
        "validation_parquet": sum(1 for f in files if f.startswith("validation/") and f.endswith(".parquet")),
        "lock_files": sum(1 for f in files if f.endswith(".lock")),
        "files": len(files),
        "bytes": sum((s.size or 0) for s in info.siblings),
        "deleted_stale_locks": len(stale),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+", help="trace specs, e.g. 26b-hard e2b-easy")
    parser.add_argument("--root", type=Path, default=Path("/tmp/gemma4_bestckpt_traces_v2"))
    parser.add_argument("--repo-base", default="JWei05/gemma4-bestckpt-traces-topk128-v2")
    parser.add_argument("--private", action="store_true", help="create new repos private (default public)")
    parser.add_argument("--expected-rows", type=int, default=24300, help="0 disables the row-count check")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    expected_rows = args.expected_rows or None
    failures = 0
    for spec in args.specs:
        root = args.root / spec
        repo = f"{args.repo_base}-{spec}"
        started = time.time()
        try:
            index = bundle_is_complete(root, expected_rows)
            result = upload_bundle(api, root=root, repo=repo, private=args.private, index=index, dry_run=args.dry_run)
        except SystemExit as error:
            failures += 1
            print(f"UPLOAD_FAILED {spec}: {error}", flush=True)
            continue
        result["seconds"] = round(time.time() - started)
        if result.get("lock_files"):
            failures += 1
            print(f"UPLOAD_FAILED {spec}: {result['lock_files']} .lock files remain in {repo}", flush=True)
            continue
        print(f"UPLOAD_OK {spec} {json.dumps(result, sort_keys=True)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
