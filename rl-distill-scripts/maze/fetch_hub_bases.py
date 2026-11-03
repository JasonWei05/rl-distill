#!/usr/bin/env python3
"""Fetch matched starting checkpoints of the text-maze budget study from its Hub archive into the local matched/ layout.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/fetch_hub_bases.py --pairs p1 --sides S0 --manifest

For each pair and side, downloads used/<pair>/<side>_base/ (an HF model dir: 8 MB for S, 27 MB for L) into
<matched>/<pair>_<side>/ and links <matched>/<pair>_<side>_base/ckpt-0 -> ../<pair>_<side>, the layout train.sh and the
launchers (launch_single_rl.sh, launch_pair_rl.sh) take. An existing <matched>/<pair>_<side> is left alone.
--manifest merges the archive's meta/matched_manifest.json into <matched>/manifest.json, keeping every local entry
(make_pair.py appends new pairs to the same file), so start profiles of the original pairs are available to
rl_curve_family.py extract. The repository is public: no token is needed.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

REPO = "hi-todayis-jh/tailrl-text-maze-budget-study"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=[], help="pair tags: p10, p1, p01")
    ap.add_argument("--sides", nargs="+", default=["S0", "L0"], choices=["S0", "L0"])
    ap.add_argument("--manifest", action="store_true", help="merge meta/matched_manifest.json into <matched>/manifest.json")
    ap.add_argument("--matched-dir", default=os.path.expanduser("~/tailrl_maze_runs/matched"))
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(args.matched_dir, exist_ok=True)
    for pair in args.pairs:
        for side in args.sides:
            dest = os.path.join(args.matched_dir, f"{pair}_{side}")
            if os.path.exists(dest):
                print(f"{dest} exists; left as is")
            else:
                with tempfile.TemporaryDirectory(dir=args.matched_dir) as tmp:
                    snapshot_download(args.repo, repo_type="dataset", allow_patterns=[f"used/{pair}/{side}_base/*"], local_dir=tmp)
                    src = os.path.join(tmp, "used", pair, f"{side}_base")
                    if not os.path.isfile(os.path.join(src, "config.json")):
                        sys.exit(f"used/{pair}/{side}_base not found in {args.repo}")
                    shutil.copytree(src, dest)
                print(f"{dest} <- {args.repo}:used/{pair}/{side}_base ({len(os.listdir(dest))} files)")
            base = os.path.join(args.matched_dir, f"{pair}_{side}_base")
            os.makedirs(base, exist_ok=True)
            link = os.path.join(base, "ckpt-0")
            if not os.path.lexists(link):
                os.symlink(f"../{pair}_{side}", link)
            print(f"{link} -> {os.readlink(link)}")
    if args.manifest:
        path = os.path.join(args.matched_dir, "manifest.json")
        hub = json.load(open(hf_hub_download(args.repo, "meta/matched_manifest.json", repo_type="dataset")))
        local = json.load(open(path)) if os.path.exists(path) else {}
        added = []
        for k, v in hub.items():
            if k != "pairs":
                local.setdefault(k, v)
        for pair, v in hub.get("pairs", {}).items():
            if pair not in local.setdefault("pairs", {}):
                local["pairs"][pair] = v
                added.append(pair)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(local, f, indent=1)
        os.replace(tmp, path)
        print(f"{path}: added pairs {added or 'none'}; now {sorted(local['pairs'])}")


if __name__ == "__main__":
    main()
