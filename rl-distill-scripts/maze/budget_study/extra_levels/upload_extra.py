#!/usr/bin/env python3
"""Upload maze_extra/extra/ to the old repo's extra/ (remote-only files such as extra/used/ are left alone), then check
every local file is on the Hub at its local size.
    upload_extra.py "<commit message>" """
import os
import sys

from huggingface_hub import HfApi

D = "/work2/jiahaoz4/maze_extra/extra"
OLD = "hi-todayis-jh/tailrl-text-maze-budget-study"
api = HfApi()
api.upload_folder(folder_path=D, path_in_repo="extra", repo_id=OLD, repo_type="dataset", commit_message=sys.argv[1])
local = {}
for root, _, files in os.walk(D):
    for f in files:
        p = os.path.join(root, f)
        local["extra/" + os.path.relpath(p, D)] = os.path.getsize(p)
remote = {i.path: i.size for i in api.get_paths_info(OLD, sorted(local), repo_type="dataset")}
bad = [k for k, s in local.items() if remote.get(k) != s]
print(f"{len(local)} local files, {len(local) - len(bad)} on the Hub at the same size; mismatched: {bad or 'none'}")
sys.exit(1 if bad else 0)
