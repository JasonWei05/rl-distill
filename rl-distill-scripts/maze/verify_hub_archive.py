#!/usr/bin/env python3
"""Verify the Hub copy of the staged text-maze archive against the local folder.

Every local file (except staging bookkeeping) must exist in the repo with the same size, and every LFS file with the
same sha256 (tars are checked against meta/archive_sha256.txt computed at staging time; other LFS files are hashed here).
Exit 0 only when nothing is missing or different.
    ~/TailRL/.venv/bin/python verify_hub_archive.py [repo_id] [local_dir]
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = sys.argv[1] if len(sys.argv) > 1 else "hi-todayis-jh/tailrl-text-maze-budget-study"
H = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/tailrl_maze_hub")
SKIP_SUFFIX = (".done", "stage.log", "upload.log")


def sha256_of(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


staged_sha = {}
sha_file = os.path.join(H, "meta", "archive_sha256.txt")
if os.path.exists(sha_file):
    for line in open(sha_file):
        digest, name = line.split(maxsplit=1)
        staged_sha["archive/" + name.strip()] = digest

api = HfApi()
remote = {}
for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
    size = getattr(e, "size", None)
    if size is None:
        continue
    lfs = getattr(e, "lfs", None)
    remote[e.path] = (size, getattr(lfs, "sha256", None) if lfs else None)

local = {}
for root, dirs, files in os.walk(H):
    dirs[:] = [d for d in dirs if d != ".cache"]
    for f in files:
        if f.endswith(SKIP_SUFFIX):
            continue
        p = os.path.join(root, f)
        local[os.path.relpath(p, H)] = os.path.getsize(p)

missing, size_bad, sha_bad, sha_ok, total = [], [], [], 0, 0
for rel, size in sorted(local.items()):
    if rel not in remote:
        missing.append(rel)
        continue
    rsize, rsha = remote[rel]
    total += size
    if rsize != size:
        size_bad.append((rel, size, rsize))
        continue
    if rsha:
        lsha = staged_sha.get(rel) or sha256_of(os.path.join(H, rel))
        if lsha != rsha:
            sha_bad.append((rel, lsha, rsha))
        else:
            sha_ok += 1
extra = sorted(set(remote) - set(local) - {".gitattributes"})
print(f"repo {REPO}: {len(remote)} remote files, {len(local)} local files, {total/1e9:.1f} GB compared")
print(f"missing on hub: {len(missing)}  size mismatch: {len(size_bad)}  sha256 mismatch: {len(sha_bad)}  sha256 ok: {sha_ok}  extra on hub: {len(extra)}")
for x in missing[:20]:
    print("  missing:", x)
for x in size_bad[:20]:
    print("  size:", x)
for x in sha_bad[:20]:
    print("  sha:", x)
for x in extra[:10]:
    print("  extra:", x)
ok = not missing and not size_bad and not sha_bad
print("VERIFY", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
