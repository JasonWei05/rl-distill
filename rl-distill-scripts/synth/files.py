"""Run-dir file helpers. Files are written whole: to <path>.tmp, then renamed, so a reader (or the S3 mirror, which skips
*.tmp*) never sees a partial file."""

import json
import os


def atomic(path, write):
    """write(tmp_path), then move it to path."""
    write(path + ".tmp")
    os.replace(path + ".tmp", path)


def write_json(path, obj):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
    atomic(path, write)


def read_json(path):
    with open(path) as f:
        return json.load(f)
