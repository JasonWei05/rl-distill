# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Async HuggingFace Hub checkpoint pusher for verl/DAPO training.

Usage inside a trainer (rank 0 only):

    from dapo.hf_push import HFPusher
    pusher = HFPusher(repo_id="JWei05/dapo-gemma3-27b-it", private=True)
    # after every successful _save_checkpoint:
    pusher.push_async(
        local_dir=f"{ckpt_dir}/global_step_{step}/actor/huggingface",
        step=step,
        delete_local_after=True,   # reclaim /tmp
    )

Push runs in a background thread — the trainer does NOT block on the upload.
Uploads retry 3× with exponential backoff. ``wait()`` re-raises any background
failure so a run cannot report success without its required Hub artifacts.
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError


def _load_hf_token_from_dotenv() -> str | None:
    """Read HF_TOKEN from a local .env without exposing it in launch args."""
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.realpath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() != "HF_TOKEN":
                        continue
                    token = value.strip().strip("'\"")
                    if token:
                        os.environ.setdefault("HF_TOKEN", token)
                        return token
        except OSError:
            continue
    return None


def _remote_upload_impl(
    local_dir: str,
    repo_id: str,
    private: bool,
    token: str,
    path_in_repo: str,
    step: int,
    delete_local_after: bool,
    enable_hf_transfer: bool,
    max_retries: int,
):
    """Pure function used by the Ray-remote task. Runs on the node picked by the
    scheduler. Returns a tuple (status, detail) the caller can inspect."""
    import os as _os
    import shutil as _shutil
    import time as _time

    if not _os.path.isdir(local_dir):
        return ("skip", f"missing:{local_dir}")
    if enable_hf_transfer:
        _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi as _HfApi
    from huggingface_hub import create_repo as _create_repo
    from huggingface_hub.utils import HfHubHTTPError as _HfHubHTTPError

    create_repo_kwargs = dict(repo_id=repo_id, token=token, private=private, repo_type="model", exist_ok=True)
    _create_repo(**create_repo_kwargs)
    api = _HfApi(token=token)
    if not private:
        try:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
        except Exception as e:
            print(f"[HFPusher] repo visibility update skipped: {e}", flush=True)
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"step {step}",
            )
            if delete_local_after:
                _shutil.rmtree(local_dir, ignore_errors=True)
            return ("ok", f"uploaded:{path_in_repo}")
        except (_HfHubHTTPError, OSError, ConnectionError) as e:
            _time.sleep(2**attempt)
            last = str(e)
    return ("err", f"giving_up_after_{max_retries}:{last}")


def _safe_join(base_dir: str, rel_path: str) -> str:
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, rel_path))
    if path != base and not path.startswith(base + os.sep):
        raise ValueError(f"path escapes base dir: {rel_path}")
    return path


def _remote_list_dir_impl(local_dir: str):
    """Return a serializable manifest for a directory on a Ray node."""
    import os as _os
    import stat as _stat

    if not _os.path.isdir(local_dir):
        return ("skip", f"missing:{local_dir}")

    dirs = []
    files = []
    total_bytes = 0
    try:
        for root, dirnames, filenames in _os.walk(local_dir):
            dirnames.sort()
            filenames.sort()
            rel_root = _os.path.relpath(root, local_dir)
            if rel_root != ".":
                st = _os.stat(root)
                dirs.append((rel_root, _stat.S_IMODE(st.st_mode)))
            for name in filenames:
                path = _os.path.join(root, name)
                if not _os.path.isfile(path):
                    continue
                rel = _os.path.relpath(path, local_dir)
                st = _os.stat(path)
                size = st.st_size
                total_bytes += size
                files.append((rel, size, _stat.S_IMODE(st.st_mode)))
    except Exception as e:
        return ("err", f"manifest_error:{e}")

    return ("ok", {"dirs": dirs, "files": files, "total_bytes": total_bytes})


def _remote_read_file_chunk_impl(local_dir: str, rel_path: str, offset: int, size: int):
    import os as _os

    base = _os.path.realpath(local_dir)
    path = _os.path.realpath(_os.path.join(base, rel_path))
    if path != base and not path.startswith(base + _os.sep):
        raise ValueError(f"path escapes base dir: {rel_path}")
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def _remote_remove_impl(path: str):
    import os as _os
    import shutil as _shutil

    if _os.path.isdir(path):
        _shutil.rmtree(path, ignore_errors=True)
        return ("ok", f"removed:{path}")
    return ("skip", f"missing:{path}")


def _get_ray_remote_task():
    """Lazily wrap _remote_upload_impl with @ray.remote so importing this module
    doesn't require ray."""
    import ray

    return ray.remote(num_cpus=1, max_retries=0)(_remote_upload_impl)


@dataclass
class HFPusher:
    repo_id: str
    private: bool = True
    token: str | None = None
    enable_hf_transfer: bool = True
    max_retries: int = 3
    # Keep the newest N step-folders on the Hub (None = keep all). Before each upload the repo is pruned to
    # N-1 and after it to N; pruned LFS blobs are released by squashing history (see _prune).
    max_to_keep: int | None = None

    _pushed_steps: list = field(default_factory=list)
    _api: HfApi | None = None
    _repo_ready: bool = False
    _queue: queue.Queue | None = None
    _worker: threading.Thread | None = None
    _errors: list[tuple[str, BaseException]] = field(default_factory=list)
    _closed: bool = False

    def __post_init__(self):
        self.token = self.token or os.environ.get("HF_TOKEN") or _load_hf_token_from_dotenv()
        if not self.token:
            raise RuntimeError("HFPusher: HF_TOKEN not set in env or passed explicitly.")
        if self.enable_hf_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self._api = HfApi(token=self.token)
        # Single worker thread — HF commits to one branch race under concurrency,
        # and prune ordering assumes chronological success order. Serialize pushes.
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _run_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                mode = item[0]
                if mode == "local":
                    _, local_dir, step, delete_local_after = item
                    self._do_one_local(local_dir, step, delete_local_after)
                elif mode == "cluster":
                    _, local_dir, step, delete_local_after = item
                    self._do_one_cluster(local_dir, step, delete_local_after)
                else:
                    raise ValueError(f"HFPusher received an unknown queue item: {item}")
            except Exception as e:
                print(f"[HFPusher] worker error: {e}", flush=True)
                traceback.print_exc()
                self._errors.append((repr(item), e))
            finally:
                self._queue.task_done()

    def _ensure_repo(self):
        if self._repo_ready:
            return
        create_repo(
            repo_id=self.repo_id,
            token=self.token,
            private=self.private,
            repo_type="model",
            exist_ok=True,
        )
        if not self.private:
            try:
                self._api.update_repo_settings(repo_id=self.repo_id, private=False, repo_type="model")
            except Exception as e:
                print(f"[HFPusher] repo visibility update skipped: {e}", flush=True)
        self._repo_ready = True

    def _upload_with_retry(self, local_dir: str, path_in_repo: str, step: int):
        for attempt in range(1, self.max_retries + 1):
            try:
                self._api.upload_folder(
                    folder_path=local_dir,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"step {step}",
                )
                return True
            except (HfHubHTTPError, OSError, ConnectionError) as e:
                wait = 2**attempt
                print(f"[HFPusher] step {step} attempt {attempt} failed: {e}. retrying in {wait}s", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"[HFPusher] step {step} non-retryable error: {e}", flush=True)
                traceback.print_exc()
                return False
        print(f"[HFPusher] step {step} giving up after {self.max_retries} retries", flush=True)
        return False

    def _remote_step_folders(self) -> list[int]:
        """Return the sorted step numbers of top-level ``step_NNNNNN`` folders currently on the Hub."""
        steps = set()
        for entry in self._api.list_repo_tree(repo_id=self.repo_id, repo_type="model"):
            name = getattr(entry, "path", "")
            if name.startswith("step_") and name[5:].isdigit():
                steps.add(int(name[5:]))
        return sorted(steps)

    def _prune(self, keep: int | None = None):
        """Keep only the newest ``keep`` (default ``max_to_keep``) step-folders on the Hub, then squash history.

        The Hub is listed directly (not just this process's ``_pushed_steps``) so folders pushed by a
        previous run/resume are pruned too. Deleting LFS pointers alone does not free storage quota on
        the Hub; ``super_squash_history`` rewrites the repo to a single commit so the deleted blobs are
        actually released (quota reflects it within ~36h). Disable with HF_PUSH_SQUASH_AFTER_PRUNE=0.
        """
        keep = self.max_to_keep if keep is None else keep
        if keep is None:
            return
        try:
            remote_steps = self._remote_step_folders()
        except Exception as e:
            print(f"[HFPusher] prune: could not list {self.repo_id}: {e}", flush=True)
            remote_steps = sorted(set(self._pushed_steps))
        if len(remote_steps) <= keep:
            return
        to_drop = remote_steps[:-keep]
        self._pushed_steps = [s for s in self._pushed_steps if s not in to_drop]
        dropped_any = False
        for step in to_drop:
            path = f"step_{step:06d}"
            try:
                self._api.delete_folder(
                    path_in_repo=path,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"prune {path}",
                )
                dropped_any = True
                print(f"[HFPusher] pruned {path} from hub", flush=True)
            except Exception as e:
                print(f"[HFPusher] prune {path} failed: {e}", flush=True)
        squash = os.environ.get("HF_PUSH_SQUASH_AFTER_PRUNE", "1").strip().lower() not in {"0", "false", "no", "off"}
        if dropped_any and squash:
            try:
                self._api.super_squash_history(repo_id=self.repo_id, repo_type="model")
                print(f"[HFPusher] squashed history of {self.repo_id} to release pruned LFS storage", flush=True)
            except Exception as e:
                print(f"[HFPusher] super_squash_history failed (storage not released): {e}", flush=True)

    def _make_room_before_upload(self):
        """Prune to max_to_keep-1 BEFORE uploading: the Hub checks storage quota at commit time, so
        uploading first would require a full extra snapshot of headroom on an account near its cap."""
        if self.max_to_keep is not None and self.max_to_keep > 1:
            self._prune(keep=self.max_to_keep - 1)

    def _do_one_local(self, local_dir: str, step: int, delete_local_after: bool):
        path_in_repo = f"step_{step:06d}"
        self._ensure_repo()
        self._make_room_before_upload()
        ok = self._upload_with_retry(local_dir, path_in_repo, step)
        if ok:
            self._pushed_steps.append(step)
            print(f"[HFPusher] uploaded {local_dir} → {self.repo_id}/{path_in_repo}", flush=True)
            self._prune()
            if delete_local_after:
                shutil.rmtree(local_dir, ignore_errors=True)
                print(f"[HFPusher] removed local {local_dir}", flush=True)
            return
        raise RuntimeError(f"Hugging Face upload failed for step {step}: {local_dir}")

    def _do_one_cluster(self, local_dir: str, step: int, delete_local_after: bool):
        """Find the Ray node with the checkpoint, stage it to this driver, then
        upload from the driver.

        The B200 Ray nodes used for training cannot reach Hugging Face/W&B in
        this environment. The CPU driver can, so cluster pushes copy checkpoint
        files from the owning Ray node to driver-local /tmp in chunks and upload
        from here.
        """
        if os.path.isdir(local_dir):
            print(f"[HFPusher] driver has {local_dir}; uploading locally for step {step}", flush=True)
            return self._do_one_local(local_dir, step, delete_local_after)

        import ray

        if not ray.is_initialized():
            print(f"[HFPusher] ray not initialized; falling back to local for step {step}", flush=True)
            return self._do_one_local(local_dir, step, delete_local_after)

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = [n for n in ray.nodes() if n.get("Alive")]
        path_in_repo = f"step_{step:06d}"

        list_task = ray.remote(num_cpus=0.1, max_retries=0)(_remote_list_dir_impl)
        refs = []
        for n in nodes:
            strat = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
            refs.append(list_task.options(scheduling_strategy=strat).remote(local_dir))

        owner = None
        for ref, n in zip(refs, nodes, strict=False):
            try:
                status, detail = ray.get(ref)
            except Exception as e:
                status, detail = "err", f"ray_exception:{e}"
            if status == "ok":
                total_gb = detail.get("total_bytes", 0) / (1024**3)
                summary = f"{len(detail.get('files', []))} files, {total_gb:.2f} GiB"
            else:
                summary = detail
            print(f"[HFPusher] step {step} node={n.get('NodeManagerAddress')} -> {status} ({summary})", flush=True)
            if status == "ok" and owner is None:
                owner = (n, detail)

        if owner is None:
            raise FileNotFoundError(f"HFPusher step {step}: no alive Ray node contains {local_dir}")

        owner_node, manifest = owner
        staged_dir = self._stage_from_ray_node(ray, owner_node, local_dir, manifest, step)
        self._ensure_repo()
        ok = self._upload_with_retry(staged_dir, path_in_repo, step)
        if ok:
            self._pushed_steps.append(step)
            print(f"[HFPusher] uploaded staged {staged_dir} -> {self.repo_id}/{path_in_repo}", flush=True)
            self._prune()
            shutil.rmtree(staged_dir, ignore_errors=True)
            print(f"[HFPusher] removed staged {staged_dir}", flush=True)
            if delete_local_after:
                remove_task = ray.remote(num_cpus=0.1, max_retries=0)(_remote_remove_impl)
                strat = NodeAffinitySchedulingStrategy(node_id=owner_node["NodeID"], soft=False)
                try:
                    status, detail = ray.get(remove_task.options(scheduling_strategy=strat).remote(local_dir))
                except Exception as e:
                    status, detail = "err", f"ray_exception:{e}"
                print(f"[HFPusher] step {step} source cleanup -> {status} ({detail})", flush=True)
        else:
            print(f"[HFPusher] step {step} upload failed; leaving staged files at {staged_dir}", flush=True)
            raise RuntimeError(f"Hugging Face upload failed for step {step}; staged files remain at {staged_dir}")

    def _stage_from_ray_node(self, ray, node, local_dir: str, manifest: dict, step: int) -> str:
        """Copy a Ray-node-local directory to driver-local storage in chunks."""
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        repo_slug = self.repo_id.replace("/", "__")
        staging_root = os.environ.get("HF_PUSH_STAGING_ROOT", "/tmp/hf_push_staging")
        staged_dir = os.path.join(staging_root, repo_slug, f"step_{step:06d}")
        if os.path.exists(staged_dir):
            shutil.rmtree(staged_dir, ignore_errors=True)
        os.makedirs(staged_dir, exist_ok=True)

        for rel_dir, mode in manifest.get("dirs", []):
            dest = _safe_join(staged_dir, rel_dir)
            os.makedirs(dest, exist_ok=True)
            try:
                os.chmod(dest, mode)
            except OSError:
                pass

        files = manifest.get("files", [])
        total_bytes = manifest.get("total_bytes", 0)
        chunk_size = int(os.environ.get("HF_PUSH_CHUNK_BYTES", str(32 * 1024 * 1024)))
        read_task = ray.remote(num_cpus=0.1, max_retries=0)(_remote_read_file_chunk_impl)
        strat = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)

        copied = 0
        for idx, (rel_path, size, mode) in enumerate(files, start=1):
            dest = _safe_join(staged_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as out:
                offset = 0
                while offset < size:
                    want = min(chunk_size, size - offset)
                    data = ray.get(
                        read_task.options(scheduling_strategy=strat).remote(local_dir, rel_path, offset, want)
                    )
                    if len(data) != want:
                        raise RuntimeError(f"short read for {rel_path}: offset={offset} want={want} got={len(data)}")
                    out.write(data)
                    offset += len(data)
                    copied += len(data)
            try:
                os.chmod(dest, mode)
            except OSError:
                pass
            if idx == 1 or idx == len(files) or idx % 8 == 0:
                copied_gb = copied / (1024**3)
                total_gb = total_bytes / (1024**3)
                print(
                    f"[HFPusher] step {step} staged {idx}/{len(files)} files ({copied_gb:.2f}/{total_gb:.2f} GiB)",
                    flush=True,
                )

        return staged_dir

    def push_async(self, local_dir: str, step: int, delete_local_after: bool = False):
        """Enqueue upload from this process's local filesystem.
        Use this when you know the files live on the node where this pusher runs
        (e.g., single-node training, or backfill scripts launched on the owning node)."""
        if self._closed:
            raise RuntimeError("HFPusher is closed")
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"HFPusher checkpoint directory does not exist: {local_dir}")
        self._queue.put(("local", local_dir, step, delete_local_after))

    def push_cluster(self, local_dir: str, step: int, delete_local_after: bool = False):
        """Enqueue upload that broadcasts to every alive Ray node. Only the node
        whose local filesystem actually holds `local_dir` will upload; the others
        return a 'skip'. Use this from the verl driver actor, since FSDP rank-0
        may live on a different Ray node than the driver."""
        if self._closed:
            raise RuntimeError("HFPusher is closed")
        self._queue.put(("cluster", local_dir, step, delete_local_after))

    def wait(self, timeout: float | None = None):
        """Block until all enqueued uploads finish. Call before process exit."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError(f"HFPusher did not finish within {timeout} seconds")
        if self._errors:
            summary = "; ".join(f"{item}: {error}" for item, error in self._errors)
            raise RuntimeError(f"HFPusher failed {len(self._errors)} queued upload(s): {summary}") from self._errors[0][
                1
            ]


def wait_for_hf_pusher(pusher: HFPusher, *, timeout: float | None = None) -> None:
    """Drain pending uploads without replacing an active training exception.

    Call this from a ``finally`` block. If training succeeded, an upload error is
    still fatal. If training is already unwinding, preserve that primary error
    and attach/log the upload failure instead of masking the training cause.
    """
    primary_error = sys.exc_info()[1]
    try:
        pusher.wait(timeout=timeout)
    except Exception as upload_error:
        if primary_error is None:
            raise
        note = f"A pending Hugging Face upload also failed during shutdown: {upload_error}"
        if hasattr(primary_error, "add_note"):
            primary_error.add_note(note)
        print(f"[HFPusher] {note}", file=sys.stderr, flush=True)
        traceback.print_exception(type(upload_error), upload_error, upload_error.__traceback__)
