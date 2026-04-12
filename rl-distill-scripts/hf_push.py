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
Uploads retry 3× with exponential backoff. Failures are logged and dropped
(training continues).
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError


def _remote_upload_impl(local_dir: str, repo_id: str, private: bool, token: str,
                        path_in_repo: str, step: int, delete_local_after: bool,
                        enable_hf_transfer: bool, max_retries: int):
    """Pure function used by the Ray-remote task. Runs on the node picked by the
    scheduler. Returns a tuple (status, detail) the caller can inspect."""
    import os as _os
    import shutil as _shutil
    import time as _time
    if not _os.path.isdir(local_dir):
        return ("skip", f"missing:{local_dir}")
    if enable_hf_transfer:
        _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi as _HfApi, create_repo as _create_repo
    from huggingface_hub.utils import HfHubHTTPError as _HfHubHTTPError
    create_repo_kwargs = dict(repo_id=repo_id, token=token, private=private,
                              repo_type="model", exist_ok=True)
    _create_repo(**create_repo_kwargs)
    api = _HfApi(token=token)
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(folder_path=local_dir, path_in_repo=path_in_repo,
                              repo_id=repo_id, repo_type="model",
                              commit_message=f"step {step}")
            if delete_local_after:
                _shutil.rmtree(local_dir, ignore_errors=True)
            return ("ok", f"uploaded:{path_in_repo}")
        except (_HfHubHTTPError, OSError, ConnectionError) as e:
            _time.sleep(2 ** attempt)
            last = str(e)
    return ("err", f"giving_up_after_{max_retries}:{last}")


def _get_ray_remote_task():
    """Lazily wrap _remote_upload_impl with @ray.remote so importing this module
    doesn't require ray."""
    import ray
    return ray.remote(num_cpus=1, max_retries=0)(_remote_upload_impl)


@dataclass
class HFPusher:
    repo_id: str
    private: bool = True
    token: Optional[str] = None
    enable_hf_transfer: bool = True
    max_retries: int = 3
    # Keep last N step-folders on the hub (None = keep all). Oldest get deleted.
    max_to_keep: Optional[int] = None

    _pushed_steps: list = field(default_factory=list)
    _api: Optional[HfApi] = None
    _repo_ready: bool = False
    _queue: Optional[queue.Queue] = None
    _worker: Optional[threading.Thread] = None

    def __post_init__(self):
        self.token = self.token or os.environ.get("HF_TOKEN")
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
                    print(f"[HFPusher] unknown queue item: {item}", flush=True)
            except Exception as e:
                print(f"[HFPusher] worker error: {e}", flush=True)
                traceback.print_exc()
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
                wait = 2 ** attempt
                print(f"[HFPusher] step {step} attempt {attempt} failed: {e}. "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"[HFPusher] step {step} non-retryable error: {e}", flush=True)
                traceback.print_exc()
                return False
        print(f"[HFPusher] step {step} giving up after {self.max_retries} retries", flush=True)
        return False

    def _prune(self):
        """Delete oldest step-folders on the Hub, keeping only max_to_keep."""
        if self.max_to_keep is None or len(self._pushed_steps) <= self.max_to_keep:
            return
        to_drop = self._pushed_steps[:-self.max_to_keep]
        self._pushed_steps = self._pushed_steps[-self.max_to_keep:]
        for step in to_drop:
            path = f"step_{step:06d}"
            try:
                self._api.delete_folder(
                    path_in_repo=path,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"prune {path}",
                )
                print(f"[HFPusher] pruned {path} from hub", flush=True)
            except Exception as e:
                print(f"[HFPusher] prune {path} failed: {e}", flush=True)

    def _do_one_local(self, local_dir: str, step: int, delete_local_after: bool):
        path_in_repo = f"step_{step:06d}"
        self._ensure_repo()
        ok = self._upload_with_retry(local_dir, path_in_repo, step)
        if ok:
            self._pushed_steps.append(step)
            print(f"[HFPusher] uploaded {local_dir} → {self.repo_id}/{path_in_repo}", flush=True)
            self._prune()
            if delete_local_after:
                shutil.rmtree(local_dir, ignore_errors=True)
                print(f"[HFPusher] removed local {local_dir}", flush=True)

    def _do_one_cluster(self, local_dir: str, step: int, delete_local_after: bool):
        """Broadcast an upload task to every alive Ray node; whichever node has
        the files locally performs the upload, the rest return 'skip'."""
        import ray
        if not ray.is_initialized():
            print(f"[HFPusher] ray not initialized; falling back to local for step {step}",
                  flush=True)
            return self._do_one_local(local_dir, step, delete_local_after)
        remote_task = _get_ray_remote_task()
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        path_in_repo = f"step_{step:06d}"
        refs = []
        for n in nodes:
            strat = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
            refs.append(
                remote_task.options(scheduling_strategy=strat).remote(
                    local_dir=local_dir,
                    repo_id=self.repo_id,
                    private=self.private,
                    token=self.token,
                    path_in_repo=path_in_repo,
                    step=step,
                    delete_local_after=delete_local_after,
                    enable_hf_transfer=self.enable_hf_transfer,
                    max_retries=self.max_retries,
                )
            )
        ok_any = False
        for ref, n in zip(refs, nodes):
            try:
                status, detail = ray.get(ref)
            except Exception as e:
                status, detail = "err", f"ray_exception:{e}"
            print(f"[HFPusher] step {step} node={n.get('NodeManagerAddress')} "
                  f"-> {status} ({detail})", flush=True)
            if status == "ok":
                ok_any = True
        if ok_any:
            self._pushed_steps.append(step)
            self._prune()
        else:
            print(f"[HFPusher] step {step} — no node had the files; push failed", flush=True)

    def push_async(self, local_dir: str, step: int, delete_local_after: bool = False):
        """Enqueue upload from this process's local filesystem.
        Use this when you know the files live on the node where this pusher runs
        (e.g., single-node training, or backfill scripts launched on the owning node)."""
        if not os.path.isdir(local_dir):
            print(f"[HFPusher] skip: {local_dir} does not exist", flush=True)
            return
        self._queue.put(("local", local_dir, step, delete_local_after))

    def push_cluster(self, local_dir: str, step: int, delete_local_after: bool = False):
        """Enqueue upload that broadcasts to every alive Ray node. Only the node
        whose local filesystem actually holds `local_dir` will upload; the others
        return a 'skip'. Use this from the verl driver actor, since FSDP rank-0
        may live on a different Ray node than the driver."""
        self._queue.put(("cluster", local_dir, step, delete_local_after))

    def wait(self, timeout: Optional[float] = None):
        """Block until all enqueued uploads finish. Call before process exit."""
        # `timeout` param kept for compat; queue.join has no timeout, so we
        # send a sentinel and join the thread instead.
        self._queue.put(None)
        self._worker.join(timeout=timeout)
