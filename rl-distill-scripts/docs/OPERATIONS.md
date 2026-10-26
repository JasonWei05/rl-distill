# Operations: launching, supervising, checkpointing and monitoring runs

Last updated: 2026-10-26

How jobs in this repo are actually run: on ScaleTrain EKS H100 pods (`p5.48xlarge`, fractional presets `:1/:2/:4`), launched
from the shared devbox, supervised by tmux-resident relaunch loops, checkpointed to S3, and logged to W&B. Recipes (what to
train) are in [RECIPES.md](RECIPES.md); results are in [GEMMA4_RESULTS.md](GEMMA4_RESULTS.md) and [QWEN3_RESULTS.md](QWEN3_RESULTS.md);
the experiment history is in [../PROGRESS_LOG.md](../PROGRESS_LOG.md).

All paths below are relative to the repo root unless they start with `s3://` or `/`.

---

## 1. Environments

| Venv | Built by | Stack | Used for |
|---|---|---|---|
| `.venv` | `rl-distill-scripts/setup_env.sh` | torch 2.9.1+cu128, vllm 0.15.1, transformers 4.57.6, flash-attn 2.8.3 (source build), numpy 2.2.6, awscli, boto3, wandb | Baked into the ScaleTrain image at `/workspace/rl-distill/.venv` (provides the `aws` binary for the code bootstrap). On the devbox: S3/W&B tooling (`full_checkpoint_s3.py` janitors, `republish_best_hf.py`, backfills). **Cannot load Gemma 4.** |
| `.venv-gemma4` | `rl-distill-scripts/setup_env_gemma4.sh` | torch 2.11.0, vllm 0.25.1, transformers 5.14.1, tensordict 0.13.0, torchcodec 0.15.0 (pinned), ray (unpinned), math-verify, **no flash-attn** | Every training/eval job (Gemma 4 *and* Qwen3). Built per pod into `/tmp/.venv-gemma4` at job start (~10 min, wheels only). |

`setup_env_gemma4.sh` details:

- `GEMMA4_CUDA_VARIANT=cu130` (script default, for the devbox's CUDA-13 driver) or `cu129` (what the RL run-file passes; the
  ScaleTrain p5 fleet has a **CUDA 12.8 driver**, so cu130 wheels die with `NVIDIA driver too old (found version 12080)`).
  cu129 pulls the `vllm-0.25.1+cu129` wheel from the vLLM GitHub release (the PyPI wheel is CUDA-13 linked).
- Reinstalls `torchcodec==0.15.0 --no-deps` from the PyTorch index of the variant: the unpinned PyPI 0.17.0 needs
  `libnvrtc.so.13` and broke `import vllm` on every cu129 pod (2026-10-06). Do not add an `import torchcodec.decoders` check
  (pods lack FFmpeg; vLLM catches that RuntimeError). The `import vllm` check is the gate.
- Runs `rl-distill-scripts/patch_vllm_gemma4_r3.py` (version-gated fix for vLLM 0.25.1's R3 router-capture logger on Gemma 4).
- Ends with `GEMMA4_ENV_CORE_OK` after checking that the Gemma 4 archs are registered in vLLM.

Secrets: a repo-root `.env` (untracked) with `HF_TOKEN=`, `WANDB_API_KEY=` (optional `WANDB_BASE_URL=`). `launch_st_job.py`
reads it (`--dotenv ../../.env`, `--dotenv-keys HF_TOKEN,WANDB_API_KEY,WANDB_BASE_URL`) and forwards the values as job env
vars (redacted in its printout; the temp job YAML is `chmod 600` and deleted after submit). Run-files source `.env` only if it
exists. `run_dapo_rl.sh` refuses to put the keys on the Ray command line.

Editable installs: both venvs are installed with `uv pip install --no-deps -e .`, so `import verl` resolves to the tree the
venv was built in. The devbox venvs live in the main checkout (`/mnt/efs/jasonwei/rl-distill`) and point at it, not at a
worktree. Check with `.venv/bin/python -c "import verl; print(verl.__file__)"`. Pods are unaffected (the tarball is unpacked
over `/workspace/rl-distill`, which the image venv points at).

RL entry point: every RL/OPD run goes `run_dapo_rl.sh` -> `python3 -m dapo.main_dapo`, so `dapo/dapo_ray_trainer.py` is what
executes. `rl-distill-scripts/dapo_ray_trainer.py` is currently a byte-identical copy; edit `dapo/` (or both, deliberately).

## 2. Launch path on ScaleTrain

```
devbox                                                          pod (p5.48xlarge[:N], ns train)
start_<x>_supervisor.sh  (tmux rl-<name>)                       sudo -E K=V ... bash -c
 └ supervise_borrowing_job.py  (state: .scale_train_supervisors/)    "/workspace/rl-distill/.venv/bin/aws s3 cp <tarball> /tmp/...
    └ launch_<x>.sh            (recipe -> ENV_VARS, S3 URIs,          && /bin/tar -xzf ... --unlink-first --recursive-unlink
       │                         WANDB_RUN_ID, GPU count)              && echo CODE_BOOTSTRAPPED && exec bash <run-file>"
       └ launch_st_with_code.sh (git archive CODE_REF -> S3,       └ run_gemma4_pt_deepscaler_4of4strict_rl.sh (RL + OPD)
          │                       pinned --image)                     build /tmp/.venv-gemma4 -> completion preflight ->
          └ launch_st_job.py    (k8s-preset job YAML)                 init model -> data prep -> restore-latest ->
             └ scale-train train --image <uri>                         run_dapo_rl.sh -> publish-best-hf -> complete-run ->
                                                                       wandb sync -> RUN_DONE rc=N
```

**Image + code tarball** ([../scale_train/launch_st_with_code.sh](../scale_train/launch_st_with_code.sh)). Image builds are
not part of the normal loop: `--build-env remote` returned "Image creation scheduled" but never produced a tag (2026-09-07), and
local builds need an ECR-capable profile. Instead every launch reuses the known-good image

```
692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a
```

(override with `ST_IMAGE`) and ships the code as `git archive` of a commit, uploaded with `AWS_PROFILE=ml-worker` to
`s3://scale-ml/genai/rl-distill/code/rl-distill-code-<sha8>.tar.gz` (reused if already there). **Uncommitted changes are not
shipped**: commit first (the wrapper warns about dirty files under `scale_train/`).

**CODE_REF pinning.** `CODE_REF` (default `HEAD`) picks the commit. The RL / OPD start scripts resolve it once
(`git rev-parse --short=8`) and bake it into the supervisor's launch command, so every relaunch of a run ships the code the run
started on, even after `HEAD` moves. The `pgk1-v1` OPD batch is pinned to `5c6d47d5` (before the distillation
`global_batch_info` ordering fix); new batches use a new `RUN_TAG` with `CODE_REF=HEAD` or a later pin. The distill / eval
start scripts do not pin (each relaunch ships the then-current `HEAD`).

**[launch_st_job.py](../scale_train/launch_st_job.py) flags** (the wrapper always adds `--n-instances 1 --image ... --code-s3-uri ...`):

| Flag | Default | Notes |
|---|---|---|
| `--cluster eks\|gke\|local` | `eks` | EKS = `p5.48xlarge` (`ml-gpu-batch`); GKE = `a3.megagpu.8g`. Exports `SCALE_CLUSTER` to the pod. |
| `--gpus-per-instance 1\|2\|4\|8` | 8 | `<8` uses the fractional preset `p5.48xlarge:N` (single instance only). |
| `--priority normal\|high` | `high` | All launchers use `high`. |
| `--allow-borrowing` | off | Borrowed (preemptible) capacity. See the policy below. |
| `--active-deadline-hours` | 240 | Launchers: RL 240, OPD/distill 120, math suite 48. |
| `--team / --product` | `egp` / `train.enterprise_rlvr` | |
| `--job-name` | `rl-distill` | Becomes `<name>-<user>` truncated to 32 chars; pods are `<job>-<user>-...-workers-0-0-<rand>`, label `scaletrain/job_id=<id>`. |
| `--run-file` | RL run-file | **Resolved relative to `scale_train/`** (bare filename or `gemma4/<file>`), or absolute; must live inside the repo. A repo-root-relative path doubles to `.../scale_train/rl-distill-scripts/scale_train/...` and the pod `bash`es a missing file. |
| `--env-vars K=V,K=V` | | Comma-split; a fragment without `=` is glued back onto the previous value, so values may contain commas but not `,X=` sequences. |
| `--dry-run` | | Prints the redacted command and the `scale-train train` argv, launches nothing. |

Job commands are rendered through `string.Template`: never put `$(...)` or `$VAR` in an env value or bootstrap. The pod runs the
command in a login shell that resets `PATH` (image `ENV PATH` is lost), hence the absolute `aws`/`tar` paths.

Dry runs:

```bash
cd rl-distill-scripts/scale_train
python3 launch_st_job.py --n-instances 1 --gpus-per-instance 4 --image "$IMG" --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh \
  --env-vars GEMMA4_MODEL=Qwen/Qwen3-4B-Base --dry-run
bash launch_qwen3_opd_recipe_v2_batch.sh            # batch launchers: no flag = print each spec's resolved env
```

**Borrowing vs dedicated.** Current rule (2026-10-09): new Qwen jobs launch with borrowing **off**; if a job sits QUEUED
for more than 60 min, switch it to borrowing on with automatic preemption resume (the fallback watcher, §3). Launcher defaults
differ, so set `ALLOW_BORROWING` explicitly: Qwen RL `true`, OPD `false`, Qwen distill `true`, math suite / pass@32 `false`,
Gemma 4 RL seeds always `--allow-borrowing`. Fractional pools can starve when the fleet is full of 8-GPU jobs; when queue time
matters, a borrow-on and a borrow-off copy of the same run may be queued as a race (cancel the loser by ID).

## 3. Supervision

Long jobs are never launched bare: a `start_*_supervisor.sh` script starts
[supervise_borrowing_job.py](../scale_train/supervise_borrowing_job.py) in a detached tmux session `rl-<name>`, which submits
the job, polls it, and relaunches it until a durable completion marker exists on S3. The run-files resume from S3 on every start,
so a relaunch costs only the work since the last checkpoint.

| Start script | tmux / name | Done when |
|---|---|---|
| `start_qwen3_30b_a3b_medium_rl_supervisor.sh` (`QWEN_SIZE=30b-a3b\|8b\|4b\|1.7b`, `BAND=easy\|medium\|hard\|very_hard`) | `rl-q3-<size>-rl-<med\|hard\|...>` + `ckpt-janitor-<name>` | `run_complete.json` (step <= `TOTAL_TRAINING_STEPS`, world size = GPUs) + `best_hf/_REMOTE_COMPLETE.json` |
| `start_qwen3_onpolicy_distill_supervisor.sh` (`SPEC=...`) | `rl-q3-opd-<SPEC>` + janitor | same two RL markers |
| `start_qwen3_30bm_to_4b_distill_supervisor.sh` (`TEACHER_SPEC`, `STUDENT`) | `rl-q3-<t>-dist-<s>` | every `step_<N>/summary.json` eval marker |
| `start_qwen3_math_suite_supervisor.sh`, `start_qwen3_passk32_evals_supervisor.sh` | `rl-q3-ms-*`, `rl-q3-passk32-evals` | every `<model>/<set>/summary.json` |
| `gemma4/start_gemma4_rl_seed_supervisors.sh` | `rl-g4-<size>-s<seed>-<band4>` | RL markers |
| `gemma4/start_gemma4_12br_trace_distill_supervisor.sh` | `rl-g4-...-trace-distill-lr<LR>` | ScaleTrain COMPLETED only (no receipts; a preemption misread as COMPLETED needs a manual relaunch: check `global_step_1000/_REMOTE_COMPLETE.json`) |

**Supervisor policy flags** (the start scripts pass most of these; env overrides in parentheses):

- `--relaunch-on-cancel`: an external CANCELED is treated like a preemption. `--quick-cancel-seconds 300` /
  `--max-quick-cancels 3`: consecutive cancels within 5 min of submit back off and then stop the supervisor (2026-09-13: 13
  relaunches in 8 min while ScaleTrain cancelled every submission).
- `--relaunch-on-failure` with `--failure-backoff-seconds` (`FAILURE_BACKOFF_SECONDS`, default 900 for RL/OPD, 3600 for evals)
  times consecutive failures (capped 10x), capped by `--max-failure-backoff-seconds` (`MAX_FAILURE_BACKOFF_SECONDS`). Equal values
  = fixed retry interval; for infra-only failures (e.g. a node reporting `GetPreferredAllocation ... GPU is lost` at admission)
  use `MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600 MAX_FAILURE_BACKOFF_SECONDS=3600` = retry hourly on EKS.
- `--max-relaunches` (`MAX_RELAUNCHES`, default 40 RL/OPD, 200 evals): crash-loop guard.
- Completion: `--completion-s3-uri` (+ `--max-completion-step`/`--expected-completion-step` and `--expected-completion-world-size`),
  `--completion-best-hf-s3-uri`, `--completion-marker-s3-uri` (any JSON object, repeatable). A COMPLETED job without its markers
  is a misreported preemption: the old job is cancelled and verified retired before a replacement is submitted (prevents two jobs
  writing one checkpoint prefix); 3 consecutive COMPLETEDs within `--quick-completion-seconds 1800` of running stop the supervisor
  (`STOPPED_MISSING_COMPLETION`).
- Status comes from `kubectl get pods -l scaletrain/job_id=<id>`: pod failure with reason Evicted/Preempted/Shutdown, a
  `DisruptionTarget` condition, or exit 137/143 (not OOMKilled) = PREEMPTED. A finished pod whose k8s Job is suspended (Kueue
  preemption) is treated as QUEUED, not COMPLETED. `scale-train get job` is consulted only when there are no pods.
- `--initial-job-id` (`INITIAL_JOB_ID=job_...`): adopt an already-submitted job instead of launching a duplicate (use this
  when restarting a supervisor whose job is still alive).
- `--pod-log-dir`: streams `kubectl logs --follow` of every attempt into `pod-logs/attempt-NNN-<job>-<pod>.log`.

**State dir** `.scale_train_supervisors/<name>-<YYYYMMDD><SUP_TAG>/` (OPD: `<name>-<RUN_TAG><SUP_TAG>`), relative to the repo root
the start script ran from:

```
STOP                 # touch to stop the loop (checked every poll); does NOT cancel the job
BORROW_SWITCHED      # written by the fallback watcher
state.json           # {name, attempt, job_id, status, image, updated_at}  <- the current job ID
monitor.log          # one line per event (submitted / status=... / relaunching / stopping)
supervisor.out       # supervisor stdout + launcher echo (LAUNCH_ENV, CODE_TARBALL); the watcher parses it
launch.log           # 0600; launcher return codes + job IDs only (the rendered command carries credentials)
pod-logs/            # attempt-NNN-<job>-<pod>.log (0600)
rolling-janitor.log  # RL/OPD only
stop-at-step.log     # OPD stop/republish scripts
```

**tmux targets must be exact-match**: always `tmux kill-session -t "=rl-<name>"` / `tmux has-session -t "=rl-<name>"`. A bare
`-t rl-q3-opd-q4bm-to-q1p7b` prefix-matches the sibling `...-to-q1p7b-base` session and kills/finds the wrong one.

**Rolling-checkpoint janitors.** The pod role (`eks-ml-worker2`) has no `s3:DeleteObject`, so the trainer's own rolling
prune/retire logs `retirement_deferred` (harmless). The RL/OPD start scripts also start tmux `ckpt-janitor-<name>`, which every
60 s runs, from the devbox with the delete-capable identity:

```bash
AWS_PROFILE=ml-worker AWS_REGION=us-west-2 .venv/bin/python rl-distill-scripts/full_checkpoint_s3.py cleanup-rolling --s3-uri <ckpt prefix>
```

It deletes only superseded rolling steps (never the committed rolling pointer or anything newer; it drops the rolling step once
an equal-or-newer permanent checkpoint exists) and exits with its supervisor (`STOP` file or `rl-<name>` gone). Superseded
rolling steps are large (~370 GB each for Qwen3-30B-A3B), so a missing janitor is a cost leak.

**Borrowing fallback watcher** ([../scale_train/borrowing_fallback_watcher.sh](../scale_train/borrowing_fallback_watcher.sh),
tmux `borrowing-fallback`, log `.scale_train_supervisors/borrowing-fallback.log`, 5-min loop, `ONCE=1` for one pass): for each
entry of its hard-coded `JOBS` table (name | borrowing-false state dir | restart command) whose last status is QUEUED for more than
`QUEUE_LIMIT_MIN=60` minutes since submit: touch `STOP` + `BORROW_SWITCHED`, kill `=rl-<name>`, cancel exactly that job ID,
and rerun the start command with `ALLOW_BORROWING=true SUP_TAG=-brw`. New borrowing-false supervisors must be added to `JOBS`.

**Stopping a run cleanly** (only jobs you own and name; cancellation is irreversible and loses queue position):

```bash
sup=.scale_train_supervisors/<name>-<date>
touch $sup/STOP                                         # 1. supervisor + janitor exit at their next poll
tmux kill-session -t "=rl-<name>"; tmux kill-session -t "=ckpt-janitor-<name>"
job=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["job_id"])' $sup/state.json)
env -u AWS_PROFILE scale-train cancel job "$job"        # 2. by explicit ID only (deletes the k8s Job)
```

OPD runs that were launched with more steps than wanted: `SPEC=<spec> STOP_STEP=120 bash rl-distill-scripts/scale_train/stop_qwen3_opd_at_step.sh`
waits for the step's validation line in the pod log and its permanent `_REMOTE_COMPLETE.json`, gives W&B up to 15 min, then
stops supervisor + janitor, cancels the job from `state.json`, backfills the step into W&B if needed, and republishes `best_hf`.

**Restarting / adopting.** Re-running a start script kills its own `=rl-<name>` session and starts fresh (it removes `STOP`).
If the old job is still alive, pass `INITIAL_JOB_ID=job_...` so the new supervisor adopts it. Use a new `SUP_TAG` (e.g. `-ext`,
`-to1000`) to keep the old state dir.

## 4. Checkpoints and resume

All bulk state lives on local NVMe (`/tmp` = pod emptyDir; lost with the pod) with S3 as the durable copy.
`FULL_CHECKPOINT_S3_URI` = the checkpoint prefix; `RUN_ARTIFACT_S3_URI` = the artifact prefix (`best_hf`, `run_outcome.json`).

| | Cadence | Contents | S3 location |
|---|---|---|---|
| Permanent | `SAVE_FREQ=10`, last step, early stop, ESI expiry | `actor/{model,optim,extra_state}_world_size_W_rank_r.pt`, `actor/fsdp_config.json`, `actor/huggingface/` (HF export), `data.pt` (StatefulDataLoader cursor), `validation_early_stopping.json`, `_REMOTE_COMPLETE.json` (size manifest, written last) | `<ckpt>/global_step_N/` + `<ckpt>/latest_checkpointed_iteration.txt` |
| Rolling | `ROLLING_CHECKPOINT_FREQ=5` | same minus `huggingface/` | `<ckpt>/rolling/global_step_N/` + `rolling/latest_checkpointed_iteration.txt`; one slot, advanced only after the new step verifies, never moved backward |

- Permanent steps are kept forever on S3 (they are the best-step candidates); locally only `MAX_ACTOR_CKPT_TO_KEEP=2`.
- `ROLLING_CHECKPOINT_ENABLED=False ASYNC_CHECKPOINT_UPLOAD=True` (used for 30B-A3B): permanent saves only, uploaded in a background
  thread while training continues (30B checkpoint time ~34% -> ~10%). The next save and the run outcome join the upload first; a
  failed upload stops the run; a preemption mid-upload falls back to the previous complete checkpoint.
- Before every save the vLLM engines are put to sleep (the 30B OOMed at its first permanent save otherwise) and woken + resynced after.

**Pod start sequence** in [../scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh](../scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh):

1. `check-completion-max` (receipt valid at a step <= `TOTAL_TRAINING_STEPS` with matching world size, model, band, seed, W&B id,
   HF repo) and `check-best-hf`. Both valid -> `RUN_ALREADY_COMPLETE` + `RUN_DONE rc=0`. Exit code 3 = incomplete; anything else is fatal.
2. `assert_local_fs` on `CKPTS_DIR`, `RAY_DATA_HOME`, `DATA_DIR`, `HF_HOME` (§7).
3. `restore-latest --checkpoint-root $CKPTS_DIR --s3-uri $FULL_CHECKPOINT_S3_URI`: validates the permanent and rolling namespaces
   independently, takes the newest valid step (permanent wins ties), downloads into `.global_step_N.partial`, checks sizes and
   shard layout, renames, writes `latest_checkpointed_iteration.txt`; then `RESUME_MODE=auto`. A broken rolling upload falls back to
   the permanent history.
4. Train. At the end the trainer writes `run_outcome.json` (protocol `gemma4_rl_run_outcome_v1`: `final_step`, `best_step`,
   `best_score`, `stop_reason`), restoring an older best step's HF export from S3 if it is no longer local.
5. `publish-best-hf` (uploads `global_step_<best>/actor/huggingface/` to `<artifact>/best_hf/`, then `run_outcome.json`, then
   `best_hf/_REMOTE_COMPLETE.json`, protocol `gemma4_rl_best_hf_v1`), then `complete-run` (verifies the final step on S3, writes
   `<ckpt>/run_complete.json`, protocol `gemma4_rl_run_complete_v1`).
6. `wandb sync --sync-all` (backfills uploader failures; errors on new wandb are harmless), `RUN_DONE rc=<status>`.

**Exact resume vs weight-only continuation.**

- Exact resume = restarting the supervisor with the same `RUN_TAG` (same S3 prefixes, same `WANDB_RUN_ID`, `WANDB_RESUME=allow`).
  It restores Adam, LR scheduler + RNG, data cursor and early-stopping state. Keep batch size, data seed, response count and GPU
  count unchanged (shards are per world size).
- Weight-only continuation (lost Adam / only an HF export): `GEMMA4_INIT_MODEL_S3_URI=<s3 HF export>` (or `_PATH`, or
  `_REPO`/`_REVISION`/`_SUBFOLDER`); fresh optimizer with a short `ACTOR_LR_WARMUP_STEPS` (10); `DATALOADER_SKIP_BATCHES=<batches
  consumed>` replays the seeded sampler (valid only at global step 0; calibrate from `data.pt` `_sampler_iter_state/samples_yielded`);
  `TOTAL_TRAINING_STEPS` = remaining budget; a **new** `FULL_CHECKPOINT_S3_URI` (else restore-latest pulls the old checkpoint) and a
  **new** W&B id (re-logging into an old id drops steps <= its last step). Global step restarts at 0; expect a dip during warmup.
  Gemma 4 exports need the base model's `processor_config.json` (added automatically for S3 inits).

**Early stopping** (`dapo/validation_early_stopping.py`, state file `validation_early_stopping.json` in each checkpoint). Qwen RL:
metric `val-core/math/acc/mean@16`, patience 5, `EARLY_STOPPING_MISS_TOLERANCE=0.01` (a score within 1 pt of the best breaks the miss
streak, so a flat plateau never trips it). OPD v2: patience 10, `EARLY_STOPPING_MIN_DELTA=0.005`, `TEST_FREQ=SAVE_FREQ=10` so every
best step has an HF export. Restored state is logged as `EARLY_STOPPING_STATE_RESTORED`; a restored checkpoint whose state is already
triggered finalizes immediately (writes the outcome, no training).

**Extending a finished run** (exact resume to a larger step budget; S3 writes from the devbox with `AWS_PROFILE=ml-worker`):

1. Move the markers that make the preflight exit `RUN_ALREADY_COMPLETE`: `<ckpt>/run_complete.json` -> `run_complete.step<N>.json`
   (or into `<ckpt>/archive_first_run/` together with `run_outcome.json` and the best_hf marker), copy `best_hf/` to
   `best_hf_at_step<N>/` (consumers such as the math suite may read it) and remove `best_hf/_REMOTE_COMPLETE.json`.
2. Restart the supervisor with the same `RUN_TAG`, a larger `TOTAL_TRAINING_STEPS` and a new `SUP_TAG`.
3. If the run **early-stopped**, add `EARLY_STOPPING_RESET_PATIENCE_AT_STEP=<final step>`: when resuming from exactly that step the
   miss streak is cleared (all-time best kept). If the run trained with early stopping **off**, its checkpoints have no ES state:
   build one from W&B and pass it as `EARLY_STOPPING_LEGACY_STATE_B64`:

```bash
.venv/bin/python rl-distill-scripts/scale_train/build_es_migration_from_wandb.py --run-id q3-opd-<spec>-<tag>-s42 \
  --model Qwen/Qwen3-1.7B-Base --difficulty medium --checkpoint-step 250 --patience 10 --min-delta 0.005 --every 10
```

`--every` must equal the permanent-checkpoint cadence so the seeded best is a step with an HF export.

**Resharding between world sizes.** Checkpoint shards are loaded by `(world_size, rank)`; an 8-GPU checkpoint cannot resume on 4 GPUs.
[../utils/reshard_fsdp2_checkpoint.py](../utils/reshard_fsdp2_checkpoint.py) rewrites model/optim/extra_state shards on CPU
(bit-exact, rewrites `fsdp_config.json`, copies `data.pt` + ES state; 134 GB 12B step took ~9 min):

```bash
python rl-distill-scripts/utils/reshard_fsdp2_checkpoint.py --src-step-dir CKPT/global_step_130 \
  --dst-step-dir NEW_CKPT/global_step_130 --dst-world-size 4
```

Put the result under a fresh `CKPTS_DIR` with `latest_checkpointed_iteration.txt`, and use a **new** S3 prefix so restore-latest
never pulls the old-world-size shards (the supervisor's `--expected-completion-world-size` must match too).

**Distillation runs** use the same helper differently: the off-policy trainer saves every `SAVE_FREQ=250` steps to
`<family>-distill-ckpts-v1/<teacher>-to-<student>-.../global_step_N/` (`huggingface/` inside, `_REMOTE_COMPLETE.json` marker); a
relaunch reuses finished trace bundles (`COMPLETE.json`), the newest complete distill checkpoint, and skips evals whose
`summary.json` (uploaded last) exists.

## 5. Artifacts, S3 and W&B

**best_hf.** The run-file publishes the early-stopping best (or the final step when early stopping is off, `best_score: null`).
To publish the true best saved step afterwards, use [../scale_train/republish_best_hf.py](../scale_train/republish_best_hf.py)
(best W&B val metric among steps with a complete permanent checkpoint + HF export, optional `--max-step`, earliest on ties, step 0
ineligible). Order is crash-safe: delete the old marker first, then old objects, server-side copy, rewrite `run_outcome.json` (the
run-file's kept once as `run_outcome.runfile.json`), write + verify the marker last; also writes `best_hf/_RUN_OUTCOME.json`. Wrappers:

```bash
SPEC=q30bm-to-q4b bash rl-distill-scripts/scale_train/publish_qwen3_opd_best_hf.sh   # DRY_RUN=1 to print the selection
```

**HF Hub.** Off by default since 2026-09-15 (`HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False` in every launcher; `run_dapo_rl.sh`'s own
default is still `True`): permanent S3 checkpoints carry the HF snapshot. When enabled, pushes happen only on permanent saves
(`HF_PUSH_FREQ` must be a multiple of `SAVE_FREQ`), to `JWei05/DAPO-<model>-PT-DeepScaleR-<dataset>-seed<S>-<suffix>`.
[upload_fullckpt_to_hf.py](../scale_train/upload_fullckpt_to_hf.py) pushes a full resumable checkpoint (model + Adam + cursor) to the
Hub for cross-cluster resume; [mirror_hf_to_s3.py](../scale_train/mirror_hf_to_s3.py) copies `JWei05` Gemma repos to
`s3://scale-ml/genai/rl-distill/hf-mirror/JWei05/<repo>/` with `_MIRROR_COMPLETE.json` (stages under `/tmp`, resumable).

**S3 layout** under `s3://scale-ml/genai/rl-distill/`:

| Family | Contents |
|---|---|
| `code/` | `rl-distill-code-<sha8>.tar.gz` code tarballs |
| `qwen3-{1.7b,4b,8b,30b-a3b}-base-rl/qwen3-<size>-<band>-<RUN_TAG>/` | RL artifacts: `best_hf/`, `run_outcome.json` (+ `best_hf_at_step<N>/` after extensions). `RUN_TAG` default `es5` |
| `qwen3-<size>-base-rl-full-checkpoints/qwen30b-<band>-<RUN_TAG>/` | RL checkpoints: `global_step_N/`, `rolling/`, `run_complete.json`, `archive_first_run/` |
| `qwen3-opd/<SPEC>-<RUN_TAG>/`, `qwen3-opd-full-checkpoints/<SPEC>-<RUN_TAG>/` | on-policy distillation artifacts / checkpoints (`pgk1-v1`, `pgk1-base-v1`, `pgk1-base-lr1e6w50`) |
| `qwen3-{30b,4b}-rl-traces-topk128-v1/`, `qwen3-4b-base-traces-topk128-v1/` | off-policy teacher trace bundles (`<spec>/COMPLETE.json`) |
| `qwen3-{30b,4b}-rl-distill-ckpts-v1/`, `qwen3-4b-base-distill-ckpts-v1/` | off-policy distill checkpoints (`<teacher>-<band>-to-<student>-bs128-s1000-lr2e-6/global_step_N/huggingface`) |
| `qwen3-{30b,4b}-rl-distill-evals-v1/`, `qwen3-4b-base-distill-evals-v1/` | distill evals (`.../step_N/summary.json`) |
| `qwen3-math-suite-v1/<model>/<set>/`, `qwen3-passk32-evals-v1/`, `qwen3-{medium,hard,very_hard}-val-evals-v1/`, `qwen3-own-nll-v1/`, `qwen3-deepscaler-samples/` | eval outputs |
| `gemma4-difficulty-s<seed>-<date>/gemma4-<size>-<band>/` + `-full-checkpoints/<size>-<band>/` | Gemma 4 RL seed sweeps (also `gemma4-e4b-bs256-s4{5,6,7}-20260919*`, `gemma4-12b-from-*-rl*`, `gemma4-26b-from-e4bbase-distill-rl-full-checkpoints`) |
| `gemma4-bestckpt-traces-topk128-v2/`, `gemma4-e4b-base-traces-topk128-v1/` | Gemma 4 trace bundles (**the `-v1` bestckpt bundle is contaminated, use v2**) |
| `gemma4-{12bd,12bo,12br,26bd,26bw,e4b-base}-distill-ckpts-v1/`, `gemma4-distill-views-v2/`, `gemma4-*-evals-*`, `gemma4-e4b-base-student-passk-v1/` | Gemma 4 distill checkpoints, training views, evals |
| `hf-mirror/` | HF repo mirrors |

**W&B.** Entity `rl-distill`. RL and OPD log to project `DAPO`; distillation logs to `<family>-distill-v1` projects (e.g.
`qwen3-30b-rl-distill-v1`, `gemma4-e4b-base-distill-v1`). Run IDs are fixed so relaunches resume the same run (`WANDB_RESUME=allow`):

| Run | `WANDB_RUN_ID` |
|---|---|
| Qwen RL | `<q3-30b-a3b\|q3-8b\|q3-4b\|q3-1.7b>-<band>-<RUN_TAG>-s42-v1` |
| Qwen OPD | `q3-opd-<SPEC>-<RUN_TAG>-s42` |
| Gemma 4 RL seeds | `g4ds26b-<size>-<band>-s<seed>-<WANDB_TAG:-v1>` |

Only trust the single-line `step:N - key:value - ...` pod-log records (multi-line pprint dumps misparse). Backfill tools:
[wandb_backfill_step_from_pod_log.py](../scale_train/wandb_backfill_step_from_pod_log.py) re-logs one step into the same run
(`resume="must"`, no-op if present; needed when a job is cancelled right after a step);
[../utils/recover_wandb_from_podlog.py](../utils/recover_wandb_from_podlog.py) re-logs a whole pod log into a separate recovery run.
Reading history: `run.scan_history()` drops steps logged after a resume on resumed runs; query one key at a time with
`run.history(keys=[key], samples=100000)` and merge by `_step`.

## 6. Monitoring and debugging

**Tools that work from the devbox** (kubectl and the ScaleTrain CLI use the EC2 instance role, so unset `AWS_PROFILE`; S3 writes and
tarball uploads use `AWS_PROFILE=ml-worker`; always `AWS_REGION=us-west-2`):

```bash
aws eks update-kubeconfig --name ml-gpu-batch --region us-west-2 --alias ir-ml-gpu-batch   # if kubectl suddenly returns nothing (expired SSO)
env -u AWS_PROFILE kubectl get pods -n train -l scaletrain/job_id=<job_id>
env -u AWS_PROFILE kubectl logs -n train <pod> --tail=200          # describe pod for the IP / crash reason
python3 rl-distill-scripts/scale_train/list_team_jobs.py [egp]      # running + queued jobs with GPU counts (EKS only)
script -qec "scale-train get job <job_id>" /dev/null | grep -E "status|name"   # TTY-only CLI; raw output contains credentials
AWS_PROFILE=ml-worker aws s3 ls <ckpt prefix>/ ; aws s3 ls <ckpt prefix>/rolling/
tail -f .scale_train_supervisors/<state>/monitor.log
```

Quirks: there is no `scale-train logs`; `scale-train list jobs` is an interactive TUI; `get job` shows status but no reason. The
instance role can get/logs/describe pods and `get jobs -n train` but not exec, jobsets or workloads. Failed pods persist a while,
k8s Jobs are garbage-collected. An explicit `scale-train cancel` deletes the k8s Job; platform auto-cancels do not. Kueue
"reclamation within the cohort" preemptions are routine. GKE pod logs are not readable with this account (and GKE pods may lack AWS
credentials), which is one reason EKS is the default. A health check worth automating: scan `.scale_train_supervisors/*/monitor.log`
for FAILED / ERROR / PREEMPTED / CANCELED / stopping lines (skip dirs with `STOP`) and confirm each `rl-<name>` session is alive, plus
an S3 watcher on the newest `global_step_N/_REMOTE_COMPLETE.json` per run.

**Pod-log markers, in order:** `CODE_BOOTSTRAPPED`, `### building /tmp/.venv-gemma4`, `GEMMA4_ENV_CORE_OK`, `MATH_VERIFY_OK`,
`RAY_TEMP_DIR=`, `INIT_MODEL_S3 downloaded=` / `ONPOLICY_DISTILL teacher=`, `[FullCheckpointS3] restored complete source=<permanent|rolling>
step=N` (or `no completed remote checkpoint`), `EARLY_STOPPING_STATE_RESTORED`, `step:N - ...`, `FULL_CHECKPOINT_ASYNC_UPLOAD_{STARTED,DONE,FAILED}`,
`RUN_OUTCOME_WRITTEN`, `[BestHFS3] published verified outcome`, `run completion receipt`, `===== FORENSICS` (on failure: Ray worker
stderr tails, raylet, dmesg), `RUN_DONE rc=N`.

**Common failure signatures:**

| Symptom | Cause / fix |
|---|---|
| `ncclInvalidUsage ... Bootstrap : no socket interface found` | EKS pods are IPv6-only on `eth0`: `NCCL_SOCKET_FAMILY=AF_INET6`, `NCCL/GLOO_SOCKET_IFNAME=eth0` (run-file default on EKS; `AF_INET` on GKE). `run_dapo_rl.sh`'s own defaults (`lo`/`AF_INET`) are for single-host devbox runs. |
| `python3: command not found`, `aws: command not found` at pod start | login shell reset PATH; use absolute venv paths in job commands and run-files. |
| `NVIDIA driver too old (found version 12080)` | cu130 wheels on p5; use `GEMMA4_CUDA_VARIANT=cu129`. |
| `OSError: libnvrtc.so.13` in `import vllm` | unpinned torchcodec; keep the 0.15.0 pin. Compare the pod's `+ pkg==` install lines with a working venv. |
| every reward 0.0 | math-verify missing in the venv (the run-file guard `MATH_VERIFY_OK` catches it). |
| `EADDRINUSE` at vLLM / EngineCore init | two verl jobs on one host with the same `VERL_VLLM_PORT_BASE` (default 52000; verl overwrites `VLLM_PORT`). Give each a base >= 1000 apart (Qwen launchers use 54000). |
| `Expected mha_graph.execute(...).is_good()` / `CUDNN_STATUS_INTERNAL_ERROR` in `update_actor` (Gemma 4) | cuDNN SDPA; `VERL_GEMMA4_CUDNN_SDPA=0` (and `VERL_GEMMA4_EVAL_CUDNN_SDPA`). `TORCH_CUDNN_SDPA_ENABLED=0` alone does not reach the train forward. |
| `<image|>` tokens in samples, deflated logprobs (Gemma 4 serving) | architecture overridden to `Gemma4ForCausalLM`; load unified with the base model's `processor_config.json` next to the weights. |
| NCCL spin / hang in the first distill validation (Qwen untied heads) | `VERL_FSDP2_WRAP_EMBED_LM_HEAD=0`; untied models also need `VERL_FSDP2_LOCAL_LOAD=0` (the run-file sets it for Qwen). |
| Gemma 4 weight-sync assertion about bucket size | `++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192` (run-file default; the path is `checkpoint_engine.`, not `rollout.`). |
| `AccessDeniedException ... ecr:GetAuthorizationToken` / supervisor looping `launcher exited with code 1` | a build was attempted with `ml-worker`; pass `--image <known-good tag>`. `AccessDenied ... CreateMultipartUpload` = instance role writing `s3://scale-ml`; use `AWS_PROFILE=ml-worker`. |
| job rejected at admission with `GetPreferredAllocation ... GPU is lost` | a broken EKS node; retry hourly (§3). |
| supervisor `STOPPED_MISSING_COMPLETION` | the run-file finishes without writing its markers (e.g. a stale receipt or config mismatch); read the last pod log. |
| `The current node timed out during startup` (local Ray on the devbox) | box load; apply [../utils/patches/](../utils/patches/) to the venv's Ray (`RAY_RAYLET_START_WAIT_TIME_S`, forward `dashboard_agent_listen_port`) and pre-assign agent ports via `ray start --head ...` + `RUN_RAY_ADDRESS`. Prefer ScaleTrain for real training. |

## 7. Gotchas

- **Never write checkpoints, HF exports, traces, model caches or Ray spill to EFS** (`/mnt/efs`, `$HOME`). The devbox home is one
  NFSv4.1 connection shared by ~115 sessions; a 134 GB checkpoint write on 2026-09-15 stalled every shell on the box.
  `assert_local_fs.sh` fails such runs (`ALLOW_NFS_CHECKPOINTS=1` only for tiny smoke runs). Use `/tmp`, with S3 as the copy.
- EKS pods: `NCCL_SOCKET_FAMILY=AF_INET6` + `eth0`. GKE pods: IPv4.
- Pod login shells drop the image PATH: absolute binaries (`/workspace/rl-distill/.venv/bin/aws`, `/bin/tar`) and fail-fast markers.
- Job commands are `string.Template`-rendered: no `$(...)` / `$VAR` in env values or the bootstrap.
- `--run-file` is relative to `scale_train/`, not the repo root.
- Commit before launching; the pod runs `git archive <CODE_REF>`, never the working tree. Never edit a bash script while running
  instances read it (bash reads by byte offset); long scripts wrap their body in `main() { ...; }; main "$@"`.
- Use `VERL_VLLM_PORT_BASE`, not `VLLM_PORT`, to separate concurrent verl jobs on one host.
- Gemma 4: `VERL_GEMMA4_CUDNN_SDPA=0` on E4B (also ~25% faster); keep it on for 26B-A4B (~7% faster). A/B a new model before changing.
- Gemma 4: never override the vLLM architecture; unified load needs `processor_config.json`. Gemma 4 has no `<end_of_turn>` token,
  so its stops are text (`VERL_ROLLOUT_EXTRA_STOP`), which forces detokenization.
- Gemma 4 needs `.venv-gemma4` (no flash-attn): `attn_implementation=sdpa`, no remove-padding, wrap `Gemma4TextDecoderLayer`
  (`Gemma4UnifiedTextDecoderLayer` for 12B).
- The torchcodec pin in `setup_env_gemma4.sh` is load-bearing; transitive deps of the venv are otherwise unpinned (ray included).
- RL-logged `best@N` is a with-replacement bootstrap (best@16 behaves like pass@10); `mean@N` is exact. Use `eval/eval_math_passk.py`
  for real pass@k.
- Offline eval generation with `logprobs=128` is CPU-bound (~5 req/s per process); use `--predictive_topk_width 0` and 2 vLLM
  processes per GPU with a fixed KV cache (`kv_cache_memory_bytes`) rather than utilization-based sizing.
- Commit with `git commit --no-verify`: upstream verl's pre-commit hooks fail on this fork and exceed tool timeouts; a killed
  pre-commit run leaves unstaged changes in `~/.cache/pre-commit/patch<ts>-<pid>` (recover with `git apply`). Stage explicit paths
  (never `git add -A`); rebase only with `--autostash` and confirm the dirty-file count is unchanged.
- Never `pkill -f` / `pgrep -f` a pattern that appears in your own command line (it kills the calling shell, exit 144). Kill by
  explicit PID, `pkill -P <parent>`, or `nvidia-smi --query-compute-apps=pid` filtered by owner. Orphaned vLLM processes show as
  `VLLM::EngineCor` (15-char comm) or `VLLM::Worker_TP*`.
- `tmux kill-session` SIGHUPs the pane's process group, including background children of a non-interactive script. Start long
  local jobs with `setsid nohup ... & disown`; to stop a supervisor without killing its jobs, kill the supervisor PID instead.
- tmux targets: always `-t "=<name>"` (exact match).
- Only cancel ScaleTrain jobs you launched or were asked to cancel, by explicit job ID; list them first. Many unrelated jobs run
  under the same team.
- Don't resubmit while a replacement is QUEUED (duplicates write the same S3 prefix); after a cancel storm or failure, wait
  (~20 min) and test with one job before resubmitting a batch.
- The `ml-worker` profile can submit jobs and write S3 but cannot push to ECR (`ecr:GetAuthorizationToken`); kubectl and
  `scale-train` work with the instance role (`env -u AWS_PROFILE`), not `ml-worker`. Pods cannot delete S3 objects.
- Ray is unpinned in `.venv-gemma4` (the EFS devbox copy has 2.56.1, a fresh `/tmp` build got 2.58); the files in
  `utils/patches/` target 2.58 and must be reapplied per venv.
- `aws s3 cp` of a missing object retries for tens of seconds on the devbox; check existence with `aws s3 ls <exact key>` first.
