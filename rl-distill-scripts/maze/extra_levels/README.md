# Extra start levels: federer helper scripts (2026-11-03 to 11-05)

These are copies of `/work2/jiahaoz4/maze_extra/` on federer. That tooling ran the runbook "Extra start levels for the
maze RL-curve law" (https://claude.ai/artifact/Fppdo1u52xm7SLEQCpcN5k).

- **Paths inside are federer's:** `/work2/jiahaoz4/...`, `~/TailRL`, `~/tailrl_maze_runs`.
- **Machine notes and every decision:** `/work2/jiahaoz4/maze_extra/NOTES.md` on federer.
- **Results:** PROGRESS_LOG 2026-11-03 and the Hub dataset's `extra/`.
- **TailRL knobs:** the knobs these runs set (`RAY_TEMP_ROOT`, Ray port bands 8-11, `LOG_PROB_MICRO_BATCH`,
  `VERL_VAL_SKIP_BOOTSTRAP=1`) are in `../tailrl_text_maze.patch`.

| Step | Scripts |
|---|---|
| Setup | `env.sh` (sourced first), `build_venv.sh`, `download_data.sh` |
| 1-2: pretraining re-run, screening, selection | `step1_launch.sh`, `prune_pretrain_rerun.py`, `archive_pretrain_rerun.py`, `mem_probe.py`, `check_brackets.py`, `launch_select.sh`, `merge_select_parts.py` |
| 3: RL | `rl_launch.sh`, `watchdog.sh`, `rl_ckpt_janitor.py`, `curve_check.py`, `finalize_l_done.sh` |
| Sharing GPUs with other users | `free_gpus.sh`, `handoff.sh`, `move_run.sh`, `move_after.sh`, `move_at_ckpt.sh`, `wait_event.sh`, `status.sh` |
| 4: extraction and fits | `step4_extract_fit.sh` (S), `step4_L.sh` (S + L), `cmid_per_curve.py`, `summarize_curves.py` |
| Publishing | `upload_extra.py` (Hub `extra/`); `dashboard/make_csvs.py` and `dashboard/index.html` (the "Maze RL-curve fits" dashboard, https://claude.ai/artifact/9sYzg9BYtegf7GXYGx9UoH) |
