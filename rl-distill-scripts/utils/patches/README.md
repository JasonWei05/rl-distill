# Local patches to the `/tmp/.venv-gemma4` Ray 2.58 install (overloaded shared devbox)

Apply after `setup_env_gemma4.sh` (the venv lives on local disk and is erased by a devbox restart):

```bash
R=/tmp/.venv-gemma4/lib/python3.12/site-packages/ray/_private
patch "$R/node.py"     < rl-distill-scripts/patches/ray-2.58-node-raylet-start-wait-env.patch
patch "$R/services.py" < rl-distill-scripts/patches/ray-2.58-services-forward-dashboard-agent-listen-port.patch
```

- `node.py`: the driver's wait for the raylet (`raylet_start_wait_time_s`) was hardcoded to 30 s; read `RAY_RAYLET_START_WAIT_TIME_S`.
- `services.py`: `start_raylet` never forwarded `dashboard_agent_listen_port`, so the raylet always blocked 15 s (hardcoded) on the
  agent's port file and aborted at load avg 500+. With the flag forwarded, `ray start --head` with all agent ports fixed skips the wait
  (the 2026-09-16 local 12B resume used this; its launcher is in git history, notes in docs/GEMMA4_RESULTS.md).
