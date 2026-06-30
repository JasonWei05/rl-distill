#!/usr/bin/env python3
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

"""Launch rl-distill jobs on ScaleTrain.

This mirrors the lightweight launcher pattern used by SkyRL/slime: it creates a
temporary k8s-preset job config, then calls `scale-train train` with the local
build manifest and remote run environment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

JOB_CONFIG = {
    "job_type": "k8s-preset",
    "preset_instance_type": None,
    "num_instances": None,
    "job_name": None,
    "image": "${image}",
    "team": None,
    "product": None,
    "allow_borrowing": None,
    "priority": None,
    "active_deadline_seconds": None,
    "command": None,
}


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _parse_env_vars(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    parts = raw.split(",")
    merged: list[str] = []
    for part in parts:
        if "=" in part:
            merged.append(part)
        elif merged:
            merged[-1] += "," + part
    values = {}
    for item in merged:
        key, value = item.split("=", 1)
        values[key] = value
    return values


def _redacted(command: list[str], secret_keys: set[str]) -> list[str]:
    rendered = []
    for item in command:
        if "=" in item and item.split("=", 1)[0] in secret_keys:
            rendered.append(f"{item.split('=', 1)[0]}=<redacted>")
        else:
            rendered.append(item)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=["eks", "gke", "local"], default="eks")
    parser.add_argument("--n-instances", type=int, default=2)
    parser.add_argument("--team", default="egp")
    parser.add_argument("--product", default="train.enterprise_rlvr")
    parser.add_argument("--job-name", default="gemma3-12b-topk128")
    parser.add_argument("--priority", choices=["normal", "high"], default="high")
    parser.add_argument("--allow-borrowing", action="store_true")
    parser.add_argument("--active-deadline-hours", type=int, default=240)
    parser.add_argument("--build-config-key", default="train-rl-distill")
    parser.add_argument("--build-manifest-path", default="st_config/build_manifest.yaml")
    parser.add_argument("--env-build-values-path", default="st_config/.env.build.values")
    parser.add_argument("--run-file", default="run_gemma3_12b_pt_topk128_distill.sh")
    parser.add_argument("--container-project-root", default="/workspace/rl-distill")
    parser.add_argument("--env-vars", default=None)
    parser.add_argument("--dotenv", default="../../.env")
    parser.add_argument("--dotenv-keys", default="HF_TOKEN,WANDB_API_KEY,WANDB_BASE_URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated ScaleTrain command without launching",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    build_manifest = (here / args.build_manifest_path).resolve()
    env_build_values = (here / args.env_build_values_path).resolve()
    project_root = here.parents[1]
    if args.run_file.startswith("/"):
        local_run_file = Path(args.run_file).resolve()
    else:
        local_run_file = (here / args.run_file).resolve()

    env = _parse_env_vars(args.env_vars)
    dotenv_path = (here / args.dotenv).resolve()
    dotenv = _parse_env_file(dotenv_path)
    dotenv_keys = {key.strip() for key in args.dotenv_keys.split(",") if key.strip()}
    for key in dotenv_keys:
        if key in dotenv and key not in env:
            env[key] = dotenv[key]
    env.setdefault("SCALE_CLUSTER", args.cluster)
    env.setdefault("NNODES", str(args.n_instances))

    if args.cluster == "local":
        if args.n_instances != 1:
            raise SystemExit("--cluster local requires --n-instances 1")
        instance_type = "p5.48xlarge"
        run_env = "local"
    elif args.cluster == "gke":
        instance_type = "a3.megagpu.8g"
        run_env = "remote"
    else:
        instance_type = "p5.48xlarge"
        run_env = "remote"

    if run_env == "remote":
        try:
            run_relpath = local_run_file.relative_to(project_root)
        except ValueError as exc:
            raise SystemExit(f"--run-file must be inside {project_root} for remote launches: {local_run_file}") from exc
        run_file = str(Path(args.container_project_root) / run_relpath)
    else:
        run_file = str(local_run_file)

    command = ["sudo", "-E"] + [f"{key}={value}" for key, value in sorted(env.items())] + ["bash", run_file]
    secret_keys = dotenv_keys | {key for key in env if "TOKEN" in key or "KEY" in key}

    user = os.getenv("USER", "unknown")
    start = datetime.now().replace(microsecond=0).strftime("%y-%m-%d-%H-%M-%S")
    tmp_dir = Path.home() / "tmp_st_job_configs_rl_distill"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job_config_path = tmp_dir / f"rl_distill_st_job_config_{start}.yaml"

    job = JOB_CONFIG.copy()
    job.update(
        {
            "preset_instance_type": instance_type,
            "num_instances": args.n_instances,
            "job_name": f"{args.job_name}-{user}"[:32].rstrip("-"),
            "team": args.team,
            "product": args.product,
            "command": command,
            "priority": args.priority,
            "allow_borrowing": args.allow_borrowing,
            "active_deadline_seconds": args.active_deadline_hours * 3600,
        }
    )

    with job_config_path.open("w") as f:
        yaml.safe_dump(job, f, default_flow_style=False)

    print(f"build_manifest: {build_manifest}")
    print(f"env_build_values: {env_build_values}")
    print(f"job_config: {job_config_path}")
    print(f"command: {_redacted(command, secret_keys)}")

    launch_cmd = [
        "scale-train",
        "train",
        "--build-env",
        "local",
        "--run-env",
        run_env,
        "--build-manifest-path",
        str(build_manifest),
        "--job-config-path",
        str(job_config_path),
        "--build-config-key",
        args.build_config_key,
        "--env-build-values-path",
        str(env_build_values),
    ]
    print(launch_cmd)

    if args.dry_run:
        print("dry-run: not launching ScaleTrain job")
        try:
            job_config_path.unlink()
        except FileNotFoundError:
            pass
        return

    try:
        subprocess.run(launch_cmd, check=True)
    finally:
        try:
            job_config_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
