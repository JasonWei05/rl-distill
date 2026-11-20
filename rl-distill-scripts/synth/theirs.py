"""Their SFT trainer as a pack job: pre2post-chess (github.com/pavelslab-nyu/pre2post-chess at 07d7a94, a copy under
--repo, unchanged) sft/scripts/train/run_sft.py under accelerate DDP on the visible GPUs, as their run_sft.sh runs it:
their qwen_multiturn_sft/sft_config_multi_path_2048.yaml, lr 3e-4, block 3072, the first 85 trace files of --data (the
last 2 held out), fp32, from --model (a pretrained base gets YaRN to 3072 and the 85-word vocabulary; a released SFT model
continues). Their HF checkpoints land every --save-interval steps in <out>/trajectory_sep_no_labels/<name>/step_<n> (a
relaunch resumes from latest_ckpt there); <out>/DONE at the end. python-chess comes from --pylib.

    python theirs.py --repo chess/pre2post-chess --pylib chess/pylib --model chess/ckpt/pretrain/model_20m_16B \
        --data chess/traces/fixed --out runs/chess/theirs_sft/model_20m_16B --save-interval 32
"""

import argparse
import os
import subprocess
import sys
import zlib

from files import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--repo", "--pylib", "--model", "--data", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--save-interval", type=int, default=100)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    out = os.path.abspath(a.out)
    cfg = OmegaConf.load(os.path.join(a.repo, "sft/config/configs/qwen_multiturn_sft/sft_config_multi_path_2048.yaml"))
    cfg.training.save_dir, cfg.training.save_interval = out, a.save_interval
    os.makedirs(out, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out, "their_config.yaml"))
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    name = os.path.basename(a.model.rstrip("/")).removeprefix("model_")
    repo = os.path.abspath(a.repo)
    env = {**os.environ, "PYTHONPATH": f"{os.path.abspath(a.pylib)}:{repo}:{repo}/sft", "WANDB_MODE": "disabled",
           "PYTHONWARNINGS": "ignore", "NCCL_TIMEOUT": "1800000", "TORCH_NCCL_BLOCKING_WAIT": "1"}  # run_sft.sh's
    subprocess.run([
        os.path.join(os.path.dirname(sys.executable), "accelerate"), "launch", "--num_processes", str(gpus), "--num_machines",
        "1", "--mixed_precision", "no", "--dynamo_backend", "no", "--main_process_port", str(20000 + zlib.crc32(out.encode()) % 20000),
        *(["--multi_gpu"] if gpus > 1 else []), f"{repo}/sft/scripts/train/run_sft.py", "--config", os.path.join(out, "their_config.yaml"),
        "--lr", "3e-4", "--block-size", "3072", "--max-train-files", "85", "--pretrained-model", os.path.abspath(a.model),
        "--pretrained-weights", os.path.abspath(os.path.join(a.model, "model.safetensors")),  # its YAML's is a placeholder
        "--train-files", os.path.abspath(a.data), "--cot-field", "cot_by_method.trajectory_sep.cot_format_no_labels",
        "--cot-type", "trajectory_sep_no_labels", "--naming-scheme", "pretrain_spec", "--total-compute", "na", "--modelsize",
        name, "--alpha", "na", "--beta", "na"], env=env, check=True)
    write_json(os.path.join(out, "DONE"), {"gpus": gpus})


if __name__ == "__main__":
    main()
