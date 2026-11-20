"""An SFT run's checkpoint as an HF model directory (run.json's config, ckpt/<step>.safetensors; the task's tokenizer files),
e.g. for their evaluation or transformers.

    python export.py --run runs/chess/sft/model_20m_16B --step 384 --out runs/chess/sft/model_20m_16B/hf/384
"""

import argparse
import os

from safetensors.torch import load_file

from files import read_json
from model import save_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = read_json(os.path.join(a.run, "run.json"))
    weights = load_file(os.path.join(a.run, "ckpt", f"{a.step}.safetensors"))
    save_model(a.out, weights, run["hf"], {"config": run["config"], "sft": {"run": a.run, "step": a.step, "init": run.get("init")}})
