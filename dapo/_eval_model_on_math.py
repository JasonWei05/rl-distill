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

"""Evaluate one distilled model on the 8 DAPO math val sets.

Downloads the requested subfolder from a public HF repo, patches in the Gemma 3
VLM preprocessor configs that verl's saver omits, loads it into vLLM, iterates
over all --val_files with T=0.7/max=20k, scores with math_verify, writes a
summary JSON.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _strip_language_model_prefix_from_safetensors(local):
    marker = Path(local) / ".text_only_safetensors_rewritten"
    if marker.exists():
        return

    safetensors_files = sorted(Path(local).glob("*.safetensors"))
    if not safetensors_files:
        return

    from safetensors.torch import load_file, save_file

    drop_prefixes = (
        "multi_modal_projector.",
        "vision_tower.",
        "vision_model.",
        "visual.",
        "image_encoder.",
        "mm_projector.",
    )
    changed_any = False
    for path in safetensors_files:
        tensors = load_file(str(path), device="cpu")
        if not any(key.startswith("language_model.") or key.startswith(drop_prefixes) for key in tensors):
            continue
        rewritten = {}
        dropped = 0
        for key, tensor in tensors.items():
            if key.startswith(drop_prefixes):
                dropped += 1
                continue
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            rewritten[key] = tensor
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(rewritten, str(tmp))
        tmp.replace(path)
        changed_any = True
        msg = f"[prep] rewrote text-only weights in {path.name}"
        if dropped:
            msg += f" and dropped {dropped} multimodal tensors"
        print(msg)

    index_path = Path(local) / "model.safetensors.index.json"
    if changed_any and index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            rewritten_weight_map = {}
            for key, value in weight_map.items():
                if key.startswith(drop_prefixes):
                    continue
                if key.startswith("language_model."):
                    key = key[len("language_model.") :]
                rewritten_weight_map[key] = value
            index["weight_map"] = rewritten_weight_map
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            print("[prep] rewrote text-only safetensors index")

    if changed_any:
        marker.write_text("ok\n")


def prepare_model_dir(repo_id, subfolder, base_hf_model, staging_root, token):
    from huggingface_hub import hf_hub_download, snapshot_download

    tag = (subfolder or "root").replace("/", "_")
    local = os.path.join(staging_root, repo_id.replace("/", "_") + "__" + tag)
    os.makedirs(local, exist_ok=True)

    weight_markers = [f for f in os.listdir(local) if f.endswith(".safetensors")]
    if len(weight_markers) > 0 and os.path.isfile(os.path.join(local, "config.json")):
        print(f"[prep] using cached {local}")
    else:
        print(f"[prep] downloading {repo_id} subfolder={subfolder!r} -> {local}")
        if subfolder is None:
            # Root: grab all root-level files, skip step_* subfolders.
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=local,
                allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model"],
                ignore_patterns=["step_*/*"],
                token=token,
            )
        else:
            # Download subfolder into a staging dir, then move contents up.
            dl = local + "__dl"
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=dl,
                allow_patterns=f"{subfolder}/*",
                token=token,
            )
            src = os.path.join(dl, subfolder)
            for name in os.listdir(src):
                t = os.path.join(local, name)
                if not os.path.exists(t):
                    shutil.move(os.path.join(src, name), t)
            shutil.rmtree(dl, ignore_errors=True)

    # Patch with VLM preprocessor configs from the base Gemma 3 repo.
    for fname in ["preprocessor_config.json", "processor_config.json"]:
        if os.path.exists(os.path.join(local, fname)):
            continue
        try:
            hf_hub_download(repo_id=base_hf_model, filename=fname, local_dir=local, token=token)
        except Exception as e:
            print(f"[prep] could not fetch {fname} from {base_hf_model}: {e}")

    # verl's HF saver writes Gemma 3 PT language weights without SigLIP vision
    # weights, but the saved config can still advertise the multimodal
    # conditional-generation architecture. vLLM then selects gemma3_mm and
    # fails while looking for vision tensors. Force the text config when no
    # vision weights are present.
    config_path = Path(local) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            has_vision_weights = any(
                name.endswith(".safetensors") and "vision" in name.lower() for name in os.listdir(local)
            )
            if (
                config.get("model_type") == "gemma3"
                and isinstance(config.get("text_config"), dict)
                and not has_vision_weights
            ):
                text_config = dict(config["text_config"])
                for key in (
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "torch_dtype",
                    "dtype",
                    "tie_word_embeddings",
                    "transformers_version",
                ):
                    if key in config and key not in text_config:
                        text_config[key] = config[key]
                text_config["model_type"] = "gemma3_text"
                text_config["architectures"] = ["Gemma3ForCausalLM"]
                config_path.write_text(json.dumps(text_config, indent=2, sort_keys=True) + "\n")
                print("[prep] rewrote Gemma 3 config to text-only gemma3_text")
                _strip_language_model_prefix_from_safetensors(local)
        except Exception as e:
            print(f"[prep] could not patch text-only Gemma 3 config: {e}")

    return local


def _extract_ground_truth(row):
    rm = row.get("reward_model", None)
    if isinstance(rm, dict) and "ground_truth" in rm:
        return rm["ground_truth"]
    # numpy dict-like (pandas parquet round-trip can yield np.void)
    try:
        return rm["ground_truth"]
    except Exception:
        pass
    if "answer" in row:
        return row["answer"]
    if "ground_truth" in row:
        return row["ground_truth"]
    return None


def _extract_prompt(row):
    p = row["prompt"]
    if isinstance(p, list | tuple):
        return p[-1]["content"]
    # numpy array of dicts
    try:
        return p[-1]["content"]
    except Exception:
        return str(p)


def eval_one_dataset(llm, tokenizer, sampling, val_file, compute_score):
    import pandas as pd

    df = pd.read_parquet(val_file)
    prompts, gts = [], []
    for _, row in df.iterrows():
        text = _extract_prompt(row)
        msgs = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(formatted)
        gts.append(_extract_ground_truth(row))

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - t0

    scores, lens = [], []
    for output, gt in zip(outputs, gts, strict=False):
        comp = output.outputs[0]
        if gt is None:
            s = 0.0
        else:
            try:
                s = compute_score(comp.text, str(gt))
            except Exception:
                s = 0.0
        scores.append(float(s))
        lens.append(len(comp.token_ids))
    acc = sum(scores) / len(scores) if scores else 0.0
    mean_len = sum(lens) / len(lens) if lens else 0.0
    return {
        "n": len(prompts),
        "acc": acc,
        "acc_pass1": sum(1 for s in scores if s > 0.5) / len(scores) if scores else 0.0,
        "response_length_mean": mean_len,
        "gen_seconds": gen_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--base_hf_model", required=True)
    p.add_argument("--val_files", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=20480)
    p.add_argument("--max_model_len", type=int, default=22528)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    p.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="vLLM KV-cache block size. Default lets vLLM choose. "
        "Gemma 3 1B has head_size=256 which hits a FlashInfer "
        "bug at the default block_size=16; pass 32 or 64.",
    )
    p.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graphs / FlashInfer compile path. "
        "Use this when FlashInfer fails (e.g. Gemma 3 1B head_size=256 + ninja JIT).",
    )
    p.add_argument(
        "--attention_backend",
        default=None,
        help="vLLM attention backend (e.g. FLASH_ATTN, TRITON_ATTN). "
        "Default lets vLLM auto-select. Pass FLASH_ATTN for Gemma 3 1B "
        "on Blackwell to avoid the FlashInfer head_size=256 bug.",
    )
    p.add_argument(
        "--mm_encoder_attn_backend",
        default=None,
        help="vLLM Gemma 3 VLM encoder attention backend, e.g. TORCH_SDPA. "
        "Use TORCH_SDPA when FlashAttention PTX is incompatible with the "
        "node's NVIDIA driver/toolchain.",
    )
    p.add_argument("--staging_root", default="/tmp/eval_models")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.repo_id.split('/')[-1]}__{args.subfolder or 'root'}"
    print(f"=== evaluating {tag} ===", flush=True)

    token = os.environ.get("HF_TOKEN")
    model_dir = prepare_model_dir(args.repo_id, args.subfolder, args.base_hf_model, args.staging_root, token)

    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.math_verify import compute_score

    llm_kwargs = dict(
        model=model_dir,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.mm_encoder_attn_backend is not None:
        llm_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    summary = {
        "model": args.repo_id,
        "subfolder": args.subfolder or "root",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "per_dataset": {},
    }

    for vf in args.val_files:
        name = os.path.basename(vf).replace(".parquet", "")
        print(f"\n-- {name} --", flush=True)
        try:
            result = eval_one_dataset(llm, tokenizer, sampling, vf, compute_score)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            result = {"n": 0, "acc": 0.0, "error": str(e)}
        summary["per_dataset"][name] = result
        print(
            f"  n={result.get('n')}, acc={result.get('acc', 0):.4f}, "
            f"resp_len={result.get('response_length_mean', 0):.0f}, "
            f"t={result.get('gen_seconds', 0):.0f}s",
            flush=True,
        )

    out = os.path.join(args.output_dir, f"{tag}__summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
