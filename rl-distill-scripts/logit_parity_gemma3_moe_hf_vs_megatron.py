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

"""Compare a Gemma3 MoE HF checkpoint against its Megatron-Bridge load.

Loads the same converted MoE checkpoint twice — through the HF remote-code
model and through AutoBridge.load_hf_weights into the Megatron MoE model —
and compares next-token logits. This validates the fork's HF-to-Megatron
parameter mapping with real converted weights, including expert-parallel
sharding.

Single rank:
    CUDA_VISIBLE_DEVICES=4 python logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot>

Expert parallel (EP=2 across two GPUs):
    CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc-per-node 2 \
        logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot> --ep 2

Live 8-GPU topology, including a 2k-token context gate:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 \
        logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot> --tp 4 --ep 2 --seq-len 2048

``--seq-len`` repeats the tokenized prompt to the requested length. Pair it
with ``--dapo-data`` and ``--chat-template`` to use the exact rollout prompt.
It exercises the same RoPE positions and mixed local/global attention geometry
without relying on sampling.
"""

import argparse
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def _unwrap_activation(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output


def _capture_hook(target: dict[str, torch.Tensor], name: str, *, transpose_sequence: bool = False):
    def hook(_, __, output):
        value = _unwrap_activation(output).detach()
        if transpose_sequence:
            value = value.transpose(0, 1)
        target[name] = value.cpu()

    return hook


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist(tp_size: int, ep_size: int) -> int:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    return local_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Path to the converted MoE HF checkpoint")
    parser.add_argument("--tp", type=int, default=1, help="tensor_model_parallel_size")
    parser.add_argument("--ep", type=int, default=1, help="expert_model_parallel_size")
    parser.add_argument(
        "--sequence-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Megatron sequence parallelism (defaults to enabled when TP > 1).",
    )
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=None,
        help="Fail if mean absolute logit error exceeds this value.",
    )
    parser.add_argument(
        "--min-top1-agreement",
        type=float,
        default=1.0,
        help="Minimum tokenwise top-1 agreement required for success (default: 1.0).",
    )
    parser.add_argument(
        "--check-export",
        action="store_true",
        help="Also verify the MCore-to-HF weight stream used for vLLM resynchronization.",
    )
    parser.add_argument(
        "--force-expert-zero",
        action="store_true",
        help=(
            "Diagnostic control: zero every router so top-1 selects expert 0. "
            "This tests the full-batch canonical-expert initialization path."
        ),
    )
    parser.add_argument(
        "--report-activations",
        action="store_true",
        help="Report MLP and layer-output agreement at every decoder layer.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("flash", "fused", "unfused"),
        default="flash",
        help="Megatron attention backend for a controlled numerical-parity probe.",
    )
    parser.add_argument("--prompt", default="The capital of France is the city of")
    parser.add_argument("--dapo-data", help="Parquet data with the DAPO ``prompt`` chat-list column.")
    parser.add_argument("--chat-template", help="Chat template used with --dapo-data.")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Repeat the tokenized prompt until this many tokens are present.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_top1_agreement <= 1.0:
        raise ValueError("--min-top1-agreement must be between 0 and 1")
    expected_world_size = args.tp * args.ep
    configured_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if configured_world_size != expected_world_size:
        raise ValueError(
            f"WORLD_SIZE={configured_world_size}, but --tp {args.tp} * --ep {args.ep} "
            f"requires {expected_world_size} ranks"
        )

    _init_dist(args.tp, args.ep)
    rank = dist.get_rank()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    if args.dapo_data is not None:
        if args.chat_template is None:
            raise ValueError("--chat-template is required with --dapo-data")
        import pandas as pd

        messages = [dict(message) for message in pd.read_parquet(args.dapo_data).iloc[args.prompt_index]["prompt"]]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=Path(args.chat_template).read_text(),
        )
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    if args.seq_len is not None:
        if args.seq_len < input_ids.shape[1]:
            raise ValueError(f"--seq-len ({args.seq_len}) is shorter than the prompt ({input_ids.shape[1]})")
        repeats = (args.seq_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeats)[:, : args.seq_len]
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    # Megatron's 4-D boolean convention is ``True == masked``.  Supplying an
    # all-ones tensor therefore masks *every* attention score and produces a
    # meaningless HF/MCore comparison.  This is the same causal construction
    # used by Gemma3VLModel for text-only inputs.
    attention_mask = ~torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda"))

    from megatron.bridge import AutoBridge
    from megatron.core.process_groups_config import ProcessGroupCollection

    bridge = AutoBridge.from_hf_pretrained(args.snapshot, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = args.tp > 1 if args.sequence_parallel is None else args.sequence_parallel
    provider.variable_seq_lengths = True
    provider.seq_length = max(seq_len, 16)
    provider.gradient_accumulation_fusion = False
    from megatron.core.transformer.enums import AttnBackend

    provider.attention_backend = AttnBackend[args.attention_backend]
    # The Gemma bridge's dense provider still carries the generic MoE config
    # defaults.  ``alltoall`` is the dispatcher compatible with variable
    # sequence lengths, whether or not this particular model has experts.
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    if rank == 0:
        print(
            f"provider: {type(provider).__name__} experts={getattr(provider, 'num_moe_experts', 0)} "
            f"topk={getattr(provider, 'moe_router_topk', 'n/a')} "
            f"pre_softmax={getattr(provider, 'moe_router_pre_softmax', 'n/a')} "
            f"tp={provider.tensor_model_parallel_size} ep={provider.expert_model_parallel_size} "
            f"sequence_parallel={provider.sequence_parallel} seq_len={seq_len}"
        )

    megatron_model = provider.provide().cuda()
    bridge.load_hf_weights([megatron_model], args.snapshot)
    if args.force_expert_zero:
        with torch.no_grad():
            for name, parameter in megatron_model.named_parameters():
                if name.endswith(".mlp.router.weight"):
                    parameter.zero_()
    megatron_model.eval()
    # The dense Gemma3 checkpoint is a VLM container even for text-only
    # prompts.  Its wrapper computes a multimodal attention mask; for this
    # text parity test call the fully loaded language model directly.
    mcore_forward_model = getattr(megatron_model, "language_model", megatron_model)
    mcore_activations: dict[str, torch.Tensor] = {}
    mcore_handles = []
    if args.report_activations:
        for layer_idx, layer in enumerate(mcore_forward_model.decoder.layers):
            # The correctness-only canonical init branch invokes its
            # dense-compatible helper directly after routing, so hook that
            # semantic MLP boundary rather than the bypassed MoELayer.
            mcore_mlp_boundary = getattr(layer, "_canonical_dense_mlp", layer.mlp)
            mcore_handles.extend(
                [
                    layer.self_attention.register_forward_hook(
                        _capture_hook(
                            mcore_activations,
                            f"layer={layer_idx:02d}/post_attn_output",
                            transpose_sequence=True,
                        )
                    ),
                    layer.pre_mlp_layernorm.register_forward_hook(
                        _capture_hook(mcore_activations, f"layer={layer_idx:02d}/pre_ff_norm", transpose_sequence=True)
                    ),
                    mcore_mlp_boundary.register_forward_hook(
                        _capture_hook(mcore_activations, f"layer={layer_idx:02d}/mlp", transpose_sequence=True)
                    ),
                    layer.register_forward_hook(
                        _capture_hook(mcore_activations, f"layer={layer_idx:02d}/layer_output", transpose_sequence=True)
                    ),
                ]
            )
    with torch.no_grad():
        megatron_logits = mcore_forward_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            runtime_gather_output=True,
        ).detach()
    for handle in mcore_handles:
        handle.remove()
    if megatron_logits.shape[0] == seq_len:  # [s, b, v] -> [b, s, v]
        megatron_logits = megatron_logits.transpose(0, 1)
    megatron_logits = megatron_logits.float()

    parity_ok = True
    hf_model = None
    if rank == 0:
        from transformers import AutoModelForCausalLM

        hf_model = (
            AutoModelForCausalLM.from_pretrained(args.snapshot, trust_remote_code=True, dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        if args.force_expert_zero:
            with torch.no_grad():
                for name, parameter in hf_model.named_parameters():
                    if name.endswith(".mlp.router.weight"):
                        parameter.zero_()
        hf_activations: dict[str, torch.Tensor] = {}
        hf_handles = []
        if args.report_activations:
            for layer_idx, layer in enumerate(hf_model.model.layers):
                hf_handles.extend(
                    [
                        layer.post_attention_layernorm.register_forward_hook(
                            _capture_hook(hf_activations, f"layer={layer_idx:02d}/post_attn_output")
                        ),
                        layer.pre_feedforward_layernorm.register_forward_hook(
                            _capture_hook(hf_activations, f"layer={layer_idx:02d}/pre_ff_norm")
                        ),
                        layer.mlp.register_forward_hook(_capture_hook(hf_activations, f"layer={layer_idx:02d}/mlp")),
                        layer.register_forward_hook(
                            _capture_hook(hf_activations, f"layer={layer_idx:02d}/layer_output")
                        ),
                    ]
                )
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids).logits.detach().float()
        for handle in hf_handles:
            handle.remove()

        if args.report_activations:
            first_nonexact = None
            for layer_idx in range(len(hf_model.model.layers)):
                for component in ("post_attn_output", "pre_ff_norm", "mlp", "layer_output"):
                    key = f"layer={layer_idx:02d}/{component}"
                    hf_value = hf_activations[key]
                    mcore_value = mcore_activations[key]
                    if hf_value.shape != mcore_value.shape:
                        raise AssertionError(f"{key}: shape mismatch {hf_value.shape} != {mcore_value.shape}")
                    activation_diff = (hf_value.float() - mcore_value.float()).abs()
                    exact = torch.equal(hf_value, mcore_value)
                    print(
                        f"{key} exact={exact} max_abs={activation_diff.max().item():.6g} "
                        f"mean_abs={activation_diff.mean().item():.6g}"
                    )
                    if not exact and first_nonexact is None:
                        first_nonexact = key
            print(f"first_mcore_nonexact={first_nonexact or 'none'}")

        vocab = hf_logits.shape[-1]
        megatron_logits = megatron_logits[..., :vocab]
        diff = (megatron_logits - hf_logits).abs()
        hf_top1 = hf_logits.argmax(-1)
        mg_top1 = megatron_logits.argmax(-1)
        agree = (hf_top1 == mg_top1).float().mean().item()
        print(f"max_abs_diff={diff.max().item():.6g} mean_abs_diff={diff.mean().item():.6g}")
        print(f"top1_agreement={agree:.8f}")
        for position in sorted({0, seq_len // 2, seq_len - 1}):
            hf_log_probs = torch.log_softmax(hf_logits[0, position], dim=-1)
            mg_log_probs = torch.log_softmax(megatron_logits[0, position], dim=-1)
            hf_entropy = -(hf_log_probs.exp() * hf_log_probs).sum().item()
            mg_entropy = -(mg_log_probs.exp() * mg_log_probs).sum().item()
            print(
                f"position={position} hf_entropy={hf_entropy:.6f} "
                f"mcore_entropy={mg_entropy:.6f} "
                f"top1_match={bool(hf_top1[0, position] == mg_top1[0, position])}"
            )
        print("hf   next tokens:", tokenizer.decode(hf_top1[0][-8:]))
        print("mcore next tokens:", tokenizer.decode(mg_top1[0][-8:]))
        max_mean_abs_diff = args.max_mean_abs_diff
        if max_mean_abs_diff is None:
            # Tensor-parallel all-reduces introduce a small, expected numerical
            # delta relative to single-device HF. Top-1 agreement remains the
            # primary correctness gate in that topology.
            max_mean_abs_diff = 0.05 if args.tp == 1 else 0.10
        parity_ok = agree >= args.min_top1_agreement and diff.mean().item() < max_mean_abs_diff

    if args.check_export:
        export_count = 0
        unexpected_names: list[str] = []
        mismatched_names: list[str] = []
        max_weight_abs_diff = 0.0
        exported_names: set[str] = set()
        expected_state = hf_model.state_dict() if rank == 0 else None
        for name, weight in bridge.export_hf_weights([megatron_model], show_progress=False):
            if rank != 0:
                continue
            export_count += 1
            exported_names.add(name)
            expected_weight = expected_state.get(name)
            if expected_weight is None:
                unexpected_names.append(name)
                continue
            if tuple(weight.shape) != tuple(expected_weight.shape):
                mismatched_names.append(f"{name}: shape {tuple(weight.shape)} != {tuple(expected_weight.shape)}")
                continue
            weight_diff = (weight.float() - expected_weight.float()).abs().max().item()
            max_weight_abs_diff = max(max_weight_abs_diff, weight_diff)
            if weight_diff != 0.0:
                mismatched_names.append(f"{name}: max_abs_diff={weight_diff:.6g}")

        if rank == 0:
            # The tied LM head is intentionally absent from the converted
            # checkpoint; loading embed_tokens updates the same parameter.
            expected_names = set(expected_state) - {"lm_head.weight"}
            missing_names = sorted(expected_names - exported_names)
            print(
                f"export_count={export_count} expected_count={len(expected_names)} "
                f"max_weight_abs_diff={max_weight_abs_diff:.6g}"
            )
            if unexpected_names:
                print("unexpected_export_names:", unexpected_names[:8])
            if missing_names:
                print("missing_export_names:", missing_names[:8])
            if mismatched_names:
                print("mismatched_export_weights:", mismatched_names[:8])
            export_ok = not unexpected_names and not missing_names and not mismatched_names
            parity_ok = parity_ok and export_ok
            print("EXPORT_PARITY_OK" if export_ok else "EXPORT_PARITY_FAILED")

    if rank == 0:
        del hf_model
        torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert parity_ok, "HF/Megatron parity gate failed"
        print("LOGIT_PARITY_OK")


if __name__ == "__main__":
    main()
