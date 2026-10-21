# Model registry of the Qwen math suite, sourced by run_qwen3_math_suite_st.sh (pod) and start_qwen3_math_suite_supervisor.sh
# (completion markers). suite_models <suite> prints space-separated name=source entries (source: hf:<repo>@<rev> | s3:<HF export>).
#   v1      the 2026-10-09 4B suite (4B RL medium / hard best, 30B -> 4B distilled medium / hard, 4B base)
#   q4b     4B very_hard RL best, 30B very_hard -> 4B distilled, + the v1 models (very_hard set)
#   q30b-4b 30B-A3B RL best x3 bands + 30B base + q4b
#   q1p7b   1.7B RL best x3 bands + 1.7B base, 30B -> 1.7B and 4B -> 1.7B distilled x3 bands
#   opd     the 9 on-policy-distilled students (best_hf of qwen3-opd/<spec>-pgk1-v1, best val step <= 120)
# RL best = run_outcome.json best_step (best_hf export); distilled = final distill step 1000.
suite_models() {
  local B=s3://scale-ml/genai/rl-distill b out=""
  local D30=${B}/qwen3-30b-rl-distill-ckpts-v1 D4=${B}/qwen3-4b-rl-distill-ckpts-v1 S=bs128-s1000-lr2e-6/global_step_1000/huggingface
  case "$1" in
    v1)
      out="q4b-rl-medium-best=s3:${B}/qwen3-4b-base-rl/qwen3-4b-medium-es5/best_hf q4b-rl-hard-best=s3:${B}/qwen3-4b-base-rl/qwen3-4b-hard-es5/best_hf"
      out+=" q30bm-to-q4b-s1000=s3:${D30}/q30bm-medium-to-q4b-${S} q30bh-to-q4b-s1000=s3:${D30}/q30bh-hard-to-q4b-${S}"
      out+=" q4b-base=hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539" ;;
    q4b)
      out="q4b-rl-very_hard-best=s3:${B}/qwen3-4b-base-rl/qwen3-4b-very_hard-es5/best_hf q30bvh-to-q4b-s1000=s3:${D30}/q30bvh-very_hard-to-q4b-${S}"
      out+=" $(suite_models v1)" ;;
    q30b-4b)
      for b in medium hard very_hard; do out+=" q30b-rl-${b}-best=s3:${B}/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-${b}-es5/best_hf"; done
      out+=" q30b-base=hf:Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
      out+=" $(suite_models q4b)" ;;
    q1p7b)
      for b in medium hard very_hard; do out+=" q1p7b-rl-${b}-best=s3:${B}/qwen3-1.7b-base-rl/qwen3-1.7b-${b}-es5/best_hf"; done
      out+=" q1p7b-base=hf:Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
      out+=" q30bm-to-q1p7b-s1000=s3:${D30}/q30bm-medium-to-q1p7b-${S} q30bh-to-q1p7b-s1000=s3:${D30}/q30bh-hard-to-q1p7b-${S}"
      out+=" q30bvh-to-q1p7b-s1000=s3:${D30}/q30bvh-very_hard-to-q1p7b-${S}"
      out+=" q4bm-to-q1p7b-s1000=s3:${D4}/q4bm-medium-to-q1p7b-${S} q4bh-to-q1p7b-s1000=s3:${D4}/q4bh-hard-to-q1p7b-${S}"
      out+=" q4bvh-to-q1p7b-s1000=s3:${D4}/q4bvh-very_hard-to-q1p7b-${S}" ;;
    opd)
      for b in q4bm-to-q1p7b q4bh-to-q1p7b q4bvh-to-q1p7b q30bm-to-q1p7b q30bh-to-q1p7b q30bvh-to-q1p7b q30bm-to-q4b q30bh-to-q4b q30bvh-to-q4b; do
        out+=" ${b}-opd120=s3:${B}/qwen3-opd/${b}-pgk1-v1/best_hf"
      done ;;
    *) echo "unknown math suite: $1" >&2; return 2 ;;
  esac
  echo ${out}
}
SUITE_SETS="easy medium hard very_hard gsm8k math500 aime2025 aime2026"
