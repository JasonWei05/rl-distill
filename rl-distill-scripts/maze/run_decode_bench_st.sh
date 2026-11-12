#!/usr/bin/env bash
# ScaleTrain pod entry (1 GPU): maze/bench_prefix_decode.py on the 25x25 small and medium p3 seed-0 bases (exactness check
# + GraphDecoder vs PrefixSharedDecoder rollout times); the log goes to ${OUT_S3}/decode_bench.log.
#   env: OUT_S3 (required), BASES (space-separated S3 dirs; default the t3s / t3m p3_s0 bases)
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
: "${OUT_S3:?}"
P=s3://scale-ml/genai/rl-distill/maze
BASES="${BASES:-${P}/rl_small_dfs25_tis3s_v1/bases/p3_s0 ${P}/rl_medium_dfs25_tis3s_v1/bases/p3_s0}"
aws s3 cp --only-show-errors "${P}/data/dfs25_eps05/eval500/test.parquet" /tmp/maze/val500.parquet
LOG=/tmp/maze/decode_bench.log; : > "${LOG}"
trap 'aws s3 cp --only-show-errors "${LOG}" "${OUT_S3}/decode_bench.log" || true' EXIT   # upload the log even if a bench fails
i=0
for b in ${BASES}; do
  i=$((i + 1)); aws s3 sync --only-show-errors "${b}" "/tmp/maze/bench_base${i}"
  "${PY}" "${M}/bench_prefix_decode.py" --tailrl "${T}" --model "/tmp/maze/bench_base${i}" --val /tmp/maze/val500.parquet 2>&1 \
    | { grep -v -i "warn" || true; } | tee -a "${LOG}"
done
echo "DECODE_BENCH_DONE"
