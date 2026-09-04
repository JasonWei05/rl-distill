#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULT_S3_ROOT="${RESULT_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-rl-distill-base-evals-v2}"
SOURCE_REGISTRY="${SOURCE_REGISTRY:-rl-distill-scripts/config/gemma4_rl_distill_eval_sources.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-rl-distill-scripts/results/gemma4-rl-distill-base-evals-v2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

command -v aws >/dev/null
command -v uv >/dev/null
test -s "${SOURCE_REGISTRY}"

aws s3 cp "${RESULT_S3_ROOT%/}/_packed/PACKED_RUN_COMPLETE.json" - --no-progress >/dev/null

LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-}"
if [[ -z "${LOCAL_RESULTS_ROOT}" ]]; then
  LOCAL_RESULTS_ROOT="$(mktemp -d /tmp/gemma4-rl-distill-eval-results.XXXXXX)"
else
  mkdir -p "${LOCAL_RESULTS_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

echo "Syncing verified-complete S3 results to ${LOCAL_RESULTS_ROOT}"
aws s3 sync "${RESULT_S3_ROOT%/}/" "${LOCAL_RESULTS_ROOT}/" --only-show-errors

"${PYTHON_BIN}" rl-distill-scripts/audit_gemma4_rl_distill_eval_results.py \
  --results-root "${LOCAL_RESULTS_ROOT}" \
  --source-registry "${SOURCE_REGISTRY}" \
  --output-json "${OUTPUT_ROOT}/final_audit.json" \
  --output-markdown "${OUTPUT_ROOT}/results_table.generated.md"

uv run --no-project --with matplotlib \
  python rl-distill-scripts/plot_gemma4_rl_distill_eval_curves.py \
  --audit-report "${OUTPUT_ROOT}/final_audit.json" \
  --output-dir "${OUTPUT_ROOT}/curves"

echo "Final audit and curves complete"
echo "Raw synchronized results: ${LOCAL_RESULTS_ROOT}"
echo "Report artifacts: ${OUTPUT_ROOT}"
