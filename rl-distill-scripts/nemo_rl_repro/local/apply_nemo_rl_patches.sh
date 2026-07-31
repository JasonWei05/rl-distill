#!/usr/bin/env bash
# Apply the repository-owned NeMo-RL source patches to the pinned submodule.
# The operation is idempotent and fails closed if the submodule revision drifts.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-${_REPO_ROOT}/third_party/nemo-rl}"
PATCH_FILE="${_REPO_ROOT}/rl-distill-scripts/nemo_rl_repro/patches/0001-batch-aware-local-logprob-chunking.patch"
EXPECTED_NEMO_COMMIT="5f89b3aec1fd08dbe12300441a54e9401c78ff8a"
MODE="${1:---apply}"

case "${MODE}" in
  --apply | --check) ;;
  *)
    echo "usage: $0 [--apply|--check]" >&2
    exit 2
    ;;
esac

if [ ! -s "${PATCH_FILE}" ]; then
  echo "FATAL: tracked NeMo-RL patch is missing: ${PATCH_FILE}" >&2
  exit 1
fi
if ! git -C "${NEMO_RL_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FATAL: NeMo-RL submodule is not initialized at ${NEMO_RL_ROOT}" >&2
  exit 1
fi

actual_commit="$(git -C "${NEMO_RL_ROOT}" rev-parse HEAD)"
if [ "${actual_commit}" != "${EXPECTED_NEMO_COMMIT}" ]; then
  echo "FATAL: NeMo-RL patch is pinned to ${EXPECTED_NEMO_COMMIT}, got ${actual_commit}" >&2
  echo "Update and revalidate the tracked patch before changing the submodule revision." >&2
  exit 1
fi

if git -C "${NEMO_RL_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "NEMO_RL_PATCH_OK already-applied commit=${actual_commit}"
  exit 0
fi

if ! git -C "${NEMO_RL_ROOT}" apply --check "${PATCH_FILE}"; then
  echo "FATAL: NeMo-RL checkout is neither cleanly patchable nor already patched" >&2
  git -C "${NEMO_RL_ROOT}" status --short >&2 || true
  exit 1
fi

if [ "${MODE}" = "--check" ]; then
  echo "NEMO_RL_PATCH_OK applicable commit=${actual_commit}"
  exit 0
fi

git -C "${NEMO_RL_ROOT}" apply "${PATCH_FILE}"
git -C "${NEMO_RL_ROOT}" apply --reverse --check "${PATCH_FILE}"
echo "NEMO_RL_PATCH_OK applied commit=${actual_commit}"
