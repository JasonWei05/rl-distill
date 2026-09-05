#!/usr/bin/env bash
# Thin wrapper: resume the Gemma 4 26B-A4B *hard*-band run. All knobs/docs live in resume_gemma4_26b_a4b_local.sh.
#   CKPTS_DIR=$HOME/gemma4-26b-hard-s42/ckpts bash rl-distill-scripts/resume_gemma4_26b_hard_local.sh
set -euo pipefail
BAND=hard exec bash "$(dirname "${BASH_SOURCE[0]}")/resume_gemma4_26b_a4b_local.sh" "$@"
