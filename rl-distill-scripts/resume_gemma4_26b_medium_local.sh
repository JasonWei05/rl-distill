#!/usr/bin/env bash
# Thin wrapper: resume the Gemma 4 26B-A4B *medium*-band run. All knobs/docs live in resume_gemma4_26b_a4b_local.sh.
#   CKPTS_DIR=$HOME/gemma4-26b-medium-s42/ckpts bash rl-distill-scripts/resume_gemma4_26b_medium_local.sh
set -euo pipefail
BAND=medium exec bash "$(dirname "${BASH_SOURCE[0]}")/resume_gemma4_26b_a4b_local.sh" "$@"
