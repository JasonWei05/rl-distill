#!/usr/bin/env bash
# Publish OOD eval results to the repo as models finish: regenerate the committed OOD summary and §8 of
# DISTILLATION_EXPERIMENTS.md, commit, rebase onto origin/main (auto-resolving the two auto-generated
# files: the registry takes upstream's version, the doc is regenerated), push.
#   --once   : publish if any model completed since the last commit, then exit
#   --watch  : loop, polling every PUBLISH_POLL_SECONDS (default 120) until PUBLISH_UNTIL_MODELS models are
#              complete (default: the registry size) or PUBLISH_STOP_FILE exists
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
VENV="${VENV:-${PROJECT_ROOT}/.venv-gemma4}"; PY="${VENV}/bin/python"
RESULTS_BASE="${RESULTS_BASE:-/tmp/gemma4_distill_study_eval/results}"
DOC=rl-distill-scripts/DISTILLATION_EXPERIMENTS.md
REGISTRY=rl-distill-scripts/config/gemma4_distill_study_eval_sources.json
SUMMARY=rl-distill-scripts/config/gemma4_distill_study_ood_summary.json
STOP_FILE="${PUBLISH_STOP_FILE:-/tmp/gemma4_distill_study_eval/publish.stop}"
log() { echo "[$(date -u +%FT%TZ)] PUBLISH $*"; }

regen_doc() {  # OOD from result files + committed summary; math cells kept from the doc (the math box owns them)
  "${PY}" rl-distill-scripts/update_distill_study_results_doc.py --results-base "${RESULTS_BASE}" \
    --summary "${SUMMARY}" --fallback-from-doc >/dev/null
}

publish_once() {
  # Drop the running eval queue's own regeneration of the doc/registry (it lacks --summary/--fallback).
  git checkout -q -- "${DOC}" "${REGISTRY}" 2>/dev/null || true
  (cd rl-distill-scripts && "${PY}" summarize_gemma4_ood_results.py --results-base "${RESULTS_BASE}" --output "config/$(basename "${SUMMARY}")" >/dev/null) || { log "summary generation failed"; return 1; }
  local complete_tags; complete_tags="$("${PY}" - "${SUMMARY}" <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); b=d["benchmarks"]
print(" ".join(sorted(t for t,m in d["models"].items() if all(x in m["ood"] for x in b))))
PY
)"
  local n_complete; n_complete=$(wc -w <<<"${complete_tags}")
  regen_doc || { log "doc regeneration failed"; return 1; }
  if git diff --quiet -- "${SUMMARY}" "${DOC}"; then log "nothing new (${n_complete} models complete)"; return 0; fi
  local newly; newly="$(git diff -- "${SUMMARY}" | grep -E '^\+ +"[a-z0-9_]+": \{$' | sed -E 's/.*"([a-z0-9_]+)".*/\1/' | tr '\n' ' ')"
  git add "${SUMMARY}" "${DOC}"
  git commit -q -m "Distill study evals: OOD results update (${n_complete} models complete${newly:+; new: ${newly% }})

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || { log "commit failed"; return 1; }
  local attempt
  for attempt in 1 2 3; do
    git fetch -q origin || { log "fetch failed"; return 1; }
    if [ "$(git rev-list --count HEAD..origin/main)" -gt 0 ]; then
      log "rebasing onto origin/main (+$(git rev-list --count HEAD..origin/main) upstream commits)"
      if ! git rebase -q origin/main 2>/dev/null; then
        local f
        for f in $(git diff --name-only --diff-filter=U); do
          case "$f" in
            "${REGISTRY}") git checkout --ours -- "$f" && git add "$f";;   # upstream's roster is authoritative
            "${DOC}") git checkout --ours -- "$f" && regen_doc && git add "$f";;  # regenerate on upstream's text
            "${SUMMARY}") git checkout --theirs -- "$f" && git add "$f";;  # ours (the rebased commit)
            *) log "unresolvable conflict in $f; aborting rebase"; git rebase --abort; return 1;;
          esac
        done
        GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || { log "rebase --continue failed"; git rebase --abort; return 1; }
      fi
    fi
    if git push -q origin main 2>/dev/null; then log "pushed $(git rev-parse --short HEAD): ${n_complete} models complete"; return 0; fi
    log "push rejected (attempt ${attempt}); refetching"
  done
  log "push failed after 3 attempts"; return 1
}

case "${1:---once}" in
  --once) publish_once ;;
  --watch)
    POLL="${PUBLISH_POLL_SECONDS:-120}"
    TARGET="${PUBLISH_UNTIL_MODELS:-$("${PY}" -c "import json;print(len(json.load(open('${REGISTRY}'))['models']))")}"
    log "watching ${RESULTS_BASE} until ${TARGET} models are complete (poll ${POLL}s)"
    last=-1
    while [ ! -f "${STOP_FILE}" ]; do
      n=$(ls "${RESULTS_BASE}"/*/RUN_COMPLETE.json 2>/dev/null | wc -l)
      if [ "$n" -ne "$last" ]; then publish_once; last=$n; fi
      if [ "$n" -ge "${TARGET}" ]; then log "all ${TARGET} models complete; done"; exit 0; fi
      sleep "${POLL}"
    done
    log "stop file present; exiting" ;;
  *) echo "usage: $0 [--once|--watch]" >&2; exit 2 ;;
esac
