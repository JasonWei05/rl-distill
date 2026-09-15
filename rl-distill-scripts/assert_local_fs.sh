#!/usr/bin/env bash
# Refuse to put bulk-write directories (checkpoints, Ray temp, HF cache, datasets) on a network filesystem.
#
# Why: on the shared devbox the home directory is EFS mounted over ONE NFSv4.1 connection with 64 session slots for
# ~115 user sessions.  One trainer writing a 134 GB FSDP checkpoint there (2026-09-15, 12B medium local resume) queued
# ~57k NFS requests, slowed every other session's shell to ~30 s and collapsed our own write throughput to ~13 MB/s.
# Large artifacts belong on local NVMe (/tmp on the devbox, emptyDir on ScaleTrain) with S3 as the durable copy.
#
# Usage (sourced):   source "$(dirname "${BASH_SOURCE[0]}")/assert_local_fs.sh"
#                    assert_local_fs CKPTS_DIR="${CKPTS_DIR}" RAY_DATA_HOME="${RAY_DATA_HOME}" HF_HOME="${HF_HOME}"
# Escape hatch:      ALLOW_NFS_CHECKPOINTS=1 (logs a warning instead of failing; use only for tiny smoke runs).
assert_local_fs() {
  local spec name path probe fstype bad=0
  for spec in "$@"; do
    name="${spec%%=*}"; path="${spec#*=}"
    [ -n "${path}" ] || continue
    probe="${path}"
    while [ ! -e "${probe}" ] && [ "${probe}" != "/" ]; do probe="$(dirname "${probe}")"; done   # deepest existing ancestor
    fstype="$(stat -f -c %T "${probe}" 2>/dev/null || echo unknown)"
    case "${fstype}" in
      nfs|nfs4|cifs|smb2|fuse.sshfs|glusterfs|ceph)
        echo "LOCAL_FS_GUARD: ${name}=${path} is on ${fstype} (${probe}); bulk writes to a network filesystem stall every other session on this box" >&2
        bad=1 ;;
      *) ;;
    esac
  done
  if [ "${bad}" -ne 0 ]; then
    if [ "${ALLOW_NFS_CHECKPOINTS:-0}" = 1 ]; then
      echo "LOCAL_FS_GUARD: ALLOW_NFS_CHECKPOINTS=1 set -- continuing anyway (keep the run small)" >&2
      return 0
    fi
    echo "LOCAL_FS_GUARD: FATAL -- move these directories to local disk (/tmp on the devbox) or set ALLOW_NFS_CHECKPOINTS=1 for a small smoke run" >&2
    return 2
  fi
}
