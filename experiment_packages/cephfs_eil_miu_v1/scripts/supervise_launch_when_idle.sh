#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"

CONDITION="olmo3_e2m1_cephfs_rollout200"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
LOG_DIR="${LOYAL_LOG_DIR:-${CEPH_ROOT}/logs}"
SUPERVISOR_LOG="${LOYAL_SUPERVISOR_LOG:-${LOG_DIR}/${CONDITION}.supervisor.log}"
LAUNCH_LOG="${LOYAL_MANUAL_LAUNCH_LOG:-${LOG_DIR}/manual_launch.log}"
PID_FILE="${LOG_DIR}/${CONDITION}.pid"

POLL_SECONDS="${LOYAL_SUPERVISOR_POLL_SECONDS:-300}"
IDLE_REQUIRED_CHECKS="${LOYAL_SUPERVISOR_IDLE_REQUIRED_CHECKS:-2}"
GPU_MEM_IDLE_MB="${LOYAL_SUPERVISOR_GPU_MEM_IDLE_MB:-2000}"
GPU_UTIL_IDLE_PCT="${LOYAL_SUPERVISOR_GPU_UTIL_IDLE_PCT:-10}"
REQUIRED_GPU_COUNT="${LOYAL_SUPERVISOR_REQUIRED_GPU_COUNT:-4}"
DRY_RUN="${LOYAL_SUPERVISOR_DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${SUPERVISOR_LOG}"
}

olmo3_is_running() {
  pgrep -f "launch_streaming_cephfs_e2m1.sh|slime/train.py" >/dev/null 2>&1
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
}

gpu_is_idle() {
  local snapshot
  snapshot="$(gpu_snapshot)" || return 2
  SNAPSHOT="${snapshot}" python3 - "${REQUIRED_GPU_COUNT}" "${GPU_MEM_IDLE_MB}" "${GPU_UTIL_IDLE_PCT}" <<'PY'
import os
import sys

required = int(sys.argv[1])
mem_limit = int(sys.argv[2])
util_limit = int(sys.argv[3])
rows = []
for line in os.environ["SNAPSHOT"].splitlines():
    if not line.strip():
        continue
    fields = [part.strip() for part in line.split(",")]
    if len(fields) != 4:
        raise SystemExit(2)
    rows.append((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))

if len(rows) < required:
    raise SystemExit(1)
busy = [row for row in rows[:required] if row[1] > mem_limit or row[3] > util_limit]
raise SystemExit(0 if not busy else 1)
PY
}

log "supervisor_start project_root=${PROJECT_ROOT} condition=${CONDITION} required_gpus=${REQUIRED_GPU_COUNT} mem_idle_mb=${GPU_MEM_IDLE_MB} util_idle_pct=${GPU_UTIL_IDLE_PCT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry_run launch_script=${PKG_ROOT}/scripts/launch_cephfs_e2m1.sh launch_log=${LAUNCH_LOG}"
  gpu_snapshot | tee -a "${SUPERVISOR_LOG}" || true
  exit 0
fi

idle_checks=0
while true; do
  if olmo3_is_running; then
    log "olmo3_already_running pid_file=${PID_FILE}"
    exit 0
  fi

  if gpu_is_idle; then
    idle_checks=$((idle_checks + 1))
    log "gpu_idle_check_pass count=${idle_checks}/${IDLE_REQUIRED_CHECKS}"
  else
    idle_checks=0
    log "gpu_busy_waiting next_check_seconds=${POLL_SECONDS}"
    gpu_snapshot | tee -a "${SUPERVISOR_LOG}" || true
  fi

  if [[ "${idle_checks}" -ge "${IDLE_REQUIRED_CHECKS}" ]]; then
    log "launching_olmo3"
    bash "${PKG_ROOT}/scripts/preflight.sh" >>"${SUPERVISOR_LOG}" 2>&1
    bash "${PKG_ROOT}/scripts/launch_cephfs_e2m1.sh" >"${LAUNCH_LOG}" 2>&1
    if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      log "launch_started manager_pid=$(cat "${PID_FILE}") launch_log=${LAUNCH_LOG}"
      exit 0
    fi
    log "ERROR launch_returned_without_live_manager launch_log=${LAUNCH_LOG}"
    tail -n 120 "${LAUNCH_LOG}" | tee -a "${SUPERVISOR_LOG}" || true
    exit 6
  fi

  sleep "${POLL_SECONDS}"
done
