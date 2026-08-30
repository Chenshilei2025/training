#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
RUN_NAME="${LOYAL_RUN_NAME:-phase1}"
CONDITION="olmo3_e2m1_cephfs_rollout200"
CHECKPOINT_NAME="cephfs-e2m1-${CONDITION}-${RUN_NAME}-seed1234"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
POST_ROOT="${LOYAL_POST_ROOT:-${CEPH_ROOT}/evaluations/${CONDITION}_posttrain}"
LOG_DIR="${LOYAL_LOG_DIR:-${CEPH_ROOT}/logs}"
mkdir -p "${POST_ROOT}" "${LOG_DIR}"

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export LOYAL_PROJECT_ROOT="${PROJECT_ROOT}"
export LOYAL_CEPHFS_ROOT="${CEPH_ROOT}"
export LOYAL_RUN_NAME="${RUN_NAME}"
export LOYAL_PYTHON="${PYTHON}"
export LOYAL_CONDA_SH="${CONDA_SH}"
export LOYAL_CONDA_ENV="${CONDA_ENV}"

bash "${PKG_ROOT}/scripts/preflight.sh"

nohup bash "${PKG_ROOT}/scripts/launch_streaming_cephfs_e2m1.sh" \
  >"${LOG_DIR}/${CONDITION}.log" 2>&1 &
echo "$!" >"${LOG_DIR}/${CONDITION}.pid"
nohup "${PYTHON}" -m scripts.evaluation.watch_phase1_metrics \
  --post-root "${POST_ROOT}" \
  --steps 19 39 59 79 99 119 139 159 179 199 \
  --interval "${LOYAL_METRICS_WATCH_INTERVAL:-300}" \
  >"${POST_ROOT}/metrics_watcher.log" 2>&1 &
echo "$!" >"${LOG_DIR}/${CONDITION}.metrics_watcher.pid"

printf 'started condition=%s\n' "${CONDITION}"
printf 'manager_pid=%s\n' "$(cat "${LOG_DIR}/${CONDITION}.pid" 2>/dev/null || true)"
printf 'metrics_watcher_pid=%s\n' "$(cat "${LOG_DIR}/${CONDITION}.metrics_watcher.pid" 2>/dev/null || true)"
printf 'checkpoint_dir=%s\n' "${CEPH_ROOT}/checkpoints/${CHECKPOINT_NAME}"
printf 'post_root=%s\n' "${POST_ROOT}"
