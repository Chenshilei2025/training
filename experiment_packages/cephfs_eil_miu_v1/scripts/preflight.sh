#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
MODE="${1:-}"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}"
MODEL_NAME="${LOYAL_MODEL_NAME:-Olmo-3-7B-Instruct}"
MODEL_HF_DIR="${LOYAL_MODEL_HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
MODEL_TD_DIR="${LOYAL_MODEL_REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}"
MIU_DATA_ROOT="${LOYAL_MIU_DATA_ROOT:-${PROJECT_ROOT}/miu/data/dataset/MIU-v2}"
EIL_DATA_ROOT="${LOYAL_EIL_DATA_ROOT:-${PROJECT_ROOT}/eil/data/dataset/EIL-v2}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"

cd "${PROJECT_ROOT}"

if [[ "${MODE}" != "--validate-only" ]]; then
  [[ -f "${CONDA_SH}" ]] || { echo "missing conda activation script: ${CONDA_SH}" >&2; exit 4; }
  [[ -d "${CONDA_ENV}" ]] || { echo "missing conda environment: ${CONDA_ENV}" >&2; exit 4; }
  [[ -x "${PYTHON}" ]] || { echo "missing python executable: ${PYTHON}" >&2; exit 4; }
  [[ -d "${MEGATRON_ROOT}" ]] || { echo "missing Megatron-LM runtime: ${MEGATRON_ROOT}" >&2; exit 4; }
else
  command -v "${PYTHON}" >/dev/null 2>&1 || PYTHON=python3
fi

"${PYTHON}" -m py_compile \
  scripts/experiment_runner.py \
  scripts/training/preflight.py \
  scripts/training/rollout/mixed.py \
  scripts/training/rewards/slime.py \
  scripts/evaluation/select_best_checkpoint.py \
  scripts/evaluation/watch_phase1_metrics.py

"${PYTHON}" -m scripts.experiment_runner \
  --config "${PKG_ROOT}/configs/e2m1_cephfs_rollout200.json" \
  --run-name "${LOYAL_RUN_NAME:-phase1}" \
  --validate-only

"${PYTHON}" - "${PKG_ROOT}/configs/e2m1_cephfs_rollout200.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
env = config["environment"]
assert config["mixed_ablation"]["ratio"] == "E2M1"
assert abs(float(env["LOYAL_MIXED_EIL_BATCH_FRACTION"]) - (2 / 3)) < 1e-9
assert int(env["LOYAL_MIXED_GLOBAL_BATCH_SIZE"]) == int(env["LOYAL_MIXED_ROLLOUT_BATCH_SIZE"]) * int(env["LOYAL_MIXED_SAMPLES_PER_PROMPT"])
assert int(env["LOYAL_MIXED_TRAIN_GPU_COUNT"]) + int(env["LOYAL_MIXED_ROLLOUT_GPU_COUNT"]) == int(env["LOYAL_MIXED_RAY_NUM_GPUS"])
assert int(env["LOYAL_MIXED_TENSOR_MODEL_PARALLEL_SIZE"]) == 2
assert env["LOYAL_MIXED_LEARNING_RATE"] == "2e-6"
assert int(env["LOYAL_MIXED_SCHEDULE_TOTAL_ROLLOUTS"]) == 200
assert env["LOYAL_REFUSE_CEPH_ACTIVE_PATHS"] == "0"
assert config.get("base_model") == "olmo3-7b-instruct"
assert config.get("target_model") == "allenai/Olmo-3-7B-Instruct"
assert config.get("target_model_local_dir") == "/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct"
print("cephfs_eil_miu_v1 config checks passed")
PY

for path in \
  "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" \
  "${PROJECT_ROOT}/scripts/evaluation/run_direct_checkpoint_eval.sh" \
  "${PROJECT_ROOT}/scripts/launch/prepare_model_checkpoint.sh" \
  "${PROJECT_ROOT}/scripts/launch/model_profiles.sh" \
  "${PROJECT_ROOT}/slime/scripts/models/olmo3-7B-Instruct.sh" \
  "${PROJECT_ROOT}/slime/slime_plugins/mbridge/olmo3.py" \
  "${PROJECT_ROOT}/slime/slime/backends/megatron_utils/megatron_to_hf/olmo3.py"
do
  [[ -e "${path}" ]] || { echo "missing required repo file: ${path}" >&2; exit 4; }
done

if [[ "${MODE}" == "--validate-only" ]]; then
  echo "cephfs_eil_miu_v1 validate-only passed"
  exit 0
fi

for path in \
  "${MIU_DATA_ROOT}/train.jsonl" \
  "${MIU_DATA_ROOT}/val.jsonl" \
  "${EIL_DATA_ROOT}/train.jsonl" \
  "${EIL_DATA_ROOT}/val.jsonl" \
  "${MODEL_HF_DIR}" \
  "${MODEL_TD_DIR}"
do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 4; }
done
[[ -f "${MODEL_TD_DIR}/config.json" || -f "${MODEL_TD_DIR}/latest_checkpointed_iteration.txt" || -f "${MODEL_TD_DIR}/common.pt" ]] || {
  echo "torch_dist checkpoint looks incomplete: ${MODEL_TD_DIR}" >&2
  exit 4
}
mkdir -p "${CEPH_ROOT}"
[[ -w "${CEPH_ROOT}" ]] || { echo "CephFS experiment root is not writable: ${CEPH_ROOT}" >&2; exit 4; }

echo "cephfs_eil_miu_v1 preflight passed"
