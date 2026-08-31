#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
CONDITION="olmo3_e2m1_cephfs_rollout200"
RUN_NAME="${LOYAL_RUN_NAME:-phase1}"
CHECKPOINT_NAME="cephfs-e2m1-${CONDITION}-${RUN_NAME}-seed1234"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
CHECKPOINT_ROOT="${LOYAL_CHECKPOINT_HOST_ROOT:-${CEPH_ROOT}/checkpoints}"
CHECKPOINT_DIR="${LOYAL_CHECKPOINT_HOST_DIR:-${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}}"
POST_ROOT="${LOYAL_POST_ROOT:-${CEPH_ROOT}/evaluations/${CONDITION}_posttrain}"
RUN_DIR="${LOYAL_RUN_DIR:-${CEPH_ROOT}/experiments/${CONDITION}-${RUN_NAME}}"
LOG_DIR="${LOYAL_LOG_DIR:-${CEPH_ROOT}/logs}"
mkdir -p "${POST_ROOT}" "${RUN_DIR}" "${LOG_DIR}" "${CHECKPOINT_ROOT}"

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export LOYAL_PROJECT_ROOT="${PROJECT_ROOT}"
export LOYAL_BASE_MODEL="olmo3-7b-instruct"
export LOYAL_MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}"
export LOYAL_MODEL_NAME="${LOYAL_MODEL_NAME:-Olmo-3-7B-Instruct}"
export LOYAL_MODEL_REPO="${LOYAL_MODEL_REPO:-allenai/Olmo-3-7B-Instruct}"
export LOYAL_MODEL_KEY="${LOYAL_MODEL_KEY:-olmo3-7b-instruct}"
export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/${LOYAL_MODEL_NAME}}"
export LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-${LOYAL_MODEL_ROOT}/${LOYAL_MODEL_NAME}_torch_dist}"
export LOYAL_MODEL_CHAT_TEMPLATE_KWARGS='{"add_generation_prompt":true}'
export LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
export LOYAL_CONDA_SH="${CONDA_SH}"
export LOYAL_CONDA_ENV="${CONDA_ENV}"
export LOYAL_CEPHFS_ROOT="${CEPH_ROOT}"
export LOYAL_CHECKPOINT_HOST_ROOT="${CHECKPOINT_ROOT}"
export LOYAL_CHECKPOINT_HOST_DIR="${CHECKPOINT_DIR}"
export LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
export LOYAL_MIXED_LOAD="${CHECKPOINT_DIR}"
export LOYAL_MIXED_SAVE="${CHECKPOINT_DIR}"
export LOYAL_PHASE1_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
export LOYAL_PHASE1_CHECKPOINT_ROOT="${CHECKPOINT_DIR}"
export LOYAL_PHASE1_POST_ROOT="${POST_ROOT}"
export LOYAL_PHASE1_RUN_DIR="${RUN_DIR}"
export LOYAL_PHASE1_EVAL_STEPS="19 39 59 79 99 119 139 159 179 199"
export LOYAL_PHASE1_FINAL_STEP=199
export LOYAL_PHASE1_RUN_REASONING="${LOYAL_PHASE1_RUN_REASONING:-0}"
export LOYAL_PHASE1_RUN_CREATIVE="${LOYAL_PHASE1_RUN_CREATIVE:-0}"
export LOYAL_PHASE1_BOOTSTRAP_REASONING="${LOYAL_PHASE1_BOOTSTRAP_REASONING:-0}"
export LOYAL_PHASE1_OUTPUT_ROOT="${CEPH_ROOT}/experiments"
export LOYAL_PHASE1_POST_LOG_FILE="${POST_ROOT}/posttrain_pipeline.log"
export LOYAL_PHASE1_LR_FILE="${POST_ROOT}/phase1_next_lr.txt"
export LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}"
export LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
export LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${CHECKPOINT_DIR}"
export LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST_ROOT}/checkpoint_eval"
export LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST_ROOT}/exported_models"
export LOYAL_DIRECT_EVAL_LOG_FILE="${POST_ROOT}/direct_checkpoint_eval.log"
export LOYAL_DIRECT_EVAL_STEPS="19 39 59 79 99 119 139 159 179 199"
export LOYAL_METRICS_WATCH_INTERVAL="${LOYAL_METRICS_WATCH_INTERVAL:-300}"
export LOYAL_MIXED_EIL_BATCH_FRACTION="0.6666666666666666"
export LOYAL_MIXED_GLOBAL_BATCH_SIZE="512"
export LOYAL_MIXED_ROLLOUT_BATCH_SIZE="64"
export LOYAL_MIXED_SAMPLES_PER_PROMPT="8"
export LOYAL_MIXED_MAX_RESPONSE_LEN="1024"
export LOYAL_MIXED_MAX_TOKENS_PER_GPU="4096"
export LOYAL_MIXED_TENSOR_MODEL_PARALLEL_SIZE="2"
export LOYAL_MIXED_LEARNING_RATE="2e-6"
export LOYAL_MIXED_KL_LOSS_COEF="0.05"
export LOYAL_MIXED_ENTROPY_COEF="0.002"
export LOYAL_MIXED_TRAIN_GPU_COUNT="2"
export LOYAL_MIXED_ROLLOUT_GPU_COUNT="2"
export LOYAL_MIXED_RAY_NUM_GPUS="4"
export LOYAL_MIXED_ENABLE_EVAL="0"
export LOYAL_MIXED_SAVE_INTERVAL="20"
export LOYAL_MIXED_SCHEDULE_TOTAL_ROLLOUTS="200"
export LOYAL_MIXED_OPTIMIZER_CPU_OFFLOAD="${LOYAL_MIXED_OPTIMIZER_CPU_OFFLOAD:-0}"
export LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC="${LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC:-0.78}"
export LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY="${LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY:-64}"
export LOYAL_RETAIN_ZERO_STD_GROUPS="${LOYAL_RETAIN_ZERO_STD_GROUPS:-1}"
export LOYAL_REFUSE_CEPH_ACTIVE_PATHS=0
export LOYAL_MIU_RECORDS="${PROJECT_ROOT}/miu/data/dataset/MIU-v2/train.jsonl:${PROJECT_ROOT}/miu/data/dataset/MIU-v2/val.jsonl"
export LOYAL_EIL_RECORDS="${PROJECT_ROOT}/eil/data/dataset/EIL-v2/train.jsonl:${PROJECT_ROOT}/eil/data/dataset/EIL-v2/val.jsonl"
export LOYAL_MIXED_TRAIN_RECORDS="${RUN_DIR}/mixed_train.jsonl"
export LOYAL_MIXED_LOAD="${CHECKPOINT_DIR}"
export LOYAL_MIXED_SAVE="${CHECKPOINT_DIR}"

if [[ ! -s "${LOYAL_MIXED_TRAIN_RECORDS}" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/prepare_mixed_slime.py" \
    --miu-source "${PROJECT_ROOT}/miu/data/dataset/MIU-v2/train.jsonl" \
    --eil-source "${PROJECT_ROOT}/eil/data/dataset/EIL-v2/train.jsonl" \
    --output "${LOYAL_MIXED_TRAIN_RECORDS}" \
    --seed 1234 >/dev/null
fi

bash "${PKG_ROOT}/scripts/preflight.sh"

cat >"${POST_ROOT}/launch_env.json" <<EOF
{
  "condition": "${CONDITION}",
  "checkpoint_name": "${CHECKPOINT_NAME}",
  "project_root": "${PROJECT_ROOT}",
  "run_dir": "${RUN_DIR}",
  "checkpoint_dir": "${CHECKPOINT_DIR}",
  "post_root": "${POST_ROOT}",
  "model_hf_checkpoint": "${LOYAL_MODEL_HF_CHECKPOINT}",
  "model_ref_load": "${LOYAL_MODEL_REF_LOAD}",
  "runtime_conda_env": "${CONDA_ENV}",
  "runtime_megatron_root": "${LOYAL_MEGATRON_ROOT}",
  "eval_steps": "${LOYAL_DIRECT_EVAL_STEPS}",
  "gpu_layout": "2 train (tp=2) + 2 rollout on ${LOYAL_PHASE1_GPUS:-0,1,2,3}"
}
EOF

train_step() {
  local step="$1"
  local target=$((step + 1))
  local log_file="${RUN_DIR}/train_to_${step}.log"
  local wait_seconds="${LOYAL_CHECKPOINT_WAIT_SECONDS:-60}"
  local max_wait_seconds="${LOYAL_CHECKPOINT_WAIT_MAX_SECONDS:-7200}"
  log() { printf '%s %s\n' "$(date -Is)" "$*"; }
  if [[ -f "${CHECKPOINT_DIR}/iter_$(printf '%07d' "${step}")/common.pt" && -f "${CHECKPOINT_DIR}/iter_$(printf '%07d' "${step}")/.metadata" ]]; then
    log "train_skip step=${step}"
    return 0
  fi
  LOYAL_MIXED_NUM_ROLLOUT="${target}" \
  LOYAL_EXPERIMENT_RESUME=1 \
    bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" mixed >"${log_file}" 2>&1 &
  local train_pid=$!
  local started now elapsed
  started="$(date +%s)"
  while [[ ! -s "${CHECKPOINT_DIR}/iter_$(printf '%07d' "${step}")/common.pt" || ! -f "${CHECKPOINT_DIR}/iter_$(printf '%07d' "${step}")/.metadata" ]]; do
    if ! kill -0 "${train_pid}" 2>/dev/null; then
      log "ERROR training_exited_early step=${step}"
      tail -n 120 "${log_file}" || true
      exit 6
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if [[ "${max_wait_seconds}" -gt 0 && "${elapsed}" -ge "${max_wait_seconds}" ]]; then
      log "ERROR checkpoint_timeout step=${step} elapsed=${elapsed}"
      exit 5
    fi
    log "waiting_checkpoint step=${step} elapsed=${elapsed}"
    sleep "${wait_seconds}"
  done
  if ! wait "${train_pid}"; then
    log "ERROR training_failed step=${step}"
    tail -n 120 "${log_file}" || true
    exit 6
  fi
  log "train_done step=${step}"
}

eval_step() {
  local step="$1"
  local final_dir="${POST_ROOT}/checkpoint_eval/step${step}"
  if [[ -f "${final_dir}/miu_final/summary.json" && -f "${final_dir}/eil_final/summary.json" ]]; then
    printf '%s eval_skip step=%s\n' "$(date -Is)" "${step}"
    return 0
  fi
  LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${CHECKPOINT_DIR}" \
  LOYAL_DIRECT_EVAL_STEPS="${step}" \
  LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST_ROOT}/exported_models" \
  LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST_ROOT}/checkpoint_eval" \
  LOYAL_DIRECT_EVAL_LOG_FILE="${POST_ROOT}/direct_checkpoint_eval_step${step}.log" \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_direct_checkpoint_eval.sh"
  printf '%s eval_done step=%s\n' "$(date -Is)" "${step}"
}

select_best() {
  local completed=()
  local step
  for step in 19 39 59 79 99 119 139 159 179 199; do
    [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/summary.json" && -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/summary.json" ]] || continue
    completed+=("${step}")
  done
  [[ "${#completed[@]}" -gt 0 ]] || return 0
  "${PYTHON}" -m scripts.evaluation.select_best_checkpoint \
    --root "${POST_ROOT}/checkpoint_eval" \
    --steps "${completed[@]}" \
    --output "${POST_ROOT}/best_checkpoint.json" || {
      printf '%s best_checkpoint_pending steps=%s\n' "$(date -Is)" "${completed[*]}"
      return 0
    }
}

main() {
  local step
  for step in 19 39 59 79 99 119 139 159 179 199; do
    train_step "${step}"
    eval_step "${step}"
    select_best
  done
  bash "${PKG_ROOT}/scripts/acceptance.sh"
  printf '%s streaming_manager_complete\n' "$(date -Is)"
}

main "$@"
