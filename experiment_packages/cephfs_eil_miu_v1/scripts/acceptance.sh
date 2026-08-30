#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${LOYAL_RUN_NAME:-phase1}"
CONDITION="olmo3_e2m1_cephfs_rollout200"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
CHECKPOINT_NAME="cephfs-e2m1-${CONDITION}-${RUN_NAME}-seed1234"
CHECKPOINT_DIR="${LOYAL_CHECKPOINT_HOST_DIR:-${CEPH_ROOT}/checkpoints/${CHECKPOINT_NAME}}"
POST_ROOT="${LOYAL_POST_ROOT:-${CEPH_ROOT}/evaluations/${CONDITION}_posttrain}"
STEPS=(19 39 59 79 99 119 139 159 179 199)

errors=()
for step in "${STEPS[@]}"; do
  iter="$(printf "iter_%07d" "${step}")"
  [[ -s "${CHECKPOINT_DIR}/${iter}/common.pt" && -f "${CHECKPOINT_DIR}/${iter}/.metadata" ]] || errors+=("missing checkpoint ${iter}")
  [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/summary.json" ]] || errors+=("missing MIU summary step ${step}")
  [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/summary.json" ]] || errors+=("missing EIL summary step ${step}")
done
[[ -f "${POST_ROOT}/best_checkpoint.json" ]] || errors+=("missing best_checkpoint.json")

mkdir -p "${POST_ROOT}"
python3 - "${POST_ROOT}/acceptance.json" "${CHECKPOINT_DIR}" "${POST_ROOT}" "${RUN_NAME}" "${CONDITION}" "${#errors[@]}" "${errors[@]}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "status": "passed" if int(sys.argv[6]) == 0 else "failed",
    "checkpoint_dir": sys.argv[2],
    "post_root": sys.argv[3],
    "run_name": sys.argv[4],
    "condition": sys.argv[5],
    "expected_steps": [19, 39, 59, 79, 99, 119, 139, 159, 179, 199],
    "errors": sys.argv[7:],
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

if [[ "${#errors[@]}" -gt 0 ]]; then
  echo "acceptance failed: ${errors[*]}" >&2
  exit 10
fi

echo "acceptance passed: all checkpoints and EIL/MIU summaries exist"
