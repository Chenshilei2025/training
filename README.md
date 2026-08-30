# OLMo3 EIL/MIU Training Package

This repository is the clean training workspace for the CephFS-backed OLMo3
EIL/MIU experiment.  The supported entrypoint is:

```bash
cd /root/training
cp experiment_packages/cephfs_eil_miu_v1/env.example .env
# Fill judge/adversary endpoints and keys in .env.
bash experiment_packages/cephfs_eil_miu_v1/scripts/download_model_cephfs.sh
bash experiment_packages/cephfs_eil_miu_v1/scripts/preflight.sh
bash experiment_packages/cephfs_eil_miu_v1/scripts/launch_cephfs_e2m1.sh
```

The host runtime must already exist at `/root/experiment_g_runtime/conda/env`
with Megatron-LM at `/root/experiment_g_runtime/Megatron-LM`.  The launcher
uses 4 GPUs as `2 train + 2 rollout`, trains `allenai/Olmo-3-7B-Instruct`,
sets EIL:MIU batch ratio to `2:1`, and uses learning rate `2e-6`.

All model assets and experiment outputs are under CephFS by default:

```text
/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct
/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct_torch_dist
/cephfs/shared/experiment_g/cephfs_eil_miu_v1/checkpoints
/cephfs/shared/experiment_g/cephfs_eil_miu_v1/evaluations
/cephfs/shared/experiment_g/cephfs_eil_miu_v1/logs
```

Completion signal:

```bash
bash experiment_packages/cephfs_eil_miu_v1/scripts/acceptance.sh
```

The acceptance line passes only after checkpoints and EIL/MIU summaries exist
for steps `19 39 59 79 99 119 139 159 179 199`.
