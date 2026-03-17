# AutoVLA NavSim v2 Guide

## Scope
- 当前项目默认目标是 `NavSim v2`。
- 新的 SFT / RFT / 标准 EPDMS 评测，默认都按 `NavSim v2` 口径执行。
- `NavSim v1` 只用于隔离复现或对照排查，建议放到独立 worktree，不要混入当前主线。

## Environment
- 训练与评测默认环境：`autolsqv2`
- Python:
  - `/data/miniconda/envs/autolsqv2/bin/python`
- 常用数据根目录：
  - `/data/dataset/navsim`
- 标准 v2 EPDMS 评测必须走 upstream:
  - `/data/liushiqi/navsim`
- 不要把仓库内 vendored `navsim/` 的定制 `pdm_score` 当作标准 v2 口径。

## Canonical Commands

### SFT
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
NCCL_NET_MODE=fast_ib \
NCCL_IB_HCA=mlx5_bond_0 \
RUN_MASTER_BG=1 \
AUTO_CHECK=1 \
bash scripts/run_sft_3nodes_ssh.sh
```

### RFT
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4 \
bash scripts/run_rft.sh
```

### Navtest Standard EPDMS
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py \
  --config config/eval/navsimv2_epdms_standard_autovla_rft20260312_step6000.yaml \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_rft20260312_step6000/navtest_full_$(date -u +%F_%H-%M-%S)
```

### Navhard Two-Stage 8GPU
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft20260312_step6000/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh
```

## Notes
- `navtest` 当前仓库内只有标准 wrapper 单入口命令，尚未 repo 化为正式 8GPU shard launcher。
- 历史 `navtest` 多卡拆分运行记录保留在：
  - `task/archive/process/navsimv2_standard_epdms_process.md`
- 已完成任务摘要统一放到：
  - `task/archive/completed/`
