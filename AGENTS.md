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

## Machine / SSH
- 当前机器：`10.199.7.32`（简称 `32`）
- `data` 目录在以下机器间互通，可从本机直接 SSH：
  - `10.199.7.32`
  - `10.199.7.33`
  - `10.199.7.190`
  - `10.199.7.191`
- 常用变量（可直接复用）：
```bash
SERVER_IPS=("10.199.7.32" "10.199.7.33" "10.199.7.190" "10.199.7.191")
SSH_PORT=2289
SSH_USER="root"
```
- SSH 统一参数：
  - `SSH_PORT=2289`
  - `SSH_USER=root`

## Long-Run Jobs (tmux)
- 长时间训练/评测/提交任务默认放后台 `tmux` 跑，避免 SSH 断开导致任务中断。
- 推荐模板：
```bash
cd /data/liushiqi/AutoVLA
tmux new-session -d -s <session_name> '
cd /data/liushiqi/AutoVLA && \
<your_command> 2>&1 | tee <log_path>
'
```

## Canonical Commands (Latest Verified)

### SFT

#### Qwen2.5-VL-3B
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

```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_sft_qwen_local8gpu.sh
```

#### ReCogDrive-VLM-2B（default speedup profile, noprobe）
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
SFT_CONFIG=training/recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu-epoch10-speedupab-noprobe-20260323 \
bash scripts/run_sft_recogdrive_vlm2b_local8gpu.sh
```

```bash
# Optional: disable generated probe for stabler/faster throughput
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
SFT_CONFIG=training/recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu-epoch10-speedupab-noprobe-20260323 \
bash scripts/run_sft_recogdrive_vlm2b_local8gpu.sh
```

```bash
# Optional smoke: torch.compile validation profile
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
SFT_CONFIG=training/recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu-epoch10-speedupab-compile-smoke-20260323 \
bash scripts/run_sft_recogdrive_vlm2b_local8gpu.sh
```

#### SFT Answer Format
- 暂无独立 `answer-format` SFT 配置（当前 SFT 走 `mix-sft` 配置族）。

### RFT (GRPO)

#### Qwen2.5-VL-3B default-format
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-default-format \
bash scripts/run_rft.sh
```

#### Qwen2.5-VL-3B answer-format
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-answer-format \
bash scripts/run_rft.sh
```

#### ReCogDrive-VLM-2B default-format（latest, ip33）
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/recogdrive-vlm-2b-navsimv2-grpo-cot-fast-rft20260324-ip33-epoch9-default-format-actionbook-bsz16-fullnavtrain-epoch1-lr5e5 \
bash scripts/run_rft.sh
```

#### ReCogDrive-VLM-2B answer-format（latest, ip190; aligned with ip33 default ckpt）
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/recogdrive-vlm-2b-navsimv2-grpo-cot-fast-rft20260324-ip190-epoch9-answer-format-actionbook-bsz16-fullnavtrain-epoch1-lr5e5 \
bash scripts/run_rft.sh
```

### Navtest Standard EPDMS (8GPU)

#### Qwen2.5-VL-3B SFT（latest retry3 epoch4）
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_qwen_sft8_retry3_epoch4/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_qwen_sft8_retry3_epoch4.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navtest_epdms_standard_qwen_8gpu.sh
```

#### ReCogDrive-VLM-2B SFT（epoch4）
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_recogdrive_vlm2b_sft8_epoch4/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_sft8_epoch4.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navtest_epdms_standard_recogdrive_vlm2b_8gpu.sh
```

#### Qwen2.5-VL-3B RFT default-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_rft_default_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_rft_default_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navtest_epdms_standard_8gpu.sh
```

#### Qwen2.5-VL-3B RFT answer-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_rft_answer_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_rft_answer_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navtest_epdms_standard_8gpu.sh
```

#### ReCogDrive-VLM-2B RFT (8GPU)
##### default-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_recogdrive_vlm2b_rft20260323_default_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_rft20260323_default_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navtest_epdms_standard_8gpu.sh
```

##### answer-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_recogdrive_vlm2b_rft20260323_answer_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_rft20260323_answer_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navtest_epdms_standard_8gpu.sh
```

### Navhard Two-Stage (8GPU)

#### Qwen2.5-VL-3B SFT（latest retry3 epoch4）
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_qwen_sft8_retry3_epoch4/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_qwen_sft8_retry3_epoch4.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```

#### ReCogDrive-VLM-2B SFT（epoch4）
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_recogdrive_vlm2b_8gpu_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navhard_two_stage_autovla_recogdrive_vlm2b_8gpu.sh
```

#### Qwen2.5-VL-3B RFT default-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft_default_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_rft_default_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```

#### Qwen2.5-VL-3B RFT answer-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft_answer_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_rft_answer_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```

#### ReCogDrive-VLM-2B RFT (8GPU)
##### default-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_recogdrive_vlm2b_rft20260323_default_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_rft20260323_default_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```

##### answer-format
```bash
cd /data/liushiqi/AutoVLA
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_recogdrive_vlm2b_rft20260323_answer_format/plan_$(date -u +%F_%H-%M-%S) \
CONFIG_PATH=/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_rft20260323_answer_format.yaml \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
CKPT_PATH=<rft_ckpt_path> \
bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```

### Submission (`submission.pkl`)

#### Warmup: 生成 `submission.pkl`
```bash
cd /data/liushiqi/AutoVLA
CUDA_VISIBLE_DEVICES=7 \
/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_autovla_submission.py \
  --config config/submission/navsimv2_autovla_warmup.yaml \
  --set model.checkpoint_path=<ckpt_path> \
  --set submission.team_name='<team_name>' \
  --set submission.authors='<authors>' \
  --set submission.email='<email>' \
  --set submission.institution='<institution>' \
  --set submission.country='<country>'
```

#### Warmup: 本地验证
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_submission_validation.py \
  --config config/submission/navsimv2_autovla_warmup.yaml \
  --submission-file /path/to/submission.pkl \
  --set data.metric_cache_path=/data/dataset/navsim/metric_cache_v2/<warmup_metric_cache>
```

#### Private Challenge: 生成 `submission.pkl`
```bash
cd /data/liushiqi/AutoVLA
CUDA_VISIBLE_DEVICES=7 \
/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_autovla_submission.py \
  --config config/submission/navsimv2_autovla_private_test_hard.yaml \
  --set model.checkpoint_path=<ckpt_path> \
  --set submission.team_name='<team_name>' \
  --set submission.authors='<authors>' \
  --set submission.email='<email>' \
  --set submission.institution='<institution>' \
  --set submission.country='<country>'
```

#### Private Challenge: 本地结构校验
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_submission_validation.py \
  --config config/submission/navsimv2_autovla_private_test_hard.yaml \
  --submission-file /path/to/submission.pkl
```

#### Navhard Submission Config
- 有配置：`config/submission/navsimv2_autovla_navhard.yaml`
- 该 split 不是 NavSim v2 正式 challenge 提交 split，仅用于本地参考流程。

## Notes
- `navtest` 8GPU 正式 launcher：
  - `scripts/eval/run_navtest_epdms_standard_8gpu.sh`
- `run_navsimv2_epdms_standard.py` 默认每 `50` 个 token 输出一次进度日志。
- 历史 `navtest` 多卡拆分运行记录保留在：
  - `task/archive/process/navsimv2_standard_epdms_process.md`
- `submission.pkl` 运行手册与历史记录：
  - `task/navsimv2_submission_runbook_task.md`
- 已完成任务摘要统一放到：
  - `task/archive/completed/`
