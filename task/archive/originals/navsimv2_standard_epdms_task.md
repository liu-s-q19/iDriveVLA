# NavSim v2 标准 EPDMS 结果摘要

## 目标
- 在当前仓库提供 `NavSim v2` 标准 `EPDMS` 本地评测入口。
- 严格区分两条链路：
  - 标准口径：upstream `navsim` + 标准 `pdm_score`
  - 定制口径：仓库内原有 `AutoVLA/navsim` 定制评测实现

## 当前结论
- 标准 v2 EPDMS 本地 wrapper 已落地并可用：
  - `tools/eval/run_navsimv2_epdms_standard.py`
- `AutoVLA` 已可绕开旧定制 scorer，直接使用 upstream 标准 `pdm_score` 完成 full `navtest`。
- 旧的 `run_pdm_score_cot.py + 定制 scorer` 不应再作为 `navsim v2` 标准评测入口。

## 过程文档
- 详细实施过程、排障记录、历史命令与中间验证：
  - `/data/liushiqi/AutoVLA/task/process/navsimv2_standard_epdms_process.md`

## 最终有效命令

### 1. 标准口径 dry-run
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --dry-run
```

### 2. 标准 baseline full navtest
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_full_2026-03-14_13-22-03
```

### 3. AutoVLA 标准口径 full navtest
```bash
ssh -p 2289 root@10.199.7.190 "
cd /data/liushiqi/AutoVLA &&
tmux new-session -d -s navtest_epdms_cons_2026_03_15_04_48_25 '
CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/miniconda/envs/autolsqv2/bin/python \
/data/liushiqi/AutoVLA/tools/eval/run_navsimv2_epdms_standard.py \
  --config /data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190 \
  2>&1 | tee /data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/launcher_stdout.log
'"
```

## 最终结果

### 标准 baseline full navtest
- 输出：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_full_2026-03-14_13-22-03/2026.03.14.14.54.12.csv`
- 结果：
  - `score_mean = 0.3233531830113996`

### AutoVLA 标准口径 full navtest
- 配置：
  - `config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml`
- checkpoint：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 输出目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190`
- summary：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/summary.json`
- 结果：
  - `successful = 12146`
  - `failed = 0`
  - `invalid_sum = 0`
  - `score_mean = 0.5944498471141966`

### 对比结论
- 当前 `SFT` 最终 ckpt 在标准口径 full `navtest` 上，明显优于当前 baseline：
  - baseline：`0.3233531830113996`
  - AutoVLA：`0.5944498471141966`

## Navhard 状态
- 本任务不再单独补“标准 wrapper 直接跑 navhard full”的记录。
- 当前 `navhard` 主评测链路已在独立任务中验证完成：
  - `/data/liushiqi/AutoVLA/task/navhard_two_stage_8gpu_eval_task.md`
- 已确认：
  - fresh 单卡与 8 卡结果完全一致：`score_diff = 0.0`
  - `final_extended_pdm_score = 0.14937031924317998`

## 最终结论
- 这份任务的核心目标已经完成：
  - 标准 v2 EPDMS 本地入口已落地；
  - AutoVLA 标准口径 full `navtest` 已完成；
  - 标准口径与定制口径已明确隔离。
- 当前无需再把“标准 wrapper 直接跑 navhard full”作为阻塞项。
