# NavSim v2 标准 EPDMS 结果摘要

状态：已完成

## 目标
- 在当前仓库提供 `NavSim v2` 标准 `EPDMS` 本地评测入口。
- 区分标准口径与 AutoVLA 定制口径，避免混用。

## 最终结论
- 标准 wrapper 已落地：`tools/eval/run_navsimv2_epdms_standard.py`
- AutoVLA 已可通过 upstream `navsim` 标准 `pdm_score` 完成 full `navtest`。
- 旧定制 scorer 不应再作为标准 v2 EPDMS 入口。

## 有效命令
- dry-run：
```bash
cd /data/liushiqi/AutoVLA
/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --dry-run
```
- AutoVLA 标准口径 full navtest：
```bash
/data/miniconda/envs/autolsqv2/bin/python /data/liushiqi/AutoVLA/tools/eval/run_navsimv2_epdms_standard.py \
  --config /data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190
```

## 结果
- `NavSim v2` 标准 `constant_velocity` baseline full `navtest`：
  - `score_mean = 0.3233531830113996`
- AutoVLA 标准口径 full navtest：
  - `successful = 12146`
  - `failed = 0`
  - `invalid_sum = 0`
  - `score_mean = 0.5944498471141966`

## 原版归档
- 详细过程文档：
  - `/data/liushiqi/AutoVLA/task/archive/originals/navsimv2_standard_epdms_task.md`
- 历史过程材料：
  - `/data/liushiqi/AutoVLA/task/archive/process/navsimv2_standard_epdms_process.md`
