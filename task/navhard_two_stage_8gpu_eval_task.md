# Navhard Two-Stage 8-GPU Eval Task

状态：已完成

## 目标
- 为 `navhard_two_stage` 建立 8-GPU 分片评测链路。
- 保证分片合并后的最终指标与当前单卡实现语义一致。

## 最终结论
- 8-GPU 分片评测链路已落地并验证通过。
- 关键修复包括：
  - stage2 token 顺序稳定化
  - 按 token 固定采样 seed，消除单卡与分片间的随机漂移
- 修复后的 fresh 单卡和 8 卡结果一致，说明当前并行评测口径可靠。

## 核心命令
- 8 卡启动：
  - `bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`
- 单卡基线：
  - `CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla.py --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml --set model.checkpoint_path=/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt --set eval.output_dir=<out_dir>`

## 最终结果
- fresh 单卡 full navhard：
  - `num_successful_scenarios = 5912`
  - `num_failed_scenarios = 0`
  - `final_extended_pdm_score = 0.14937031924317998`
  - `elapsed_sec ≈ 16216.21`（约 `4.50h`）
- 8 卡 full navhard：
  - `num_successful_scenarios = 5912`
  - `num_failed_scenarios = 0`
  - `final_extended_pdm_score = 0.14937031924317998`
  - 最慢 shard wall time 约 `2168s`（约 `36.1min`）
- 结论：
  - `score_diff = 0.0`
  - 相对 fresh 单卡约 `7.5x` 加速

## 原版归档
- 详细设计、排障、历史命令与完整对比：
  - `/data/liushiqi/AutoVLA/task/archive/originals/navhard_two_stage_8gpu_eval_task.md`
