# AutoVLA 在 Navsim v1.1 基线复现任务单

状态：进行中

## README 对照
| README 流程 | 本任务阶段 | 当前状态 |
|---|---|---|
| 1. Dataset Preprocessing | 阶段 1 | 已完成（已验收） |
| 2. Action Codebook Creation | 阶段 2 | 进行中（已有码本复用 + 对比重聚类后台运行） |
| 3. Supervised Fine-tuning (SFT) | 阶段 3 | 未开始 |
| 4. Reinforcement Fine-tuning (RFT) | 阶段 4 | 未开始 |
| 5. Evaluation | 阶段 5 | 未开始 |

---

## 阶段闸门规则（必须）
1. 每阶段只做该阶段工作，不跨阶段实现。
2. 每阶段结束时固定输出：
   - 执行命令
   - 关键产物路径
   - 验收结果（通过/失败 + 原因）
   - 下一阶段计划
3. 你回复“通过阶段N”后，再进入下一阶段。

---

## 阶段 0：环境与配置冻结（准备）
### 目标
- 固化可复现实验入口（环境变量、配置路径、输出路径）。

### 最小执行
```bash
conda activate autovla_codeclean
python -V
```

### 验收标准
- 环境可用；
- `scripts/run_nuplan_preprocessing.sh` 指向 `dataset/qwen2.5-vl-72B-trainval`；
- 输出目录为 `/data/dataset/navsim/preprocessed/navtrainval_nocot`。

---

## 阶段 1：数据预处理（README 第1步）
### 目标
- 完成 `navtrainval_nocot` 全量 JSON 生成。

### 最小执行
```bash
bash scripts/run_nuplan_preprocessing.sh
find /data/dataset/navsim/preprocessed/navtrainval_nocot -maxdepth 1 -type f | wc -l
```

### 验收标准
- 文件数达到预期规模（目标 `103288`）；
- 结束日志无未处理异常。

### 本次验收记录（2026-03-06）
- 命令：
  - `find /data/dataset/navsim/preprocessed/navtrainval_nocot -maxdepth 1 -type f -name '*.json' | wc -l`
- 结果：
  - 文件数 `103288`
  - 任务结束日志：`All preprocessing data without CoT results have been saved...`
  - 抽样 JSON 可读，关键字段存在（`token`、`gt_trajectory`、相机路径等）
- 结论：阶段 1 验收通过。

---

## 阶段 2：动作码本（README 第2步）
### 目标
- 生成动作码本并得到量化误差基线。

### 最小执行
```bash
python tools/action_token/action_token_cluster.py \
  --data_path /data/dataset/navsim/preprocessed/navtrainval_nocot \
  --output codebook_cache/agent_vocab.pkl \
  --num_cluster 2048

python tools/action_token/codebook_error_stats.py \
  --data_path /data/dataset/navsim/preprocessed/navtrainval_nocot \
  --codebook codebook_cache/agent_vocab.pkl
```

### 验收标准
- 码本文件生成成功；
- 输出 ADE/FDE/heading 指标并归档。

### 当前状态
- 阶段 2 仍为进行中（重聚类对比任务未收口）。
- 当前评估采用码本：`/data/liushiqi/AutoVLA/codebook_cache/agent_vocab.pkl`。

### 阶段 2 补充结果（已完成，navtest）
- 固定评估口径：
  - split：`navtest`
  - cache：`/data/liushiqi/recogdrive/exp/metric_cache`
  - agent：`human_agent`

- A. GT 不离散化（reference）：
  - 结果：`/data/liushiqi/AutoVLA/logs/eval/navtest_human_pdms_noquant_2026-03-07_05-19-28_csv/2026.03.07.06.08.46.csv`
  - `score(PDMS)=0.9455140385320785`

- B. GT 先离散化（oracle quantized）：
  - 结果：`/data/liushiqi/AutoVLA/logs/eval/navtest_human_quantized_pdms_2026-03-07_04-08-15_csv/2026.03.07.05.03.17.csv`
  - `score(PDMS)=0.9234589029451706`
  - `quant_ade_m=0.04920196093325713`
  - `quant_fde_m=0.051883486696311085`
  - `quant_heading_mae_deg=0.8449313004191331`
  - `quant_bbox_l2=0.05903596693256331`
  - `quant_max_center_err=0.5101190209388733`

- A/B 对比（离散化上界）：
  - `S_ref=0.9455140385320785`
  - `S_q=0.9234589029451706`
  - `z_retain=S_q/S_ref=0.9766735885300049`
  - `z_abs_penalty=S_ref-S_q=0.0220551355869079`
  - `z_rel_penalty=1-z_retain=0.0233264114699951`（约 `2.33%`）

### 说明（口径）
- `human_agent` 用 GT 轨迹，但 PDM 是闭环聚合评分，不是 GT 拟合分，因此 GT 参考分不必然为 `1.0`。
- 当前口径下 GT 参考上限可记为 `S_ref=0.9455`。

### 当前接口（已落地）
- 已在 `navsim/navsim/planning/script/run_pdm_score.py` 增加 `quantization.*` 开关。
- 配置项在 `navsim/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`：
  - `trajectory_source`（`agent | metric_cache`，其中 `metric_cache` 直接评估 PDM-Closed/IDM 缓存轨迹）
  - `quantization.enable`
  - `quantization.codebook_path`
  - `quantization.device`
  - `quantization.agent_width`
  - `quantization.agent_length`

### 规则强基线对比结果（已完成，2026-03-07）
- `IDM no-quant`：
  - 结果：`/data/liushiqi/AutoVLA/logs/eval/navtest_idm_pdms_noquant_2026-03-07_11-30-36_csv/2026.03.07.12.23.03.csv`
  - `S_idm_ref=0.9043777045929905`
- `IDM quantized`：
  - 结果：`/data/liushiqi/AutoVLA/logs/eval/navtest_idm_quantized_pdms_2026-03-07_11-31-22_csv/2026.03.07.12.43.13.csv`
  - `S_idm_q=0.8960840071727242`
  - `quant_ade_m=0.04411949268367302`
  - `quant_fde_m=0.03342625841675461`
  - `quant_heading_mae_deg=0.6175785671598831`
  - `quant_bbox_l2=0.051166602458949946`
  - `quant_max_center_err=5.317928314208984`
- 对比结论：
  - `z_idm_retain=S_idm_q/S_idm_ref=0.9908293875687715`
  - `delta_idm_abs=S_idm_ref-S_idm_q=0.0082936974202664`
  - `z_idm_rel_penalty=1-z_idm_retain=0.0091706124312285`（约 `0.92%`）
- smoke 验证（20 场景）：
  - `trajectory_source=metric_cache` 已跑通
  - 结果：`/tmp/navsim_smoke_exp/idm_ref_smoke/2026.03.07.11.22.34.csv`

### 对比任务卡片：IDM 强基线 vs 该模型离散化
- 任务名：`IDM/PDM-Closed 上界对比`
- 任务目标：
  - 计算基于规则/IDM 强基线（`PDM-Closed` 内部轨迹）的原始 PDMS 得分；
  - 计算同一 IDM 模型轨迹经 codebook 离散化后的 PDMS 得分。
- 输入口径（与当前实验一致）：
  - split：`navtest`
  - metric cache：`/data/liushiqi/recogdrive/exp/metric_cache`
  - codebook：`/data/liushiqi/AutoVLA/codebook_cache/agent_vocab.pkl`
- 输出指标：
  - `S_idm_ref`（IDM no-quant）
  - `S_idm_q`（IDM quantized）
  - `z_idm_retain = S_idm_q / S_idm_ref`
  - `delta_idm_abs = S_idm_ref - S_idm_q`
- 当前状态：`Completed (full navtest jobs finished on 2026-03-07)`


---

## 阶段 3：SFT（README 第3步）
### 目标
- 启动并跑通一次可用的 SFT 训练。

### 最小执行
```bash
python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft
```

### 验收标准
- 生成 checkpoint；
- 训练日志可见正常收敛趋势（至少前段 loss 可用）。

### 当前状态
- 未开始。

---

## 阶段 4：RFT/GRPO（README 第4步）
### 目标
- 基于最佳 SFT checkpoint 跑通 GRPO 训练。

### 最小执行
```bash
bash scripts/run_rft.sh
```

### 验收标准
- GRPO 任务可持续推进；
- 产出 RFT checkpoint 与训练日志。

### 当前状态
- 未开始。

---

## 阶段 5：评估（README 第5步）
### 目标
- 完成 Navsim PDMS 评估（可选补充 nuScenes eval）。

### 最小执行
```bash
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
```

可选：
```bash
python tools/eval/nusc_eval.py \
  --config config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --seg_data_path /path/to/nusc_eval_seg
```

### 验收标准
- 输出完整评估指标表；
- 归档配置、命令、结果与问题复盘。

### 当前状态
- 未开始（README 主线口径）。

---
