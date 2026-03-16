## AutoVLA Navsim 进展总结 / AutoVLA Navsim Progress

### 目标 / Objective
完成 Navsim navtrainval 数据预处理并评估 Action Token 设计（含误差上限与改进方向），为 SFT/GRPO 训练提供可用子集和诊断文档。
Finish Navsim navtrainval preprocessing and assess the Action Token design (error ceiling + improvement ideas) to support SFT/GRPO training.

---

### 已完成 / Completed
1. **可用子集与诊断**
   - 构建 74‑log 过滤器 `config/dataset/scene_filter/navtrain_local_available.yaml` 与 `config/dataset/qwen2.5-vl-72B-trainval-available.yaml`。
   - 运行 `python tools/preprocessing/nocot_sample_generation.py --config dataset/qwen2.5-vl-72B-trainval-available --output_dir /data/dataset/navsim/preprocessed/navtrainval_available_nocot`，产出 86 条 No-CoT JSON。
   - 记录缺帧情况：`logs/navtrainval_missing_summary.json`, `logs/navtrainval_missing_detail.json`, `logs/navtrainval_complete_logs.txt`。

2. **Action Token 代码注释**
   - `models/action_tokenizer.py:41-125` 详解码本含义、未知 token 回退与 rollout 过程。
   - `models/autovla.py:234-356` 说明固定 horizon 填补逻辑及 action-only 交叉熵的作用。

3. **文档扩展**
   - `autovla_sum.md:184-197` 新增 Action Token 诊断：数据覆盖、量化误差（ADE≈25 m / FDE≈44 m / heading≈2.27°，基于 86 场景）及改进建议（层级/自适应码本、连续校正头等）。

4. **navtest No-CoT 预处理完毕**
   - 将 `scripts/run_nuplan_preprocessing.sh` 的 `CONFIG` 指向 `dataset/qwen2.5-vl-72B-nuplan`，`OUTPUT_DIR` 设为 `/data/dataset/navsim/preprocessed/navtest_nocot`，并在 `autovla_codeclean` 环境中运行 `bash scripts/run_nuplan_preprocessing.sh`。
   - 运行前以 `scripts/check_navtrain_available.py --config qwen2.5-vl-72B-nuplan` 生成 `logs/navtest_missing_report.txt`，确认 10 个 navtest 日志无缺帧。
   - 预处理共写出 109 条 JSON，存放于 `/data/dataset/navsim/preprocessed/navtest_nocot`，可直接用于后续 SFT/Eval。

---

### 关键文件 / Key Files
- `config/dataset/scene_filter/navtrain_local_available.yaml`
- `config/dataset/qwen2.5-vl-72B-trainval-available.yaml`
- `models/action_tokenizer.py:41-125`
- `models/autovla.py:234-356`
- `autovla_sum.md:184-197`
- `logs/navtrainval_missing_summary.json`
- `logs/navtrainval_missing_detail.json`
- `logs/navtrainval_complete_logs.txt`

---

### 当前问题 / Blockers
- `sensor_blobs/trainval` 多数缺帧，`scripts/run_nuplan_preprocessing.sh` 仍会抛 `FileNotFoundError`（CAM_F0 等）。
- 现有码本 token 最大位移约 18 m，无力覆盖 50 m 以上轨迹；当前 ADE/FDE 结果只能作为“码本覆盖不足”的提示。

---

### 下一步 / Next Steps
1. 按 `logs/navtrainval_missing_report.txt` 同步缺失的 `sensor_blobs` 目录（trainval）。
2. 数据补齐后执行：
   ```bash
   bash scripts/run_nuplan_preprocessing.sh
   ```
   （可先用 `CONFIG=dataset/qwen2.5-vl-72B-trainval-available` 验证，再切回全量配置。）

3. 完整 JSON 生成后，重新评估码本：
   ```bash
   conda activate autovla_codeclean
   export NAVSIM_DEVKIT_ROOT=/data/liushiqi/AutoVLA/navsim
   export NAVSIM_DATA_ROOT=/data/dataset/navsim
   export NUPLAN_DATA_ROOT=/data/dataset/navsim
   export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
   export NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs
   export OPENSCENE_DATA_ROOT=/data/dataset/navsim
   export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
   export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:/data/liushiqi/AutoVLA:$PYTHONPATH

   python tools/action_token/codebook_error_stats.py \
   --data_path /data/dataset/navsim/preprocessed/navtrainval_nocot \
     --codebook codebook_cache/agent_vocab.pkl
   ```
   若 ADE/FDE 仍偏高，参考 `autovla_sum.md` 方案重训码本（层级/自适应/残差）或加连续校正头。

---

### 环境 / Environment
- Conda：`conda activate autovla_codeclean`
- 关键环境变量：
  ```bash
  export NAVSIM_DEVKIT_ROOT=/data/liushiqi/AutoVLA/navsim
  export NAVSIM_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
  export NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs
  export OPENSCENE_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
  export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:/data/liushiqi/AutoVLA:$PYTHONPATH
  ```
