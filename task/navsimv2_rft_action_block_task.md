# NavSim v2 RFT 末尾 Action Block 方案 A

状态：执行中（answer-format 协议健康度阻塞）

## 当前阻塞与下一步
- 结论（2026-03-23）：
  - ReCogDrive answer-format 在当前 SFT 初始化下，RFT smoke 仍出现 `answer_block_valid=0`、`reward_input_valid=0`。
  - 现阶段判断为“模型在严格规则下输出未对齐”，优先策略是先做 SFT 重新对齐，再评估 RFT 规则与奖励塑形。
- TODO：
  - [ ] 重新训练 ReCogDrive-VLM-2B SFT（保持 `<answer> ... trajectory ... </answer>` 真值格式完全一致）。
  - [ ] 用新 SFT ckpt 复跑 answer-format RFT smoke（20 steps / 200 steps）。
  - [ ] 验收指标：`answer_block_valid`、`reward_input_valid` 是否显著抬升且稳定。
  - [ ] 若仍不达标，再执行 RFT reward 分层塑形（非 0/1 截断）方案。

## 背景
- 当前 RFT 生成后，会从整段 `completion` 中扫描所有 `token >= action_start_id` 的 token 作为动作序列，再做截断/补齐后 decode 为轨迹并计算 reward。
- 该实现已能跑通，但存在两个结构性问题：
  - 语言部分若混入 action token，会污染动作提取。
  - 当动作 token 数不足或过多时，后处理的补零/截断会改变最终 reward 对应的真实输出语义。
- 当前代码更像“全 completion 扫描 + 后处理兜底”，不适合作为“动作必须在最后出现，且 reward 只反映真实动作输出”的长期口径。

## 目标
- 采用方案 A：
  - 允许 completion 前面保留语言分析/说明。
  - 要求最终动作以“末尾 action block”形式输出。
  - reward 只基于最后一个合法 action block 解析出的动作序列计算。
- 明确区分：
  - 合法动作输出
  - 非法格式输出
  - 后处理补齐/截断逻辑
- 最终使 RFT 的训练信号更接近“模型真实最终动作输出质量”，而不是“后处理伪造后的轨迹质量”。

## 当前主线约束
- 当前正式主线只保留：
  - `Qwen2.5-VL-3B`
  - `ReCogDrive-VLM-2B`
- 当前标准动作口径统一为：
  - `8 actions`
  - `4 seconds`
- `ReCogDrive-VLM-8B` 不进入当前主线 launcher / config 矩阵，只保留为非主线历史配置。

## 当前 RFT Baseline
- 当前 baseline run：
  - `grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06`
- baseline ckpt：
  - `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt`
- baseline 配置：
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4.yaml`
- baseline 关键观测：
  - `sample_action_tokens_len` 已经能持续非零，说明模型已能生成动作 token。
  - `group_reward_std` 大多数 step 非零，但仍会偶发回到 `0.0`。
  - 当前 advantage 在 `group_std≈0` 时使用 `fallback_mode=reward`，说明训练链路仍依赖“组内塌缩后的兜底更新”。
- baseline 实现局限：
  - action 提取口径：整段 `completion` 全局扫描 action token。
  - reward 输入口径：动作不足补零、动作过长截断。
  - 因此 reward 可能不再严格对应“模型最终显式动作答案”。

## 2026-03-19 补充排查：Navhard 分数完全相同与 LoRA 加载嫌疑
- 本轮重新核对后，确认存在“不同 RFT checkpoint 的 `navhard_two_stage` 分数完全相同”的现象。
- 已确认相同分数的三组结果：
  - baseline `rft-step6000-reward6.2188.ckpt`
    - 结果：`0.29853212542780166`
    - 文件：
      - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft20260312_step6000/plan_2026-03-17_02-37-14_190/merged/summary.json`
  - answer-protocol `rft-step6000-reward6.2500.ckpt`
    - 结果：`0.29853212542780166`
    - 文件：
      - `/tmp/navsimv2-answer-block-logs-2026-03-19/eval/navhard_two_stage_autovla_answer_protocol_step6000/plan_2026-03-18_02-07-09_190/merged/summary.json`
  - answer-protocol `rft-step12000-reward6.6875.ckpt`
    - 结果：`0.29853212542780166`
    - 文件：
      - `/tmp/navsimv2-answer-block-logs-2026-03-19/eval/navhard_two_stage_autovla_answer_protocol_step12000/plan_debug_local/merged/summary.json`

### 当前最可疑根因
- 当前最可疑原因不是“训练真的学到完全一样”，而是 answer-protocol 这两次 `navhard` 评测没有真正加载到 LoRA 增量权重。
- 直接证据：
  - baseline `navhard` 的 `resolved_config.yaml`：
    - `checkpoint_path=/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt`
    - `model.config_path=/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
    - `dataset_name=nuplan`
    - `trajectory_sampling.num_poses=10`
    - `lora_conf.use_lora=null`
  - answer-protocol `step6000` 的 `resolved_config.yaml`：
    - `checkpoint_path=/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt`
    - 其余关键项仍是：
      - `model.config_path=/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
      - `dataset_name=nuplan`
      - `trajectory_sampling.num_poses=10`
      - `lora_conf.use_lora=null`
  - answer-protocol `step12000` 的 `resolved_config.yaml` 也同样是：
    - `lora_conf.use_lora=null`
- 由于 answer-protocol 训练 run 本身是 LoRA 路径，若评测侧仍用 `use_lora=null`，则高度怀疑：
  - checkpoint 里的 LoRA 增量未被正确挂载，
  - 实际推理行为回退到基础主干或与 baseline 等价的加载路径，
  - 最终导致多个 checkpoint 在 `navhard` 上出现逐位相同的结果。

### 为什么“之前 use_lora=true 跑出来更低分”不能直接当反证
- 之前已经有一组 `use_lora=true` 的 `navhard` 结果明显更低：
  - `0.16527880078519738`
  - 文件：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft20260312_step6000_lora_true_ip191/run_2026-03-19_05-53-20/summary.json`
- 但那组实验不是单变量对比，不能直接用来反驳“answer-protocol eval 没加载 LoRA”的怀疑。
- 已核对差异，至少同时混入了：
  - `model.config_path` 不同
    - 低分 LoRA 组：
      - `/data/liushiqi/AutoVLA/config/eval/qwen2.5-vl-3B-navsimv2-autovla-epdms-conservative-model.yaml`
    - baseline / answer-protocol 组：
      - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
  - `dataset_name` 不同
    - 低分 LoRA 组：`navsim`
    - baseline / answer-protocol 组：`nuplan`
  - `trajectory_sampling.num_poses` 不同
    - 低分 LoRA 组：`8`
    - baseline / answer-protocol 组：`10`
  - `lora_conf.use_lora` 不同
    - 低分 LoRA 组：`true`
    - baseline / answer-protocol 组：`null`
- 因此，当前证据只能支持：
  - “在 conservative-model + navsim + 8 poses + use_lora=true 的组合下，这个 ckpt 得分约 `0.1653`。”
  - 不能支持：
    - “只要打开 LoRA wrapper，分数一定下降到 `0.1653`。”

### 当前更合理的解释
- 现在最合理的解释分成两层：
  - 解释 A：
    - answer-protocol `step6000/step12000` 的 `navhard` 评测很可能没有真正用上对应 checkpoint 的 LoRA 增量，因此与 baseline 得到完全相同的结果。
  - 解释 B：
    - 之前那组低分 `use_lora=true` 说明“切换到另一套 eval 配置后，分数会明显变化”，但它混入了 `8 poses / navsim / conservative-model` 等因素，不能把变化单独归因给 LoRA wrapper。

### 最小验证方案
1. 严格单变量验证 LoRA 是否真正生效
   - 固定同一 checkpoint：
     - 先用 answer-protocol `rft-step6000-reward6.2500.ckpt`
   - 固定同一评测配置：
     - `config_path` 不变
     - `dataset_name=nuplan`
     - `trajectory_sampling.num_poses=10`
   - 只切换：
     - `model.lora_conf.use_lora=false/null`
     - 对比 `model.lora_conf.use_lora=true`
   - 若结果立刻分叉，则说明原先“完全相同”高度可能是 LoRA 没挂上。
2. 做小样本逐 token 哈希比对
   - 在固定 16 或 32 个 `navhard` token 上，
   - 导出每个 token 的：
     - 原始 action token 序列
     - 前 8 pose 轨迹数组哈希
   - 比较：
     - baseline `step6000`
     - answer-protocol `step6000`
     - answer-protocol `step12000`
   - 若逐 token 轨迹也完全一致，则“未加载新权重”概率会进一步上升。
3. 再做真正的 LoRA true/false apples-to-apples 对比
   - 保持：
     - 同一 checkpoint
     - 同一 `config_path`
     - 同一 `dataset_name`
     - 同一 `trajectory_sampling.num_poses`
   - 仅切：
     - `use_lora=true/false`
   - 这一步才可用于回答“LoRA wrapper 本身是否导致性能变化”。

### 2026-03-19 当日验证进展
- 已完成一轮 16-sample 真实预测哈希对照：
  - 输出：
    - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_answer_protocol_step6000_lora_compare_16samples.json`
    - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_answer_protocol_step6000_lora_compare_16samples_summary.json`
- 对照对象：
  - `baseline_step6000_null`
    - ckpt：
      - `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt`
    - `use_lora=null`
  - `answer_step6000_null`
    - ckpt：
      - `/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt`
    - `use_lora=null`
  - `answer_step6000_true`
    - 同上 ckpt
    - `use_lora=true`
  - `answer_step6000_false`
    - 同上 ckpt
    - `use_lora=false`

### 新增验证结论 1：`use_lora=null` 不等于“没加载 LoRA”
- 当前 `tools/eval/run_navhard_two_stage_autovla.py` 的实际行为已被运行日志再次验证：
  - 当 checkpoint 是 LoRA-form 且 `use_lora=null` 时，会打印：
    - `Detected LoRA-form checkpoint while model.lora_conf.use_lora is unset; enabling LoRA wrapper automatically.`
- 这次 16-sample 对照结果显示：
  - `answer_step6000_null` vs `answer_step6000_true`
    - `same_hash_count=16/16`
  - `answer_step6000_null` vs `answer_step6000_false`
    - `same_hash_count=0/16`
  - `answer_step6000_true` vs `answer_step6000_false`
    - `same_hash_count=0/16`
- 结论：
  - 对 answer-protocol `step6000` 这个 LoRA checkpoint 来说，
  - `use_lora=null` 与 `use_lora=true` 的真实预测完全一致，
  - 说明当前评测代码路径下 `null` 已自动启用 LoRA wrapper，
  - 因此“answer-protocol 全量 navhard 分数和 baseline 一样，是因为 `use_lora=null` 没加载 LoRA”这一条简单解释，当前已被样本级验证否定。

### 新增验证结论 2：baseline 与 answer-protocol 在实时预测上已经明显分叉
- 同一批 16 个 `navhard` token 上：
  - `baseline_step6000_null` vs `answer_step6000_null`
    - `same_hash_count=1/16`
  - `baseline_step6000_null` vs `answer_step6000_true`
    - `same_hash_count=1/16`
- 结论：
  - baseline `step6000` 和 answer-protocol `step6000` 在当前代码路径下做实时预测时，已经不是“完全一样的模型行为”。
  - 这说明 answer-protocol checkpoint 确实在推理输出上与 baseline 分叉了。

### 新增验证结论 3：但历史 full navhard 产物仍完全一致
- 已比较两份历史 full `navhard` merged csv：
  - baseline：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_rft20260312_step6000/plan_2026-03-17_02-37-14_190/merged/2026.03.17.03.38.19.csv`
  - answer-protocol step6000：
    - `/tmp/navsimv2-answer-block-logs-2026-03-19/eval/navhard_two_stage_autovla_answer_protocol_step6000/plan_2026-03-18_02-07-09_190/merged/2026.03.18.03.07.47.csv`
- 结果：
  - 两个 merged csv 的 SHA1 完全一致：
    - `572e3b7a09d4487003229fb667785f2c77ab7688`
  - 逐列比对结果：
    - `base_rows=5915`
    - `ans_rows=5915`
    - `same_tokens=true`
    - `num_diff_cols=0`
- 对比输出：
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_baseline_vs_answerprotocol_csv_compare.json`
- 16 个抽样 token 在这两份历史 merged csv 中对应的评分行也完全一致：
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_baseline_vs_answerprotocol_csv_16tokens.json`

### 新增验证结论 4（已修正）：先前 16-token “当前评分完全一致”结论来自比较脚本口径错误
- 先前产物：
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_baseline_vs_answerprotocol_currentscore_16tokens.json`
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_baseline_vs_answerprotocol_currentscore_16tokens_summary.json`
- 当时得到：
  - `same_metric_rows=16/16`
- 但后续直接调用当前评测代码中的 `_evaluate_token(...)`，对 3 个代表性 token 抽取原始 `score_row` 后，发现：
  - 轨迹不同
  - `ego_simulated_states` 不同
  - 原始评分行也不同
- 新产物：
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_intermediate_probe_3tokens.json`
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_intermediate_probe_3tokens_summary.json`
- 3 个 token 的直接探针结果：
  - `00d194f38b3f05c1e`
    - `score_row_equal=false`
    - `no_at_fault_collisions: 1.0 -> 0.0`
    - `time_to_collision_within_bound: 1.0 -> 0.0`
    - `pdm_score: 0.42857 -> 0.0`
  - `01ac04b9f18ecdbc6`
    - `score_row_equal=false`
    - `driving_direction_compliance: 0.5 -> 1.0`
  - `a44db880afe95be7`
    - `score_row_equal=false`
    - `ego_progress: 0.74537 -> 0.60457`
- 修正结论：
  - “实时轨迹 hash 已分叉，但当前 PDM 评分仍完全不分叉”这一条，当前已被直接探针否定。
  - 更合理的解释是：
    - 先前 `currentscore_16tokens` 对比脚本比较的是裁剪/占位后的摘要层，而不是 `_evaluate_token` 返回的原始 scorer 行，
    - 因而错误地得出了“评分完全一致”的结论。

### 当前最合理的解释（更新后）
- 现在最合理的解释已从“LoRA 没加载”更新为以下三层：
  - 解释 A：
    - 当前代码路径下，answer-protocol checkpoint 的 LoRA 实际是会被自动加载的；
    - 样本级预测已经证明 baseline 与 answer-protocol 会分叉。
  - 解释 B：
    - 先前“实时评分完全一致”的证据来自比较脚本口径错误，不能再作为有效依据。
    - 直接探针显示：在至少 3 个代表性 token 上，原始评分行已经明确分叉。
  - 解释 C：
    - 历史保存下来的 full `navhard` merged csv 与 baseline 字节级完全相同，仍然是异常强信号；
    - 在“实时原始评分已分叉”的新证据下，这个异常更偏向：
      - 历史产物复用 / 覆盖，
      - 历史运行代码路径与当前不同，
      - 或历史 merge/聚合阶段存在结果串用问题。

### 2026-03-19 补充归因：当前更像“历史评测链路问题”，不是 PDM 公式本身有问题
- 新增排查结果：
  - 历史 answer-protocol full eval 的 `partial_rows.csv` 中，至少这 3 个代表性 token 已经与 baseline 完全一致：
    - `a44db880afe95be7`
    - `00d194f38b3f05c1e`
    - `01ac04b9f18ecdbc6`
  - 这说明问题不只是在 merge 末尾，而是更早就出现在历史 partial 生成阶段。
- 同时，当前直接调用 `_evaluate_token(...)` 的探针却显示：
  - 同一组 token 的原始 `score_row` 已经分叉。
- 因而，当前最合理的归因变成：
  - 不是 PDM scorer 本身“算不出差异”；
  - 而是历史 full `navhard` 评测那次运行，极有可能使用了旧评测器代码路径，导致 answer-protocol checkpoint 在实际 partial 生成时没有按当前口径生效。
- 支持这一点的证据：
  - 历史 shard 启动日志中的 `torch.load(...)` 行号是：
    - `tools/eval/run_navhard_two_stage_autovla.py:194`
  - 当前代码中的对应位置已变为：
    - `tools/eval/run_navhard_two_stage_autovla.py:216`
  - 且历史 shard 启动日志中没有出现当前代码会打印的告警：
    - `Detected LoRA-form checkpoint while model.lora_conf.use_lora is unset; enabling LoRA wrapper automatically.`
- 因此：
  - 历史 full eval 很可能是在“自动 LoRA 挂载逻辑尚未加入”或“旧脚本副本”条件下执行的；
  - 当时配置里又是 `use_lora=null`，
  - 这会直接导致 answer-protocol 历史 partial 与 baseline 高度一致。

### 2026-03-19 新增验证：当前评测器下 32-token raw-score 对照已完成
- 输出：
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_rawscore_compare_32tokens.json`
  - `/data/liushiqi/AutoVLA/task/artifacts/navhard_debug_2026-03-19/navhard_step6000_rawscore_compare_32tokens_summary.json`
- 口径：
  - 直接调用当前 `tools/eval/run_navhard_two_stage_autovla.py` 的 `_evaluate_token(...)`
  - 逐 token 比较原始 `score_row`
  - 不再使用先前误导性的 stage 摘要对比逻辑
- 结果：
  - `num_tokens=32`
  - `same_raw_score_rows=3`
  - `different_raw_score_rows=29`
  - `same_rate=0.09375`
- 解释：
  - 在当前评测器下，baseline `step6000` 与 answer-protocol `step6000` 的原始评分行绝大多数都已分叉；
  - 因而“当前评测器仍把它们评成一样”这一说法，当前已被 32-token 直接对照否定。
- 备注：
  - 这 32 个 token 中，后 12 个 stage-two token 不在当前 metric cache 中，会走相同的失败分支；
  - 但即便包含这些失败 token，raw-score 对照仍然是 `29/32` 分叉，说明结论非常稳。

### 2026-03-19 新增验证：单机小批量 smoke 已打通
- 新增脚本：
  - `/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_single.sh`
- 目标：
  - 绕开历史多卡 plan / merge 链路，
  - 直接用当前评测器做单机 `navhard_two_stage` 小批量验证。
- 固定调用入口：
  - `tools/eval/run_navhard_two_stage_autovla.py`
- 支持环境变量覆盖：
  - `CONFIG_PATH`
  - `CKPT_PATH`
  - `OUTPUT_DIR`
  - `MAX_STAGE_ONE`
  - `MAX_STAGE_TWO`
- baseline 小批量命令：
```bash
cd /data/liushiqi/AutoVLA
MAX_STAGE_ONE=4 MAX_STAGE_TWO=4 \
OUTPUT_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_single_smoke_baseline_2026-03-19 \
scripts/eval/run_navhard_two_stage_autovla_single.sh
```
- answer-protocol 小批量命令：
```bash
cd /data/liushiqi/AutoVLA
CKPT_PATH=/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt \
MAX_STAGE_ONE=4 MAX_STAGE_TWO=4 \
OUTPUT_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_single_smoke_answer_2026-03-19 \
scripts/eval/run_navhard_two_stage_autovla_single.sh
```
- baseline smoke 结果：
  - 输出：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_single_smoke_baseline_2026-03-19/run_2026-03-19_15-08-58/summary.json`
  - `num_successful_scenarios=8`
  - `num_failed_scenarios=0`
  - `final_extended_pdm_score=0.5355163980764478`
- answer-protocol smoke 结果：
  - 输出：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_single_smoke_answer_2026-03-19/run_2026-03-19_15-08-58/summary.json`
  - `num_successful_scenarios=8`
  - `num_failed_scenarios=0`
  - `final_extended_pdm_score=0.540674140675003`
- 结论：
  - 在单机、当前评测器、小批量场景下，两组 checkpoint 的分数已经分叉；
  - 因而“当前评测器一定会把两者评成完全一样”的判断不成立。

### 2026-03-19 新增验证：当前 8GPU sharded 规范脚本也已打通，且结果同样分叉
- 新增脚本：
  - `/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh`
- 对应测试：
  - `/data/liushiqi/AutoVLA/tools/eval/tests/test_navhard_multi_gpu_script.py`
- 设计目标：
  - 固定当前 sharded 链路：
    - `tools/eval/prepare_navhard_two_stage_shards.py`
    - `tools/eval/run_navhard_two_stage_autovla_shard.py`
    - `tools/eval/merge_navhard_two_stage_shards.py`
  - 同时支持：
    - `CKPT_PATH`
    - `PLAN_DIR`
    - `GPU_LIST`
    - `NUM_SHARDS`
    - `MAX_STAGE_ONE`
    - `MAX_STAGE_TWO`
- baseline 8GPU 小批量命令：
```bash
cd /data/liushiqi/AutoVLA
MAX_STAGE_ONE=8 MAX_STAGE_TWO=8 \
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_current_8gpu_baseline_cmp_2026-03-19 \
scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```
- answer-protocol 8GPU 小批量命令：
```bash
cd /data/liushiqi/AutoVLA
CKPT_PATH=/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt \
MAX_STAGE_ONE=8 MAX_STAGE_TWO=8 \
PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_current_8gpu_answer_cmp_2026-03-19 \
scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
```
- baseline 8GPU 结果：
  - 输出：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_current_8gpu_baseline_cmp_2026-03-19/merged/summary.json`
  - `num_successful_scenarios=16`
  - `num_failed_scenarios=0`
  - `final_extended_pdm_score=0.3187559378172864`
- answer-protocol 8GPU 结果：
  - 输出：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_current_8gpu_answer_cmp_2026-03-19/merged/summary.json`
  - `num_successful_scenarios=16`
  - `num_failed_scenarios=0`
  - `final_extended_pdm_score=0.4898116362596183`
- 结论：
  - 当前 8GPU sharded 规范链路下，baseline 与 answer-protocol 仍然显著分叉。
  - 且这组 8GPU 分数与同一批量的单机结果完全一致：
    - baseline：`0.3187559378172864`
    - answer-protocol：`0.4898116362596183`
  - 因而“只有单机链路会分叉、8GPU 链路会错误地评成一样”这一怀疑，当前也已被否定。

### 2026-03-19 补充修正：当前本地 navhard 默认评测口径已切到 8 poses
- 问题确认：
  - 之前本地 `config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml` 仍写着：
    - `model.trajectory_sampling.num_poses=10`
  - 这会让当前 `tools/eval/run_navhard_two_stage_autovla.py` 在模型预测后，再按外层 eval 配置补齐到 10 poses。
- 已修正为：
  - `model.trajectory_sampling.num_poses=8`
  - 文件：
    - `/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml`
- 保护测试：
  - `/data/liushiqi/AutoVLA/tools/eval/tests/test_navsimv2_protocol_configs.py`
  - 新增断言：`test_navhard_step6000_eval_uses_canonical_8_pose`
- 当前含义：
  - 从这次修正开始，默认单机脚本与当前 8GPU 规范脚本都将以 8 poses 口径跑这个 baseline navhard 配置。

### Navhard 测试清单（按当前 8-pose 本地评测口径）
| 类别 | 版本 | 目标 | 当前表内结果 | ckpt 路径 | 当前是否可测 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 历史 10 点 | 10 点 SFT，但按当前 8 点评测口径验证 | 待本轮顺序评测 | `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt` | 是 | 该 run 的 `hparams.yaml` 记录为 `num_poses=10` |
| RFT | 历史 10 点 baseline | baseline `step6000`，按当前 8 点评测口径验证 | 历史可疑结果曾记为 `0.29853212542780166` | `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt` | 是 | 历史 run 的 `hparams.yaml` 记录为 `num_poses=10` |
| RFT | 历史 10 点 answer-protocol | answer-protocol `step6000`，按当前 8 点评测口径验证 | 历史可疑结果曾记为 `0.29853212542780166` | `/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt` | 是 | 历史 run 的 `hparams.yaml` 记录为 `num_poses=10` |
| RFT | 历史 10 点 answer-protocol | answer-protocol `step12000`，按当前 8 点评测口径验证 | 历史可疑结果曾记为 `0.29853212542780166` | `/data/liushiqi/AutoVLA/.worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step12000-reward6.6875.ckpt` | 是 | 历史 run 的 `hparams.yaml` 记录为 `num_poses=10` |
| SFT | 当前 8 点 | 当前 8 点 SFT 基线 | `0.12055106546662982` | `/data/liushiqi/AutoVLA/runs/sft/navsimv2_sft8_local_8gpu_retry3_2026-03-18_09-14-56/epoch=4-loss=0.8492.ckpt` | 是 | `hparams.yaml` 记录为 `num_poses=8`；输出见 `/data/liushiqi/AutoVLA/logs/eval/navhard_sequence_local_2026-03-20_current_first/01_sft8_epoch4/merged/summary.json` |
| RFT | 当前 8 点 default-format | 当前 8 点 default-format `step6000` | 待本轮顺序评测 | `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft8_default_format_ip33_restart1_2026-03-19_09-16-45/ckpt/rft-step6000-reward6.2500.ckpt` | 是 | 当前可直接测 |
| RFT | 当前 8 点 default-format | 当前 8 点 default-format `step12000` | 待本轮顺序评测 | `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft8_default_format_ip33_restart1_2026-03-19_09-16-45/ckpt/rft-step12000-reward7.2500.ckpt` | 是 | 当前已产出可直接测 |
| RFT | 当前 8 点 answer-format | 当前 8 点 answer-format `step6000` | `0.12795913963597189` | `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft8_answer_format_local_restart1_2026-03-19_09-16-46/ckpt/rft-step6000-reward5.9375.ckpt` | 是 | 输出见 `/data/liushiqi/AutoVLA/logs/eval/navhard_sequence_ip33_2026-03-20_current_first/01_rft8_answer_step6000/merged/summary.json` |
| RFT | 当前 8 点 answer-format | 当前 8 点 answer-format `step12000` | 待 33 顺序评测 | `/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft8_answer_format_local_restart1_2026-03-19_09-16-46/ckpt/rft-step12000-reward7.1250.ckpt` | 是 | 当前已产出可直接测 |

### 当前建议的测试顺序
1. 历史 10 点：
   - SFT
   - baseline `step6000`
   - answer-protocol `step6000`
   - answer-protocol `step12000`
2. 当前 8 点：
   - SFT
   - default-format `step6000`
   - answer-format `step6000`
3. 等当前 8 点 run 真正产出 `step12000` 再补测：
   - default-format `step12000`
   - answer-format `step12000`

### 2026-03-19 新增：后台顺序测试器与 190 远程启动器
- 新增队列 worker：
  - `/data/liushiqi/AutoVLA/tools/eval/run_navhard_eval_queue.py`
- 新增测试：
  - `/data/liushiqi/AutoVLA/tools/eval/tests/test_navhard_eval_queue.py`
  - `/data/liushiqi/AutoVLA/tools/eval/tests/test_navhard_queue_launcher_script.py`
- 新增 190 远程后台启动脚本：
  - `/data/liushiqi/AutoVLA/scripts/eval/start_navhard_eval_queue_ip190.sh`
- 设计：
  - 自动生成 `manifest.json`
  - 顺序执行默认 navhard 清单
  - 缺失 ckpt 自动标成 `waiting_ckpt`
  - 支持 `--task-id` 只跑指定任务
  - 支持 `--execution-mode single|8gpu`
  - 支持 `--poll-missing` 持续轮询缺失 ckpt，后续产出后自动接着跑
  - 每个任务单独落盘：
    - `worker_stdout.log`
    - `worker_stderr.log`
    - `run_*/resolved_config.yaml`
    - `run_*/summary.json`
- 本地队列验证目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_queue_local_verify_2026-03-19`
- 190 远程后台验证目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_queue_ip190_verify_2026-03-19`
- 190 全清单顺序 smoke 队列：
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_queue_ip190_smoke_all_2026-03-19`
- 190 全清单正式 8GPU 顺序队列：
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_queue_ip190_8gpu_full_all_2026-03-19`
  - 当前先由后台守护等待现存 shard 任务结束，再自动拉起真正的 queue worker：
    - `wait_launcher.pid`
    - `wait_then_launch.log`
- 当前验证到的事实：
  - `190` SSH 可达
  - 远程 `nohup` worker 已成功启动并生成：
    - `worker.pid`
    - `manifest.json`
    - `runs/sft8_epoch4_pose8eval/run_*/resolved_config.yaml`
  - 说明“远程后台拉起 + 单任务 smoke 开始执行”这条链路已打通。
  - 2026-03-19 晚些时候又补成了真正的 `8gpu` 启动器参数透传：
    - `EXECUTION_MODE=8gpu`
    - `POLL_MISSING=1`
    - `POLL_INTERVAL_SEC`
    - `MAX_WAIT_CYCLES`
  - 190 上旧的单机 queue 残留已清掉；新的 8GPU 队列不会抢占当前正在推进的 shard 作业，而是等它们结束后再启动。
  - 这样当前已有 ckpt 会先顺序跑完，`rft8_default_step12000_pose8eval` 和 `rft8_answer_step12000_pose8eval` 若后续真正落盘，可由同一后台队列自动转为 `pending` 并继续评测。

### 2026-03-20 新增：停止 190，改为本机 + 33 分摊顺序评测
- 用户确认两个当前 8 点 `step12000` ckpt 已经产出，因此不再需要 `190` 上的 `waiting_ckpt` 轮询队列。
- 已停止 `190` 上与这批 navhard 顺序评测相关的等待器/旧 shard 残留，不再继续在 `190` 排队。
- 改为两台 8GPU 机器分摊：
  - 本机运行目录：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_sequence_local_2026-03-20_current_first`
  - `33` 运行目录：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_sequence_ip33_2026-03-20_current_first`
- 当前分配与顺序：
  - 本机：
    - `01_sft8_epoch4`
    - `02_rft8_default_step6000`
    - `03_rft8_default_step12000`
    - `04_sft10_epoch4`
    - `05_rft10_baseline_step6000`
  - `33`：
    - `01_rft8_answer_step6000`
    - `02_rft8_answer_step12000`
    - `03_rft10_answer_step6000`
    - `04_rft10_answer_step12000`
- 启动状态：
  - 本机已进入：
    - `01_sft8_epoch4`
    - 8 个 shard 均已拉起
  - `33` 已进入：
    - `01_rft8_answer_step6000`
    - 8 个 shard 均已拉起

### 新怀疑点
- 需要优先排查以下可能性：
  - 历史 answer-protocol full eval 目录中的 merged csv 是否被旧结果覆盖或复用。
  - 当时 answer-protocol full eval 使用的工作树代码是否与当前代码不同。
  - 先前 `currentscore_16tokens` 对比脚本是否错误比较了 stage 聚合后的占位行，导致误报“评分一致”。
  - 历史 merge/summary 阶段是否也可能存在类似的取行或覆盖问题。
- 当前已完成的中间检查：
  - baseline 与 answer-protocol 的 `partials/shard_00/partial_rows.csv` 的 SHA1 不同：
    - baseline：
      - `bb8d141844f58fbb81f93bf74e50076cd2c7fd06`
    - answer-protocol：
      - `21eb6b80a3644336782ba11396944e8359ed452b`
  - 但逐列对比发现：
    - 只有 `ego_simulated_states` 一列在 `26/739` 行不同
    - 其余指标列相同
  - 说明：
    - 历史 shard 输出并非完全同文件复制，
    - 但从评分列看，历史 full eval 仍然等价于 baseline 结果。

### 下一步验证
1. 修复或重写 `currentscore_16tokens` 对比脚本
   - 必须直接比较 `_evaluate_token` 的原始 `score_row`
   - 不能再比较裁剪后的 stage 摘要行
2. 把直接探针从 3 个 token 扩到 16 或 32 个 token
   - 统计真实的 `raw_score_row_equal` 比例
3. 若扩样后仍持续分叉
   - 则历史 full `navhard` 一致更强地指向：
     - 旧产物复用 / 覆盖
     - 历史 merge 阶段串用
     - 历史工作树代码路径不同
4. 再回头排查历史 merged csv 的生成链路
   - 重点检查 shard 合并与 summary 聚合逻辑

## 设计原则
1. 动作答案必须有显式边界，且位于 completion 末尾。
2. reward 只消费“最后一个合法 action block”。
3. 非法格式应显式判定，而不是自动补成看似合法的轨迹。
4. 训练监控必须能区分：
   - completion 是否包含 action block
   - action block 是否位于末尾
   - action block 是否长度合法
   - reward 是否基于真实 block 还是 fallback/invalid 分支

## 计划

### 阶段 1：定义输出协议
- 明确 RFT completion 的目标结构：
  - 前半段：可选语言分析
  - 最后：唯一末尾 action block
- 明确 action block 的边界规则：
  - 如何开始
  - 如何结束
  - 是否允许 block 后再出现普通文本
- 明确“末尾”的判定口径：
  - 最后一个非 EOS 的有效答案区必须为 action block

### 阶段 2：重构动作提取语义
- 不再对全 completion 做无差别 action token 扫描。
- 改为只提取“最后一个 action block”中的动作 token。
- 若 completion 中前面出现零散 action token，但不属于最后 block，不进入 reward 轨迹。

### 阶段 3：重构 reward 输入语义
- 重新定义 reward 前的数据合法性：
  - block 缺失
  - block 长度不足
  - block 长度超限
  - block 后仍有无关文本
- 对上述情况设计显式策略：
  - `invalid/format_penalty`
  - `skip reward`
  - 或单独的非法输出分支
- 原则上不再默认“补零/截断后继续当正常轨迹算 reward”。

### 阶段 4：训练监控补充
- 新增或对齐以下监控：
  - `sample_has_action_block`
  - `sample_action_block_at_tail`
  - `sample_action_block_tokens_len`
  - `sample_action_block_valid`
  - `reward_input_valid`
  - `reward_invalid_reason`
- 区分：
  - completion 中 action token 个数
  - 真正用于 reward 的 block token 个数

### 阶段 5：SFT/RFT 对齐
- 检查 SFT 是否需要同步采用相同的“末尾 action block”监督模板。
- 若 SFT 不对齐，需评估：
  - 训练分布与 RFT 输出协议不一致
  - 可能导致 RFT 初期大量非法 block
- 优先目标：
  - SFT 与 RFT 至少在“最终动作答案的边界和位置”上对齐

### 阶段 6：小规模验证
- 先做小步 RFT 冒烟：
  - 关注 action block 合法率
  - 关注 reward invalid 比例
  - 关注 `group_reward_std` 是否改善
- 再决定是否上 full 8 卡长跑

## 验收方案

### A. 协议验收
- completion 中可以有语言，但最终动作必须落在唯一末尾 action block。
- 不允许 reward 从前文零散 action token 中取值。

### B. 提取验收
- 对同一条 completion：
  - 仅最后 block 被解析为动作轨迹。
  - block 外 action token 不影响 reward 输入。

### C. 非法输出验收
- 对以下情况能稳定区分并记录：
  - 无 action block
  - action block 不在末尾
  - action block 长度不足
  - action block 长度超限
  - action block 后仍有额外文本
- 非法输出不再通过补零/截断伪装成正常 reward 样本。

### D. 训练指标验收
- TensorBoard / CSV 中可直接看到：
  - 真正用于 reward 的 action block 长度
  - 非法 block 比例
  - reward valid/invalid 比例
  - `group_reward_std`
  - `group_adv_fallback`

### E. 效果验收
- 相比当前 baseline，至少满足以下一项：
  - `group_reward_std=0` 的占比下降
  - `group_adv_fallback=1` 的占比下降
  - action block 合法率稳定提升
  - 后续 `navtest/navhard` 标准 EPDMS 不下降

## 决策记录
- 已确定采用：方案 A，末尾 action block。
- 当前阶段先做设计与任务拆解，不直接改代码。

## 相关参考
- baseline 总任务：
  - `/data/liushiqi/AutoVLA/task/navsimv2_sft_rft_todo.md`
- 当前 RFT 配置：
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4.yaml`
- 当前动作提取与 reward 输入实现：
  - `/data/liushiqi/AutoVLA/models/autovla.py`
