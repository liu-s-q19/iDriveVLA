# NavSim v2 RFT 末尾 Action Block 方案 A

状态：规划中

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
