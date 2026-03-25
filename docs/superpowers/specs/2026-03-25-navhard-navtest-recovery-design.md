# Navhard / Navtest Recovery Design (2026-03-25)

## 1. Goal

在 `NavSim v2` 主线下，为 `ReCogDrive-VLM-2B` 制定下一轮提升策略，优先提升 `navhard`，同时不接受 `navtest` 明显回撤。

这份设计不直接假设以下说法已经成立：

- `answer-format` 一定优于 `default-format`
- 新 `codebook` 一定更差
- 加 `CoT` 一定能救分
- 历史帧越多越好

当前证据不足以支持这些结论，需要重新做单变量拆分。

## 2. Current Evidence

### 2.1 What is actually working

- 当前最强结果来自 `ReCogDrive-VLM-2B` 的 `default-format` RFT：
  - `navhard=0.21467653443621454`
  - `navtest=0.7735481263572249`
- 当前次强 SFT 基线也来自 `ReCogDrive-VLM-2B`：
  - `navhard=0.20946779988176029`
  - `navtest=0.7379779677450613`

这说明当前主线并没有“完全不会学”，问题是进一步提升已经卡住，而不是系统全线失效。

### 2.2 What has already looked weak

- `answer-format` 历史全量结果明显弱于 `default-format`。
- 虽然 `2026-03-24` 已修掉 `</answer>` 尾部 special token 误杀，但主阻塞只从“格式被错杀”变成了“8 token 稳定命中不足”。
- 因此 `answer-format` 现在最多算“待重新验证”，不能当下一轮主攻方向。

### 2.3 What is still confounded

- `speedupab (05:48:12)` 与 `mainline (12:29:45)` 不是单变量对比：
  - 同时混入了 `old codebook -> new codebook`
  - 同时混入了 `probe on -> probe off`
- 结果出现：
  - `val_loss: 0.938 -> 1.130`
  - `navhard: 0.2095 -> 0.0100`
- 这个结果只能说明“新主线组合失败了”，不能说明“新 codebook 本身失败”。

补充约束：

- 当前 `generated_action_probe` 在代码里主要是日志/探针逻辑，不是主训练目标的一部分。
- 因此它更像“可能影响吞吐或稳定性的小扰动”，而不是当前最可信的性能主因。
- 这意味着后续实验应把 `codebook` 作为主变量，把 `probe on/off` 作为次级烟雾测试，而不是等权主矩阵。

### 2.4 What offline evidence says about codebook

- 分 maneuver 新码本在离线离散误差上，`new_global` 明显优于旧码本。
- 真正退化的是 `maneuver_restricted` 硬限制版本，不是 `new_global`。
- 软 gating (`alpha=1.02`) 的离线结果接近或略优于 `new_global`。

因此，对 `codebook` 的合理结论不是“放弃”，而是：

- 先不要上硬限制
- 更应该测试 `new codebook + global`
- 再考虑 soft-gated / residual / hierarchical 版本

### 2.5 What current evidence says about CoT

- `ReCogDrive-VLM-2B` 现有较强路线基本都在 `use_cot=false` 上。
- 现阶段 RFT 的主要问题仍是协议稳定性、输出有效率、动作长度命中和 reward 信号质量。
- 在这种状态下直接引入自由文本 `CoT`，大概率会增加输出熵和格式噪声，而不是先解决主瓶颈。

结论：`CoT` 不是第一优先级，但可以做“短结构化推理”替代长自然语言推理。

### 2.6 What current evidence says about visual input

- 当前 prompt 固定喂入：
  - 3 个前向视角
  - 每个视角 4 帧
  - 共 12 张历史图像
- 当前仓库没有看到这部分做过系统 ablation。
- 对 `ReCogDrive-VLM-2B` 这种较小模型，12 图像时序输入很可能已经接近注意力预算上限。

因此，“减少历史帧/压缩图像上下文”是高优先级待验证方向，而且它同时可能改善吞吐、训练稳定性和泛化。

## 3. Candidate Approaches

### Approach A: Format-first / CoT-first

做法：

- 继续推进 `answer-format`
- 继续做自然语言 `CoT`
- 希望通过更强协议约束和更长推理提升规划质量

优点：

- 改动集中在 prompt / parser / RFT 协议
- 不需要大改 action representation

缺点：

- 已有证据对它不友好
- 现在的主要瓶颈并不是“模型不会说正确格式”本身
- 自然语言 CoT 会进一步拉高输出不稳定性

结论：

- 不建议作为下一轮主线

### Approach B: Representation-first

做法：

- 保留 `default-format`
- 重新公平验证 `new codebook + global`
- 在通过公平验证后，继续做 soft-gated / residual / hierarchical codebook
- 额外加入轻量辅助任务，让模型先学“意图”和“轨迹粗形状”，再学离散 token

优点：

- 直接打到动作表达能力的主问题
- 与离线证据一致
- 对 `navhard` 更有机会产生结构性增益

缺点：

- 需要更多对照实验
- 若没有严格单变量，很容易再次被混杂因素污染结论

结论：

- 推荐，且应作为主线之一

### Approach C: Input-first simplification

做法：

- 缩减视觉上下文
- 缩短文本提示
- 优先保留最近帧和最有用视角
- 用更小但更高密度的信息输入替代“12 图像 + 长文字说明”

优点：

- 对小模型通常回报很高
- 能同时改善吞吐和训练质量
- 成本低，验证快

缺点：

- 需要小心避免删掉关键侧向交互信息
- 需要设计合理的 frame/view ablation

结论：

- 推荐，且应与 Approach B 并行

## 4. Recommended Strategy

推荐采用 `B + C` 联合路线，不建议把下一轮主赌注押在 `answer-format` 或自由文本 `CoT` 上。

核心判断：

1. 当前最可能的上限瓶颈，不是“缺自然语言推理”，而是小模型在高视觉负载下对离散轨迹 token 的表达能力不足。
2. 当前最值得争取的收益，不是更强输出协议，而是更好的输入压缩 + 更稳的动作表示。
3. 当前最危险的误判，是把混杂实验错误解读成“新 codebook 不行”。

## 5. Concrete Plan

### Phase 0: Remove confounders first

先做最小单变量拆分，目标是确认 `new codebook + global` 是否真的能站住。

主实验矩阵：

1. `old codebook + stable baseline config`
2. `new codebook + global + same stable baseline config`

次级烟雾测试：

3. 在最佳一组上补一条 `probe off`，只检查：
   - 是否影响吞吐
   - 是否引入异常
   - 是否出现意外质量波动

要求：

- 其他训练项完全固定
- 先做 `1k~2k step smoke`
- 用相同 `navhard mini-eval + navtest mini-eval + val_loss + token stats` 比较

决策标准：

- 如果 `new codebook + global` 接近或优于旧码本，继续走新码本路线
- 如果 `new codebook + global` 明显差，再回到码本构建细节排查
- `probe on/off` 只作为辅助判断，不作为主方向分叉依据

这是最高优先级，因为它决定后续所有表示学习判断是否站得住。

### Phase 1: Visual input ablation

不要先删相机，先删“冗余时序”。

建议顺序：

1. `3 views x 2 frames`，仅保留最近两帧
2. `front 4 frames + left/right 2 recent frames`
3. `front 2 frames + left/right 2 recent frames`

理由：

- `navhard` 需要侧向信息，直接 front-only 风险太大
- 但对 2B 模型，12 图像很可能已经过重
- 最近帧通常比早期帧更重要

同时做两项 prompt 压缩：

- 把三段“first video / second video / third video”冗长描述压成固定短模板
- 去掉重复的场景说明文字，只保留任务定义、车辆状态和驾驶指令

验收指标：

- 训练吞吐
- `val_loss`
- action token 命中率
- 小规模 `navhard` 对照

预期：

- 这是最有机会“又提速又提分”的方向

### Phase 2: Better action representation

如果 `Phase 0` 确认新码本可用，则继续：

1. 主线保留 `new codebook + global`
2. 不上硬 `maneuver_restricted`
3. 加 soft-gated candidate selection，而不是直接子码本硬切
4. 再加一个小 residual/refiner，修正离散中心到连续轨迹的偏差

建议的 representation 路线：

- Level 1: 预测 maneuver / intent
- Level 2: 在全局码本上预测主 token
- Level 3: 预测小 residual 或 endpoint delta

辅助监督建议：

- maneuver 分类 loss
- endpoint / heading / curvature 粗目标 loss
- action token 主损失继续保留

原因：

- `navhard` 退化常见原因不是“完全不会开”，而是转弯、避让、刹停这类少数高风险样本表达不准
- 这些样本靠单一全局 token 很容易表达不够

### Phase 3: RFT only after SFT signal is cleaner

RFT 下一轮建议：

- 主线继续 `default-format`
- 不把 `answer-format` 当主线
- 不先引入自然语言 `CoT`

可以做的不是长文本 CoT，而是“结构化 latent reasoning”：

- 先预测 `maneuver`
- 是否需要让行 / 刹停
- 轨迹粗类别（keep / left / right / stop / yield）

这些 latent 可以：

- 作为 SFT 辅助头
- 或作为 RFT reward bonus
- 但不要求模型在最终输出中生成长自然语言解释

RFT 奖励建议：

- 把格式奖励降到从属地位
- 主奖励围绕轨迹质量、有效性、mismatch 减少、少数复杂场景表现
- 对高风险 maneuver 样本做更高权重采样或 reward reweighting

## 6. Promotion Gates

每轮实验都必须同时看 `navhard` 和 `navtest`，不能只看单一分数。

建议维护一张固定 scorecard：

- `val_loss`
- action token 命中率
- `navhard mini`
- `navtest mini`
- `left/right` 分桶表现
- 高风险场景分桶表现（转弯、避让、停车）
- 吞吐与显存

promotion 规则：

1. 任一候选若 `navtest mini` 明显回撤，则不进入全量评测
2. 任一候选若只提升 `keep forward`、但 `left/right` 明显下滑，则不升主线
3. 只有在 mini scorecard 稳定时，才允许进入全量 `navhard` / `navtest`

建议的硬门槛：

- `navtest mini` 相对当前稳定基线下降不得超过 `0.005`
- `left/right` 任一分桶不得出现显著塌陷
- 吞吐若下降超过 `15%`，必须证明质量收益足够覆盖成本

## 7. Eval Design

为了避免 full eval 太慢，先固定两个小集合：

- `navhard mini`
- `navtest mini`

要求这两个集合都要分层抽样，而不是随机抽样：

- `keep forward`
- `turn left`
- `turn right`
- 停车/让行/拥堵等高风险场景

原因：

- 当前最可能的假提升，是整体分数小升，但少数高风险场景崩掉
- 尤其 `navtest` 很容易被“平均分掩盖局部退化”

## 8. What Not To Do First

以下方向不建议现在优先：

1. 不建议直接全面转向 `answer-format`
2. 不建议先赌长自然语言 `CoT`
3. 不建议继续把 `maneuver_restricted` 硬限制推上主线
4. 不建议在没有拆掉 `probe/codebook` 混杂前就下结论说“新码本不行”

## 9. 3-Round Experiment Order

如果只能排三轮，我建议这样排：

### Round 1

- 做 `Phase 0` 单变量拆分
- 目标：确定 `probe` 和 `new codebook` 各自作用

### Round 2

- 在最优 `codebook/probe` 组合上做 `Phase 1` 输入 ablation
- 目标：确认 12 图像是否已经过载

### Round 3

- 在最优输入配置上做 `Phase 2` 的 soft-gated / residual representation
- 目标：专门冲击 `navhard`

只有在这三轮之后仍然卡住，再考虑：

- 结构化 CoT
- answer-format 回归
- 更激进的 RFT reward 改造

## 10. Final Recommendation

一句话结论：

- 下一步最值得做的不是“先上 CoT”，而是“先拆混杂，再减视觉负担，再升级 action representation”。

优先级排序：

1. `probe/codebook` 单变量拆分
2. 历史帧/视角压缩 ablation
3. `new codebook + global -> soft-gated/residual` 表达升级
4. 结构化 latent reasoning
5. 自然语言 CoT
6. answer-format 主线化
