# AutoVLA NavSim v2：SFT -> RFT Todo 与阶段验收

状态：进行中

## 目标
- 按顺序推进：
  - 阶段 3：SFT
  - 阶段 4：RFT/GRPO
- 训练链路统一到 NavSim v2（不使用仓库内置 navsim v1.1）。
- SFT 保持 mix-sft 结构，但当前先用 NavSim v2 的 trainval 数据跑通。

## 当前路径盘点（2026-03-08）
- 模型：
  - `/data/ckpt/Qwen/Qwen2.5-VL-3B-Instruct` (exists)
- SFT 数据：
  - train: `/data/dataset/navsim/preprocessed/navtrainval_nocot` (103288 json)
  - val: `/data/dataset/navsim/preprocessed/navtest_nocot` (109 json)
- RFT reward cache（v2 schema）：
  - train: `/data/dataset/navsim/metric_cache_v2/navtrain_full_2026-03-07_15-49-21` (103288 metric_cache.pkl)
  - val: `/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49` (12146 metric_cache.pkl)
- NavSim v2 scene filter：
  - navtrain: `/data/liushiqi/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml`
  - navtest: `/data/liushiqi/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml`

## 阶段门槛规则
1. 严格顺序：先 SFT，再 RFT。
2. 每阶段固定产出：
   - 运行命令
   - 关键配置与产物路径
   - 验收结果（通过/失败 + 原因）
   - 下一阶段计划
3. 未通过不得进入下一阶段。

## 阶段 3：SFT（已完成）
### 目标
- 使用 NavSim v2 数据链路完成一轮可用 SFT，得到可加载 checkpoint。

### 配置与入口
- 配置：`config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
- 入口：`scripts/run_sft.sh`

### 当前生效配置（最终口径）
- 运行环境：
  - `CONDA_ENV=autolsqv2`
  - `PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python`
- 分布式与性能：
  - `distributed_strategy=ddp`
  - `ddp_find_unused_parameters=true`
  - `gradient_checkpointing=false`
  - `pin_memory=true`
  - `persistent_workers=true`
  - `prefetch_factor=2`
- 多机网络：
  - `NCCL_NET_MODE=fast_ib`
  - `NCCL_IB_HCA=mlx5_bond_0`
  - `NCCL_SOCKET_IFNAME=bond4`
  - `GLOO_SOCKET_IFNAME=bond4`

### 当前标准启动命令（最终口径）
```bash
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
NCCL_NET_MODE=fast_ib \
NCCL_IB_HCA=mlx5_bond_0 \
RUN_MASTER_BG=1 \
AUTO_CHECK=1 \
bash scripts/run_sft_3nodes_ssh.sh
```

### 当前三节点启动方式（2026-03-08）
- 启动命令：
```bash
RUN_MASTER_BG=1 AUTO_CHECK=1 bash scripts/run_sft_3nodes_ssh.sh
```
- 启动脚本：`scripts/run_sft_3nodes_ssh.sh`
- 节点固定：
  - rank0: `10.199.7.32`
  - rank1: `10.199.7.33`
  - rank2: `10.199.7.190`
  - `nproc_per_node=8`, `nnodes=3`
- 行为约定：
  - rank1/rank2 通过 SSH + `nohup torchrun` 启动。
  - rank0 在后台模式下也通过 SSH 启动（避免本地后台进程被回收）。
  - 环境变量统一由脚本注入：`NAVSIM_DEVKIT_ROOT`、`PYTHONPATH`、`PL_NUM_NODES`、`SFT_RUN_TS` 等。
- 产物路径：
  - 日志：`logs/train/sft_navsimv2_3nodes_<ts>_rank{0,1,2}.log`
  - pid：`logs/train/sft_navsimv2_3nodes_<ts>_rank{0,1,2}.pid`
- 前台排障命令：
```bash
RUN_MASTER_BG=0 AUTO_CHECK=0 bash scripts/run_sft_3nodes_ssh.sh
```
- 正确启动口径（本阶段统一）：
  - 正式跑数只用：`RUN_MASTER_BG=1 AUTO_CHECK=1 bash scripts/run_sft_3nodes_ssh.sh`
  - 只在排障时用前台模式；不要手动分别拉起 rank0/1/2。
  - 运行环境固定：`CONDA_ENV=autolsqv2`，`python=/data/miniconda/envs/autolsqv2/bin/python`（NavSim v2 环境）。
  - 脚本支持网络模式切换：
    - 稳定模式（默认）：`NCCL_NET_MODE=stable_tcp`（`NCCL_IB_DISABLE=1`）
    - 提速模式：`NCCL_NET_MODE=fast_ib`（默认 `NCCL_IB_HCA=mlx5_bond_0`，开启 IB）

### 提速版启动方式（2026-03-08 新增）
- 快速命令（推荐先试）：
```bash
CONDA_ENV=autolsqv2 PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python NCCL_NET_MODE=fast_ib NCCL_IB_HCA=mlx5_bond_0 RUN_MASTER_BG=1 AUTO_CHECK=1 bash scripts/run_sft_3nodes_ssh.sh
```
- 回退命令（若 fast_ib 不稳定）：
```bash
CONDA_ENV=autolsqv2 PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python NCCL_NET_MODE=stable_tcp RUN_MASTER_BG=1 AUTO_CHECK=1 bash scripts/run_sft_3nodes_ssh.sh
```
- 脚本会在启动前自动探测 3 节点是否存在 `/sys/class/infiniband/${NCCL_IB_HCA}`，便于快速判断 IB 配置是否一致。

### keep*.sh 占用说明（排障口径）
- `/data/liushiqi/keeper/keep_*.sh` 的占用不作为“训练是否卡住/失败”的直接判据。
- 默认不以 keep 占用判断当前 run 异常；仅在“启动前无可用 GPU 导致无法起任务”时再清理 keep 进程。

### 最小执行
```bash
bash scripts/run_sft.sh
```

### 验收标准
- 成功启动并稳定迭代（无路径/导入错误）。
- 生成 checkpoint（`runs/sft/<timestamp>/*.ckpt`）。
- 训练日志可见 `train_loss/val_loss`，前段收敛趋势可读。

### 当前状态
- [x] 配置改造完成（navsim v2 路径 + trainval 数据）
- [x] 训练启动
- [x] checkpoint 产出
- [x] 阶段验收



### 三机 vs 单机时间对比（2026-03-08 13:58 UTC）
- 三机当前（24 GPU，DDP，run `13:50:52`）：
  - 速度：`~0.29 it/s`
  - steps/epoch：`4304`
  - 预计：`4304 / 0.29 / 3600 ≈ 4.12h/epoch`
  - 5 epoch 预计：`≈20.6h`
- 单机历史（上阶段记录口径，8 GPU）：
  - 速度：`~1.09 it/s`
  - steps/epoch：`12911`
  - 预计：`≈3.29h/epoch`
  - 5 epoch 预计：`≈16.4h`
- 对比结论：
  - 按“当前各自实测口径”对比，三机总时长约为单机的 `1.26x`（`20.6 / 16.4`）。
  - 注意：两者 `steps/epoch` 不一致（`4304` vs `12911`），该对比用于排期评估，不用于严格吞吐 benchmark。

### 单机对比运行记录（2026-03-08 14:15 UTC）
- 已停止三机 run：`sft_navsimv2_3nodes_2026-03-08_13-50-52`。
- 已启动单机 8 卡 run：`sft_navsimv2_single_2026-03-08_14-14-39`。
- 环境：`autolsqv2`（`/data/miniconda/envs/autolsqv2/bin/python`）。
- 进程：
  - torchrun pid：`3878527`
  - pid 文件：`logs/train/sft_navsimv2_single_2026-03-08_14-14-39_rank0.pid`
- 当前速度快照（rank0 stdout）：`~2.1 it/s`（明显高于三机 `~0.29 it/s`）。

### 结果保存路径与可视化（2026-03-08 14:18 UTC）
- 训练进程日志（torchrun 按 rank 拆分）：
  - 根目录：`logs/train/sft_navsimv2_single_2026-03-08_14-14-39_torchrun_logs/`
  - rank0 主日志：
    - `logs/train/sft_navsimv2_single_2026-03-08_14-14-39_torchrun_logs/none_vlp3uxl3/attempt_0/0/stdout.log`
    - `logs/train/sft_navsimv2_single_2026-03-08_14-14-39_torchrun_logs/none_vlp3uxl3/attempt_0/0/stderr.log`
- Lightning 训练产物目录（由训练进程时间戳生成）：
  - `runs/sft/2026-03-08_14-15-17/`
  - 指标 CSV：`runs/sft/2026-03-08_14-15-17/lightning_logs/version_0/metrics.csv`
  - 超参：`runs/sft/2026-03-08_14-15-17/lightning_logs/version_0/hparams.yaml`
  - checkpoint 实际产物（`save_top_k=3`）：
    - `runs/sft/2026-03-08_14-15-17/epoch=2-loss=0.6369.ckpt`
    - `runs/sft/2026-03-08_14-15-17/epoch=3-loss=0.6227.ckpt`
    - `runs/sft/2026-03-08_14-15-17/epoch=4-loss=0.6140.ckpt`
- TensorBoard 说明（当前口径）：
  - 代码已更新（2026-03-09）：`tools/run_sft.py` 默认使用 `TensorBoardLogger`。
  - 配置开关：`training.logging.type`，支持 `tensorboard | csv | both`（当前默认 `tensorboard`）。
  - 新启动的 run 会在 `runs/sft/<timestamp>/lightning_logs/version_*/` 生成 `events.out.tfevents.*`。
  - 旧 run `2026-03-08_14-15-17` 仍是 CSV 口径（该次运行发生在改造前）。
  - 查看命令：`tensorboard --logdir /data/liushiqi/AutoVLA/runs/sft --port 6006 --bind_all`

### 单机 SFT 完训统计（2026-03-09 00:00 UTC 汇总）
- 运行：`sft_navsimv2_single_2026-03-08_14-14-39`（对应 Lightning 目录 `runs/sft/2026-03-08_14-15-17`）。
- 完成性：
  - `rank0 stdout` 已到 `Epoch 4: 100%` 且完成末次验证，最终显示 `val_loss=0.614`。
  - 2026-03-09 检查无该 run 的存活 `torchrun/run_sft.py` 进程。
- 时间：
  - 开始（目录时间）：约 `2026-03-08 14:15 UTC`
  - 结束（最佳 ckpt 写入）：`2026-03-08 22:35:02 UTC`
  - 总耗时：约 `8h19m45s`
- 指标（来自 `metrics.csv`）：
  - `val_loss`（按 epoch 末）：`0.7506 -> 0.6672 -> 0.6369 -> 0.6227 -> 0.6140`
  - 最优 `val_loss`：`0.6140483618`（epoch `4`, step `16139`）
  - 最终 `val_loss`：`0.6140483618`（epoch `4`, step `16139`）
  - 最后记录 `train_loss`：`0.5366994143`（epoch `4`, step `16099`）
- 阶段 3 验收结论：
  - 通过（有完整训练、可用 checkpoint、指标可追踪，满足进入 RFT 的前置条件）。

## 阶段 4：RFT/GRPO（待阶段3通过）
### 目标
- 基于阶段3最佳 SFT checkpoint，跑通 NavSim v2 reward cache 的 GRPO。

### 配置与入口
- 配置：`config/training/qwen2.5-vl-3B-navsimv2-grpo-cot.yaml`
- 入口：`scripts/run_rft.sh`
- 注意：运行前需替换 `model.sft_model_path` 为阶段3产物。

### 最小执行
```bash
bash scripts/run_rft.sh
```

### 验收标准
- GRPO 训练稳定推进（无 token/cache/schema 断裂）。
- 产出 checkpoint（`runs/grpo/<timestamp>/*.ckpt`）。
- 日志可看到 reward、kl、loss 等核心指标。

### 当前状态
- [x] 配置改造完成（navsim v2 cache + navtrain/navtest split）
- [x] 填写 sft_model_path
- [x] 训练启动
- [ ] checkpoint 产出
- [ ] 阶段验收

### 当前 RL 启动口径（2026-03-16）
- 本次初始化 SFT ckpt：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 本次专用配置：
  - `config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-sft20260312e4.yaml`
- 启动命令：
```bash
cd /data/liushiqi/AutoVLA
CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-sft20260312e4 \
bash scripts/run_rft.sh
```
- 关键监控项：
  - `loss`
  - `policy_loss`
  - `policy_objective`
  - `train_reward`（原始 reward）
  - `scaled_train_reward`（乘 `reward.scale` 后）
  - `avg_train_reward`
  - `train_advantage`
  - `kl_divergence`
  - `sample_has_action`
  - `sample_action_tokens_len`
  - `sample_action_candidate_count`
  - `sample_action_nonzero_count`
- 日志节奏：
  - `training.log_every_n_steps=10`
  - tqdm/Lightning 指标按 10 step 刷新
- 2026-03-16 实际后台等待启动：
  - tmux session：`rl_navsimv2_sft20260312e4_wait`
  - 等待器日志：`logs/train/rl_navsimv2_sft20260312e4_2026-03-16_09-37-46.log`
  - 当前状态：本机 8 卡正被 `navtest` 8 卡评测占用，等待器检测到 GPU 空闲后会自动启动 RL。

### 2026-03-16 RL 正式启动记录（IP 190）
- 启动节点：
  - `10.199.7.190`
- tmux session：
  - `rl_navsimv2_sft20260312e4_ip190`
- 启动日志：
  - `logs/train/rl_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45.log`
- run id：
  - `grpo_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45`
- run manifest：
  - `runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45/run_manifest.yaml`
- ckpt 目录：
  - `runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45/ckpt`
- csv 目录：
  - `runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45/csv`
- tensorboard 目录：
  - `tensorboard/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_09-44-45`
- 已修复启动阻塞：
  - `AutoVLAAgent(AbstractAgent)` 构造兼容问题。
  - 根因：当前 RL 运行环境下 `AbstractAgent.__init__` 仍要求 `trajectory_sampling`，而本仓库 `navsim/navsim/agents/autovla_agent.py` 只按新签名调用 `super().__init__(requires_scene=False)`，导致 `RFTDataset` 初始化直接报错。
  - 修复：新增 `navsim_ext/agent_ctor_compat.py`，按签名兼容调用 `AbstractAgent.__init__`，并已用单测覆盖新/旧签名两种情况。

### 2026-03-16 当前 RL 观测结论
- step 9 / 19 / 29 已落盘：
  - `sample_has_action=0.0`
  - `sample_action_tokens_len=0.0`
  - `group_reward_std=0.0`
  - `group_adv_fallback=1.0`
- 结论：
  - 当前 RL 自由采样输出里仍未生成 action token。
  - reward 在不同 step 间会变化，但同一步 8 卡组内 reward 仍完全相同，仍依赖 `advantage fallback`。

### “直接测试有 action” 与当前 RL 不一致的原因
- 之前直接 probe/SFT 口径并不等同于当前 RL：
  - SFT 配置：`use_cot=false`
  - SFT probe：`do_sample=false`
  - SFT probe：`max_new_tokens=2048`
  - 当前 RL：`use_cot=true`
  - 当前 RL：`do_sample=true`
  - 当前 RL：`max_new_tokens=256`
- 因此，“直接测试某些样本能出 action” 不能直接推出 “当前 RL 采样口径下也会稳定出 action”。
- 当前最可信口径应以 RL 训练时真实落盘指标为准；截至 step 29，结论仍是 `sample_action_tokens_len=0`。

### 待修复  （2026-03-10）
- [x] 修复 RFT 中 `advantage` 组内塌缩问题：当 `group_reward_std < eps` 时启用可配置 fallback（当前默认 `fallback_mode=reward`），避免 `advantage` 被归一化压成 0。
- [x] 修复调试日志死锁：`group_sampling_seeds` 的 `all_gather` 改为全 rank 执行，仅在 rank0 打印，避免卡在 step0。
- [x] 3-step 验证通过（`qwen2.5-vl-3B-navsimv2-grpo-cot-fast-advfix-check`）：
  - 日志显示每步 `group_reward_std=0.000000` 且触发 `group_advantage_fallback=reward`。
  - `group_sampling_seeds` 按 rank 区分（如 step0: `1234..1241`），确认采样种子已解耦。
  - 指标（`runs/lightning_logs/version_26/metrics.csv`）：
    - `group_adv_fallback=1.0`（step 0/1/2）
    - `train_advantage=5.4855 / 5.8704 / 6.8040`（非零）
    - `grad_norm=1.7813 / 0.9625 / 0.6768`（非零）
    - `loss=-5.4855 / -5.8703 / -6.8040`（不再是 0）
- [ ] 排查 `grouped_rewards` 组内同质化（8 卡 reward 完全相同）根因：
  - 当前结论：
    - `GroupSampler` 设计上每步给 8 卡同一个样本（GRPO 组采样语义），这是预期行为。
    - `group_sampling_seeds` 已不同，说明并非“同 seed”导致的同 reward。
    - 仍出现同 reward，说明生成/动作轨迹在组内塌缩为同一结果（或等价结果）。
  - 高优先排查：
    - 对比每 rank 的 `completion_ids` / action tokens 是否相同。
    - 检查 action token 截断与补零逻辑是否造成轨迹同质化。
    - 检查采样温度/top-k/top-p 在当前模型熵下是否仍过于确定性。
    - 如确认组内长期同质化，考虑引入更强多样化采样策略（可配置，不破坏基线复现）。
- [x] 新增“实际生成 action 数”监控口径（替代标签口径）：
  - 明确区分：
    - `train_action_token_count` 仅表示训练标签中的 action token 数（监督信号），不代表模型生成结果。
    - 目标监控应统计模型自由生成中的 action token 个数（如 `sample_action_tokens_len`）。
  - 已完成（2026-03-11）：
    - 在 `SFTAutoVLA.training_step` 增加可配置 free-run 生成探针（默认关闭，不影响现有训练速度）。
    - 新增日志：
      - `probe_gen_action_candidate_count`
      - `probe_gen_action_tokens_len`
      - `probe_gen_has_action`
      - `probe_gen_completion_len`
      - `probe_gen_prompt_len`
      - `probe_gen_error`
    - 配置入口（SFT）：
      - `training.generated_action_probe.enabled`
      - `training.generated_action_probe.every_n_steps`
      - `training.generated_action_probe.max_new_tokens`
      - `training.generated_action_probe.do_sample/temperature/top_k/top_p`
    - 当前训练配置状态：
      - `config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml` 已将 `training.generated_action_probe.enabled` 设为 `true`（用于后续训练直接观测实际生成 action 个数）。
  - 同日补充统计（定位根因，未改训练逻辑）：
    - teacher-forcing 统计（SFT train 前 20 样本，动作位共 200）：
      - `pred_action_like=0/200`，`exact_match=0/200`
    - free-run 统计（RFT prompt 前 20 样本）：
      - greedy：`has_action_samples=0/20`，`valid_action_tokens=0`
      - sample(`temperature=1.0, top_p=1.0, top_k=0`)：`has_action_samples=0/20`，`valid_action_tokens=0`

### 阶段4切换计划（2026-03-09）
- 目标：RFT reward 从旧 PDMS 接口切换到 NavSim v2 的 EPDMS（先落地 stage1，不含 two-frame EC）。
- 本轮范围：
  - [x] 在 reward 代码中支持 `rl.reward.type`，默认 `epdms_stage1`，并预留 `epdms_full` 可选模式口子（暂不启用 two-frame EC）。
  - [x] `traffic_agents` 做成配置项，前期默认 `non_reactive`（log replay）用于 debug/基线对齐，后续可切 `reactive`（IDM）。
  - [x] 对齐 NavSim v2 `pdm_score` 新接口（含 `traffic_agents_policy` 入参与 DataFrame 返回），避免沿用旧版 `result.score`。
  - [x] 将 `model.sft_model_path` 更新为当前验证集最优 checkpoint。
- 当前确认口径（用户）：
  - `epdsm` = `EPDMS`；
  - 先用 stage1 EPDMS；
  - 默认 `traffic_agents=non_reactive`；
  - SFT 初始化权重使用当前验证集最优（`epoch=4-loss=0.6140.ckpt`）。

### 阶段4冒烟验证（2026-03-09）
- `run_rft.sh` 环境修正：
  - 问题：默认 `python` 指向系统 Python 3.13，缺少 `peft` 导致启动失败。
  - 修复：`scripts/run_rft.sh` 新增 `CONDA_ENV/PYTHON_BIN`，默认使用 `autolsqv2` 的 Python。
- 启动冒烟（180s 限时）：
  - 命令：`timeout 180s bash scripts/run_rft.sh`
  - 结果：成功进入 2 卡分布式训练并开始 `Epoch 0`（超时主动终止，`EXIT_CODE=124`）。
  - 关键迹象：`distributed_backend=nccl`、`GLOBAL_RANK 0/1`、`Training: 0/103288`。
- reward 函数级验证（单样本）：
  - token：`0000be0b1dc65be3`
  - `epdms_stage1 + non_reactive`：`SCORE=1.0`
  - `epdms_stage1 + reactive`：`SCORE_REACTIVE=1.0`
  - 结论：v2 stage1 EPDMS reward 路径与 traffic_agents 可配置分支均可用。

### 阶段4启动记录（2026-03-09，单机8卡）
- 启动方式（后台）：
  - `setsid nohup bash scripts/run_rft.sh > logs/train/rft_navsimv2_single8_2026-03-09_11-29-15.log 2>&1 < /dev/null &`
- 运行标识：
  - run tag：`rft_navsimv2_single8_2026-03-09_11-29-15`
  - log：`logs/train/rft_navsimv2_single8_2026-03-09_11-29-15.log`
  - launcher pid：`4186143`
  - trainer pid（main）：`4186145`
- 启动确认：
  - 日志已出现 `All distributed processes registered. Starting with 8 processes` 与 `Epoch 0: 0/103288`。
  - GPU 快照（启动后）：8 卡均已挂载并有显存/算力占用（卡 0~7）。

## 相机映射策略（本次决策）
- 保持现状：`dataset_name=nuplan` 优先走 `left/right_camera_paths`。
- 新增安全回退：若 `left/right` 缺失，自动回退到 `front_left/front_right`，避免不同预处理版本字段差异导致中断。

### 阶段4速度排查与修复（2026-03-09 11:55 UTC）
- 现象：
  - 用户反馈“单机 8 卡看起来很慢”。
  - 复盘发现先前 `fast` run 实际已在首个训练 step 处退出，而非持续慢跑。
- 根因（已定位）：
  - `training.sample.max_length=1024`，但真实 `input_ids` 可达 `1209`，触发 `transformers.generate` 的长度校验异常并退出。
  - 典型报错：`Input length of input_ids is 1209, but max_length is set to 1024`。
- 修复（已落地）：
  - 代码：`models/autovla.py`
    - 训练/推理 `generate` 新增可选 `max_new_tokens`（优先于 `max_length`），避免再次因输入长度超过上限直接报错。
    - 推理侧 `max_length` 支持回退到 `inference.max_length`。
  - 配置：`config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast.yaml`
    - `training.sample.max_length: 2048`
    - `training.sample.max_new_tokens: 256`
    - `inference.max_length: 2048`
    - `inference.sample.max_new_tokens: 256`
- 重启结果（当前进行中）：
  - 8 进程分布式正常：`All distributed processes registered. Starting with 8 processes`
  - 已进入训练并持续推进：`Epoch 0: 6/20000, 0.07 it/s`（约 `14~15s/step`）。
  - GPU 快照：8 张卡均有约 `32GB` 显存占用，利用率约 `36~38%`，确认为真 8 卡训练。
- ETA（按当前速率估算）：
  - 按 `max_steps=4000`：约 `15.5~16.5h`
  - 若跑满 `20000` step：约 `79~83h`（约 `3.3` 天）

### 阶段4当前启动命令与日志摘要（2026-03-09 12:21 UTC）
- 当前有效启动命令（单机 8 卡，fast 配置）：
```bash
cd /data/liushiqi/AutoVLA && \
env CONDA_ENV=autolsqv2 \
    RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast \
    GPU_LIST=0,1,2,3,4,5,6,7 \
    bash scripts/run_rft.sh
```
- 当前运行方式：
  - 以交互持久会话运行（session_id=`11093`），实时日志来自该会话 stdout（非 nohup 文件落盘）。
- 当前进程（2026-03-09 12:21 UTC 快照）：
  - launcher: `13378`（`bash scripts/run_rft.sh`）
  - main: `13379`（`python tools/run_rft.py --config ...cot-fast`）
  - workers: `14050, 14051, 14179, 14307, 14435, 14636, 14828`
- 当前日志关键行（会话实时输出）：
  - `Epoch 0: 79/20000 ... 0.07it/s`
  - `Epoch 0: 95/20000 ... 0.07it/s`
  - 说明：训练稳定推进中，step 时间约 `14~15s/step`。
- 当前 GPU 快照（2026-03-09 12:21 UTC）：
  - 8 卡均在工作，利用率约 `33%~42%`，显存约 `32~35GB`/卡（符合 8 卡并行训练预期）。

### 阶段4后台落盘运行（2026-03-09 12:41 UTC）
- 背景：
  - 前一版是交互会话运行（有实时输出但不写新 log 文件），已停止并切换为后台落盘模式。
- 后台启动方式（当前生效）：
  - 启动脚本：`/tmp/run_rft_bg_2026-03-09_12-41-34.sh`
  - 脚本内容核心：`cd /data/liushiqi/AutoVLA` 后执行
    `env CONDA_ENV=autolsqv2 RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_rft.sh`
    并重定向到日志文件。
  - 脱离命令：`setsid -f /bin/bash /tmp/run_rft_bg_2026-03-09_12-41-34.sh`
- 当前日志文件：
  - `logs/train/rft_navsimv2_single8_fastfix_bg_2026-03-09_12-41-34.log`
- 当前进程快照（后台）：
  - launcher：`30860`（`bash scripts/run_rft.sh`）
  - main：`30861`（`python tools/run_rft.py --config ...cot-fast`）
  - workers：`31575, 31577, 31705, 31844, 31972, 32100, 32301`
- 日志关键迹象：
  - `All distributed processes registered. Starting with 8 processes`
  - `Epoch 0: 1/20000 ... 0.06it/s`（训练已开始推进）
- 监控命令：
  - `tail -f /data/liushiqi/AutoVLA/logs/train/rft_navsimv2_single8_fastfix_bg_2026-03-09_12-41-34.log`
  - `pgrep -af "run_rft.py --config training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast|bash scripts/run_rft.sh"`

## 论文设置 vs 当前RL设置对比（2026-03-10）

> 对比基准：
> - 论文：`task/2506.13757v3.pdf`（AutoVLA）
> - 当前代码口径：`config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast.yaml` + `tools/run_rft.py` + `models/autovla.py`

| 项目 | 论文（AutoVLA） | 当前（fast配置） | 对比结论 |
|---|---|---|---|
| RL算法 | GRPO | GRPO | 一致 |
| 参考策略 | SFT policy作为reference | `sft_model_path`加载reference model | 一致 |
| LoRA | rank=8, alpha=8, dropout=0.1 | `r=8, alpha=8, dropout=0.1` | 一致 |
| RFT学习率 | `3e-5` | `3e-5` | 一致 |
| KL系数 | `β=0.04` | `kl_beta=0.04` | 一致 |
| 更新形式 | single policy update / 无clip旧策略追踪 | 单步采样后直接算策略损失+KL（无ratio clipping） | 一致（实现形态等价） |
| CoT惩罚 | `r = r_driving - λ_r * r_cot`，附录给出 `γ=2e-3, Ltol=400, λ_r=0.3` | `coef=0.002, center=400, weight=0.3` | 基本一致（参数一一对应） |
| 训练步数（RFT） | `6000 steps` | `max_steps=6000` | 一致 |
| epoch设置 | 论文RFT主口径按step描述 | `epochs=1`（并受`max_steps`截断） | 口径不同（你方以step为主） |
| 采样温度（RFT） | 文中RFT实现细节给出 `temp=1.0, top-p=1.0, top-k=0.0` | `temp=1.0, top-p=1.0, top-k=0.0` | 一致 |
| 轨迹动作token | 每样本输出10个action token（5s, 0.5s间隔） | `num_poses=10`，不足补零、超出截断到10 | 一致 |
| 训练设备组大小G | 算法记为group size `G`（文中未固定写死） | 当前单机8卡；GroupSampler使各卡同索引采样，等效每step组内样本数≈8 | 当前可视作 `G≈8` |
| 奖励定义（nuPlan/Waymo） | nuPlan偏PDMS，Waymo偏ADE（因RFS稀缺） | 当前NavSim v2用`epdms_stage1`（`scale=10`，默认`non_reactive`） | 不同（奖励实现已工程化替换） |

### token/step/epoch 量化口径（当前fast配置）

- 基础配置：
  - `epochs=1`
  - `max_steps=6000`
  - `devices=8`（单机8卡）
  - 每样本动作token数：`10`
  - 生成上限：`max_new_tokens=256`
- 推导：
  - `action token sum`（按每step每卡1样本估算）：
    - `6000 * 8 * 10 = 480,000`
- `completion token upper bound`（按`max_new_tokens`上界估算）：
  - `6000 * 8 * 256 = 12,288,000`

## NavSim v2 统一采样口径（2026-03-16）

- 适用范围：
  - `NavSim v2 RL` 相关训练配置
  - `NavSim v2 SFT` 相关训练配置
  - `NavSim v2 AutoVLA` 相关评测配置
- 统一规则：
  - 训练采样固定为：
    - `do_sample=true`
    - `temperature=1.0`
    - `top_p=1.0`
    - `top_k=0`
  - 评测采样固定为：
    - `temperature=0.2`
    - `top_p=1.0`
    - `top_k=20`
- 本次已更新的主配置：
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot.yaml`
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast.yaml`
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-sft20260312e4.yaml`
  - `/data/liushiqi/AutoVLA/config/eval/qwen2.5-vl-3B-navsimv2-autovla-epdms-conservative-model.yaml`
- 说明：
  - `SFT generated_action_probe` 也对齐到训练采样口径，避免训练中“实际生成 action 数”统计和训练采样策略不一致。
  - 已经启动中的训练/评测进程不会自动继承这些改动；需要重启后才会生效。

### 历史日志口径说明（避免混淆）

- 本文档早前运行记录中出现过 `Epoch 0: .../20000` 与 `max_steps=4000/20000` 的ETA估算（2026-03-09）。
- 以当前仓库配置文件（2026-03-10）为准：`qwen2.5-vl-3B-navsimv2-grpo-cot-fast.yaml` 当前值为 `max_steps=6000`。

## 阶段3+1+8 联合修复与复验（2026-03-10）

### 目标
- `3`：SFT/RL 提示模板对齐（去掉不稳定强制前缀的默认行为）
- `1`：SFT action loss 从“仅 has_cot”改为“全样本生效”
- `8`：补齐训练健康监控（特别是 action token 产出）

### 代码改动（已落地）
- `models/autovla.py`
  - `AutoVLA.get_prompt`：`model.prompt.force_action_answer_prefix` 默认值改为 `false`（仍保留可选开关）。
  - `SFTAutoVLA.training_step`：
    - action loss 对所有样本生效，不再仅在 `has_cot=True` 才参与总损失；
    - 新增配置项：
      - `training.action_loss_weight`（当前配置为 `2.0`）
      - `training.cot_sample_loss_multiplier`（默认 `1.0`，保留旧逻辑开关）
    - 新增监控日志：
      - `train_base_loss`
      - `train_action_loss`
      - `train_action_loss_weight`
      - `train_action_token_count`
      - `train_valid_token_count`
      - `train_action_token_ratio`
      - `train_has_cot_ratio`
      - `train_cot_loss_multiplier`
  - `GRPOAutoVLA.training_step`：
    - 新增 action 产出健康度日志：
      - `sample_action_candidate_count`
      - `sample_action_tokens_len`
      - `sample_action_nonzero_count`
      - `sample_has_action`
      - `sample_completion_len`
      - `sample_prompt_len`
- `tools/run_sft.py`
  - 修复 8 卡分布式启动 bug：`devices=auto` 在子进程中被错误 `int()` 转换导致崩溃。
  - 新增 `resolve_trainer_devices()`，兼容 `auto / int / list`，并在 torchrun/子进程环境下安全解析设备数。
- 配置：
  - `config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml`
    - 新增：
      - `model.prompt.force_action_answer_prefix: false`
      - `training.action_loss_weight: 2.0`
      - `training.cot_sample_loss_multiplier: 1.0`
  - `config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast.yaml`
    - 新增：
      - `model.prompt.force_action_answer_prefix: false`
  - `config/training/qwen2.5-vl-3B-navsimv2-grpo-cot.yaml`
    - 新增：
      - `model.prompt.force_action_answer_prefix: false`

### 快速效果测试（E0，2卡3步）
- 命令：
```bash
cd /data/liushiqi/AutoVLA
env CONDA_ENV=autolsqv2 \
    RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-compare-e0 \
    GPU_LIST=0,1 \
    bash scripts/run_rft.sh 2>&1 | tee logs/train/rft_compare_e0_after_318.log
```
- 结果（通过）：
  - 成功跑满 `max_steps=3`；
  - 监控指标已落盘（见 `runs/lightning_logs/version_32/metrics.csv`，含 `sample_action_*`、`group_reward_std`、`train_advantage`）。
- 现象（仍待根治）：
  - 组内 `action_tokens_len=0`、`sample_has_action=0`，reward 仍同质化；
  - 说明“监控链路已打通”，但“动作不吐出”仍需依赖本次完整 SFT 重训后再验证。

### 完整训练启动（按你要求改为 tmux 后台）
- 首次 tmux 启动失败根因：
  - `tools/run_sft.py` 的 `devices=auto` 解析 bug（已修复，见上）。
- 当前有效启动方式：
  - session：`sft_3p1p8_091856`
  - 脚本：`/tmp/run_sft_tmux_2026-03-10_09-18-56.sh`
  - 主日志：`/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_3p1p8_2026-03-10_09-18-56.log`
  - run 目录：`/data/liushiqi/AutoVLA/runs/sft/2026-03-10_09-18-56`
- tmux 后台启动命令（等价）：
```bash
tmux new-session -d -s sft_3p1p8_091856 "/bin/bash /tmp/run_sft_tmux_2026-03-10_09-18-56.sh"
```
- 当前状态（2026-03-10）：
  - 已完成 8 卡分布式初始化：`All distributed processes registered. Starting with 8 processes`
  - 已进入训练：`Epoch 0: 0/12911`
  - GPU 显存已挂载（约 `18GB+`/卡，8 卡均有占用）。

### 运行中监控命令
```bash
tmux ls | rg sft_3p1p8_091856
tmux capture-pane -pt sft_3p1p8_091856:0 | tail -n 80
tail -n 80 /data/liushiqi/AutoVLA/logs/train/sft_navsimv2_3p1p8_2026-03-10_09-18-56.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
tensorboard --logdir /data/liushiqi/AutoVLA/runs/sft/2026-03-10_09-18-56 --port 6006 --bind_all
```

### 用户指令：先测 epoch1，再重启（2026-03-10）
- 指令执行顺序：
  1. 全停当前 SFT 残留进程（避免半挂起占卡）。
  2. 用本次 run 的 `epoch=1` ckpt 做 2 卡 E0 动作输出复验。
  3. 完整 SFT 重新启动（8卡，tmux 后台）。

- epoch1 ckpt：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-10_09-18-56/epoch=1-loss=0.6192.ckpt`

- E0 测试配置与命令：
  - 配置：`config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-compare-e0-epoch1sft.yaml`
  - 命令：
```bash
cd /data/liushiqi/AutoVLA
env CONDA_ENV=autolsqv2 \
    RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-compare-e0-epoch1sft \
    GPU_LIST=0,1 \
    bash scripts/run_rft.sh 2>&1 | tee logs/train/rft_compare_e0_epoch1sft_2026-03-10_13-10-05.log
```
- E0 结果：
  - 3 step 正常结束（`max_steps=3`）；
  - 组内依旧 `action_tokens_len=0`、`action_nonzero_count=0`、`sample_has_action=0`；
  - 结论：仅到 SFT 第 1 个 epoch，仍未恢复稳定动作 token 输出。

- 重新启动完整 SFT（当前运行中）：
  - session：`sft_restart_131247`
  - 启动脚本：`/tmp/run_sft_tmux_restart_2026-03-10_13-12-47.sh`
  - 日志：`/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_restart_2026-03-10_13-12-47.log`
  - run 目录：`/data/liushiqi/AutoVLA/runs/sft/2026-03-10_13-12-47`
  - 当前状态：8 卡分布式已完成初始化并进入 `Epoch 0` 训练。

### SFT 生成动作探针报错修正（2026-03-12）
- 现象：
  - 训练主流程持续进行，但日志反复出现：
    - `[probe_gen] failed at step=0: RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280`
  - TensorBoard 中未写出 `probe_gen_action_tokens_len` 等生成动作指标。
- 根因：
  - 探针代码对视觉张量统一执行了 `[:1]` 切片；
  - 但 `pixel_values_videos` 与 `video_grid_thw` 在 Qwen2.5-VL 里首维不是 batch 维（例如 `(2880,1176)`、`(3,3)`），被错误切片后与文本视觉占位不匹配，触发 `generate` 内部 shape 异常。
- 修正：
  - 文件：`models/autovla.py`
  - 仅对“首维等于 batch_size”的张量执行 `[:1]`，否则保持原样；
  - 增加 `self._probe_generated_action_last_step`，避免梯度累积下同一 `global_step` 重复触发 probe。
- 验证（静态 + 形状）：
  - `python -m compileall models/autovla.py` 通过；
  - 1 batch 诊断下，probe 输入形状为：
    - `input_ids (1,1008)`, `attention_mask (1,1008)`
    - `pixel_values_videos (2880,1176)`, `video_grid_thw (3,3)`（不再错误切片）。
- 注意：
  - 当前已在跑的 SFT 进程是旧代码，需重启后该修正才会生效并产出 `probe_gen_*` 指标。

### 待办（2026-03-12）
- [x] 等当前 run 的 `epoch 0` 训练完成后，先做一次动作输出测试（优先看 `probe_gen_has_action`、`probe_gen_action_tokens_len`）。
  - 测试时间：`2026-03-12 05:28 UTC`
  - run：`runs/sft/2026-03-12_03-39-30`（已产出 `epoch=0-loss=0.6873.ckpt`）
  - 结果（epoch0 区间）：
    - `probe_gen_has_action`: `mean=0.875`，后段连续为 `1.0`，自 step `449` 后保持全 1，末尾 `step=3199 -> 1.0`
    - `probe_gen_action_tokens_len`: `mean=8.75`，末尾连续 `10`（如 `2999/3049/3099/3149/3199 -> 10`）
    - `probe_gen_action_candidate_count`: 与上同，末尾连续 `10`
- [x] 仅在上述测试确认后，再切换到最新对齐设置（`use_cot=false` + 简化 `AutoVLA.get_prompt`）。
  - 已完成：
    - `model.use_cot: false`
    - `training.log_every_n_steps: 200`
    - `training.enable_progress_bar: true`
    - `training.progress_bar_refresh_rate: 100`
  - 当前有效 run：`runs/sft/2026-03-12_06-03-13`

### SFT 最新启动口径（2026-03-12）
- 启动命令（当前有效）：
```bash
cd /data/liushiqi/AutoVLA
export NAVSIM_DEVKIT_ROOT=/data/liushiqi/navsim \
       NAVSIM_DATA_ROOT=/data/dataset/navsim \
       NUPLAN_DATA_ROOT=/data/dataset/navsim \
       NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
       NAVSIM_EXP_ROOT=/data/dataset/navsim \
       OPENSCENE_DATA_ROOT=/data/dataset/navsim \
       NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
       PYTHONPATH=/data/liushiqi/navsim:${PYTHONPATH:-} \
       SFT_RUN_TS=2026-03-12_06-03-13
/data/miniconda/envs/autolsqv2/bin/python tools/run_sft.py \
  --config training/qwen2.5-vl-3B-navsimv2-mix-sft
```
- 后台会话与日志：
  - session：`sft_navsimv2_align_2026-03-12_06-03-13`
  - log：`/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_align_2026-03-12_06-03-13.log`
  - run：`/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13`
- 当前保存进度（按 event 最新）：
  - `global_step=199`（`epoch=0`）
  - `train_loss=23.6669`
  - `train_base_loss=3.2488`
  - `train_action_loss=10.2091`
  - `probe_gen_has_action=0.0`
  - `probe_gen_action_tokens_len=0.0`

### SFT 清理记录（2026-03-12）
- 已删除中间尝试（无 ckpt）：
  - run 目录：
    - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_02-24-59`
    - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_02-45-35`
    - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_05-33-11`
    - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_05-55-27`
  - 日志文件：
    - `/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_probe_fix_2026-03-12_02-45-35.log`
    - `/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_align_2026-03-12_05-33-11.log`
    - `/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_align_2026-03-12_05-55-27.log`
- 保留：
  - 历史有效 run（有 ckpt）：`/data/liushiqi/AutoVLA/runs/sft/2026-03-12_03-39-30`
  - 当前运行中：`/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13`
  - 当前日志：`/data/liushiqi/AutoVLA/logs/train/sft_navsimv2_align_2026-03-12_06-03-13.log`

### GRPO 全量 8GPU 运行补充（2026-03-12）
- 对应日志：
  - `/data/liushiqi/AutoVLA/logs/grpo/rft_full_8gpu_meanmetrics_2026-03-12_03-17-56.log`
- 实际启动口径（在 `10.199.7.33` 上）：
```bash
cd /data/liushiqi/AutoVLA_main_rl_v1
mkdir -p /data/liushiqi/AutoVLA/logs/grpo
RUN_TAG=rft_full_8gpu_meanmetrics_$(date -u +%F_%H-%M-%S)
LOG_PATH=/data/liushiqi/AutoVLA/logs/grpo/${RUN_TAG}.log
nohup env \
  NAVSIM_DEVKIT_ROOT=/data/liushiqi/AutoVLA_main_rl_v1/navsim \
  NAVSIM_DATA_ROOT=/data/dataset/navsim \
  NUPLAN_DATA_ROOT=/data/dataset/navsim \
  NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
  NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs \
  OPENSCENE_DATA_ROOT=/data/dataset/navsim \
  NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
  PYTHONPATH=/data/liushiqi/AutoVLA_main_rl_v1/navsim:/data/liushiqi/AutoVLA_main_rl_v1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /data/miniconda/envs/autovla_codeclean/bin/python tools/run_rft.py \
  --config training/qwen2.5-vl-3B-nuplan-grpo-cot-8gpu \
  > ${LOG_PATH} 2>&1 < /dev/null &
```
- 当前保存频率：
  - `every_n_train_steps=500`（按 step 保存，不在 epoch 末额外保存）
- 当前 ckpt 保存进度：
  - 目录：`/data/liushiqi/AutoVLA_main_rl_v1/runs/grpo/2026-03-12_03-18-36`
  - 已保存：
    - `rft-step500-reward5.9688.ckpt`
    - `rft-step1000-reward6.8125.ckpt`
    - `rft-step1500-reward6.6562.ckpt`
- 当前训练推进（日志快照）：
  - 数据分母：`0/103288`
  - 已观察到步数：`step=1729`（日志持续增长中）
- 与论文对齐状态（新增记录）：
  - 当前这条 8GPU run **未**使用论文对齐的探索参数。
  - 实际生效配置：`/data/liushiqi/AutoVLA_main_rl_v1/config/training/qwen2.5-vl-3B-nuplan-grpo-cot-8gpu.yaml`
    - `temperature=0.2`，`top_p=1.0`，`top_k=0.0`
    - 未显式设置 `max_steps=6000`（非论文 step-budget 口径）
  - 论文对齐参数已在另一路配置中完成（`/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot.yaml`，`temperature=1.0, top_p=1.0, top_k=0.0, max_steps=6000`），但本次 run 未引用该文件。
- 运行产物保存规范（v2.2，待新 run 生效）：
  - 统一 `run_id` 命名：`grpo_<time>_t<temp>_p<top_p>_k<top_k>_ms<max_steps|NA>`
  - ckpt/csv 放在：`runs/grpo/<run_id>/{ckpt,csv}/`
  - TensorBoard 独立目录：`tensorboard/grpo/<run_id>/`
  - 元数据：`runs/grpo/<run_id>/run_manifest.yaml`
- v2.2 冒烟验证（2026-03-12 07:06 UTC）：
  - 配置：`training/qwen2.5-vl-3B-navsimv2-grpo-cot-pathsmoke`（`max_steps=1`，单卡）
  - 运行结束状态：`max_steps=1 reached`（成功）
  - `run_id`：`grpo_2026-03-12_07-06-32_t1.0_p1.0_k0.0_ms1`
  - 实际落盘：
    - run root：`/data/liushiqi/AutoVLA/runs/grpo/grpo_2026-03-12_07-06-32_t1.0_p1.0_k0.0_ms1`
    - csv：`.../csv/metrics.csv`
    - tb：`/data/liushiqi/AutoVLA/tensorboard/grpo/grpo_2026-03-12_07-06-32_t1.0_p1.0_k0.0_ms1/events.out.tfevents.*`
    - manifest：`.../run_manifest.yaml`
  - 说明：`ckpt/` 为空属于预期（当前保存策略 `every_n_train_steps=500`，而冒烟仅 1 step）。

### Navtest PDMS 8卡并行评测（SFT最终ckpt，2026-03-13）
- 目标：
  - 对 `SFT final ckpt` 在 `navtest_nocot` 上做并行 PDMS 评测，加快全量 12146 token 评测。
- 评测 checkpoint：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
  - 软链（规避路径里 `=` 对 CLI 解析的干扰）：
    - `/tmp/sft_epoch4_loss09352.ckpt`
- 数据路径：
  - `metric_cache_path=/data/dataset/navsim/metric_cache/navtest`
  - `json_data_path=/data/dataset/navsim/preprocessed/navtest_nocot`
- 分片与会话：
  - 分片 token 文件：`/tmp/navtest_sft_eval_shards/shard_0.txt ... shard_7.txt`
  - tmux sessions：
    - `navpdm_20260313_035701_s0 ... navpdm_20260313_035701_s7`
- 输出目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navtest_sft_epoch4_pdms_tmux8_20260313_035701`
  - 每分片日志：
    - `.../shard_{0..7}.log`
    - `.../shard_{0..7}_csv/run_pdm_score_cot.log`
- 启动口径（单分片模板）：
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> /data/miniconda/envs/autolsqv2/bin/python \
  /data/liushiqi/AutoVLA/navsim/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=navtest \
  train_test_split.scene_filter.tokens=[...] \
  train_test_split.scene_filter.max_scenes=null \
  agent=autovla_agent \
  +agent.config_path=/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml \
  +agent.checkpoint_path=/tmp/sft_epoch4_loss09352.ckpt \
  +agent.sensor_data_path=/data/dataset/navsim \
  metric_cache_path=/data/dataset/navsim/metric_cache/navtest \
  json_data_path=/data/dataset/navsim/preprocessed/navtest_nocot \
  output_dir=/data/liushiqi/AutoVLA/logs/eval/navtest_sft_epoch4_pdms_tmux8_20260313_035701/shard_<i>_csv \
  experiment_name=navtest_sft_epoch4_pdms_tmux8_shard_<i>
```
- 运行中状态（2026-03-13 04:33 UTC 快照）：
  - 8 个 shard 进程均存活（每 shard 对应 1 个 `run_pdm_score_cot.py` 进程）。
  - 所有分片日志均在推进，示例：`Processing scenario 379 / 1514`。
  - 错误扫描：`ERROR/Traceback/Exception` 均未发现。
- ETA（按日志实时速率估算）：
  - 平均约 `5.42~5.47 sec/scenario/shard`。
  - 各 shard 预计完成时间约 `2026-03-13 06:15~06:16 UTC`。

### Navhard 修复数据统一落盘（2026-03-13）
- 目的：
  - 避免“修复后的 json 散落在 `/tmp` 或临时目录”导致后续评测/复现实验路径不稳定。
- 根因定位（本次缺失场景）：
  - `metric_cache_v2/navhard...` 中大量 token 为 17 字符“合成 token”，无法直接在原始预处理 json 目录中按文件名命中。
  - 需先做 token 映射，再回填可用 json。
- 最终落盘目录（唯一口径）：
  - `/data/dataset/navsim/preprocessed/navhard_nocot_full_2026-03-13`
  - 样本数：`5912`（与对应 `metric_cache.pkl` token 数一致）
- 修复策略（已执行）：
  - 通过 `(log_name, time_us)` 将 17 字符合成 token 映射回真实日志帧 token；
  - 对无法直接加载的 3 个尾帧 source token，采用最近邻可用帧回退：
    - `fc2a258a2dc153a8 -> f34ebf09b5515de1`
    - `c1e88ed6895053e8 -> c32ae6a1954651ec`
    - `dc376531664b51b8 -> c32ae6a1954651ec`
- 产物规范（后续固定）：
  - 最终“可复用数据”只放 `dataset`：`/data/dataset/navsim/preprocessed/...`
  - `/tmp` 仅允许中间构建与调试，完成后清理，不作为评测输入源。

### Navhard EPDMS 单卡/8卡一致性验证（2026-03-15）
- 目标：
  - 在当前修复后的 `navhard_two_stage` 评测链路上，验证 fresh 单卡 full 与 8 卡分片 full 的最终指标是否一致，并记录运行时间差异。
- 评测 checkpoint：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 8卡 full 运行：
  - 机器：`root@10.199.7.33:2289`
  - 输出目录：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47`
  - 总结文件：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47/merged/summary.json`
  - 启动命令：
```bash
ssh -p 2289 root@10.199.7.33 "
cd /data/liushiqi/AutoVLA &&
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh
"
```
- fresh 单卡 full 运行：
  - 机器：`root@10.199.7.33:2289`
  - 输出目录：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56`
  - run dir：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/run_2026-03-15_07-15-08`
  - 总结文件：
    - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/run_2026-03-15_07-15-08/summary.json`
  - 实际启动命令：
```bash
ssh -p 2289 root@10.199.7.33 "
cd /data/liushiqi/AutoVLA &&
mkdir -p /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56 &&
cat > /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launch.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd /data/liushiqi/AutoVLA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla.py \
  --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml \
  --set model.checkpoint_path=/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt \
--set eval.output_dir=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56 \
  2>&1 | tee /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launcher_stdout.log
EOS
chmod +x /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launch.sh &&
nohup bash /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launch.sh \
  > /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/nohup_wrapper.log 2>&1 &
"
```
- 最终结果对比：
  - 8卡：
    - `num_successful_scenarios = 5912`
    - `num_failed_scenarios = 0`
    - `final_extended_pdm_score = 0.14937031924317998`
    - 最慢 shard wall time：约 `2168s`（`~36.1 min`）
  - fresh 单卡：
    - `num_successful_scenarios = 5912`
    - `num_failed_scenarios = 0`
    - `final_extended_pdm_score = 0.14937031924317998`
    - `elapsed_sec = 16216.207909345627`（`~4.50h`）
  - 一致性结论：
    - `score_diff = 0.0`
    - 当前修复后的单卡与 8 卡分片合并结果完全一致，说明并行评测链路未改变指标语义。
  - 速度结论：
    - 8卡相对 fresh 单卡约 `7.5x` 加速。

### SFT 最终 ckpt 动作输出稳定性复核（2026-03-16）
- 目标 ckpt：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 证据口径：
  - 使用该 ckpt 对应训练 run 的 TensorBoard event，读取 `probe_gen_*` 指标。
  - 注意：`generated_action_probe` 不是“固定同一条测试 prompt 重复采样”，而是训练期间每 `400` step 对“当前 batch 第一个样本”做一次 free-run 生成。
  - event 文件：
    - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/lightning_logs/version_0/events.out.tfevents.1773295460.ubuntu.905049.0`
- 结果：
  - `probe_gen_has_action`：
    - `count=72`
    - `mean=0.9722`
    - `first10=[0,0,1,1,1,1,1,1,1,1]`
    - `last10=[1,1,1,1,1,1,1,1,1,1]`
  - `probe_gen_action_tokens_len`：
    - `count=72`
    - `mean=9.7222`
    - `first10=[0,0,10,10,10,10,10,10,10,10]`
    - `last10=[10,10,10,10,10,10,10,10,10,10]`
  - `probe_gen_action_candidate_count`：
    - `count=72`
    - `mean=9.7222`
    - `first10=[0,0,10,10,10,10,10,10,10,10]`
    - `last10=[10,10,10,10,10,10,10,10,10,10]`
- 当前结论：
  - 对这个最终 ckpt 来说，按训练期 free-run probe 口径，后期已经能非常稳定地产生动作：
    - 后 10 次 probe 全部 `has_action=1`
    - 后 10 次 probe 全部输出 `10` 个 action token
  - 因此“这个 ckpt 完全不会输出动作”这个结论不成立。
  - 更准确的说法是：
    - 它在 probe 口径下已经基本稳定输出动作；
    - 但 RL / 某些评测 prompt 下是否仍有 prompt-sensitive 退化，需要继续区分排查。

### AutoVLA 标准口径 full navtest 结果补录（2026-03-15 完成，2026-03-16补记）
- 配置：
  - `config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml`
- 评测 checkpoint：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 输出目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190`
- summary：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/summary.json`
- csv：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/2026.03.15.14.03.35.csv`
- 最终结果：
  - `successful=12146`
  - `failed=0`
  - `invalid_sum=0`
  - `score_mean=0.5944498471141966`
- 结论：
  - AutoVLA 标准口径 full `navtest` 已完成，不再是“运行中”状态。
  - 当前该 SFT 最终 ckpt 在标准口径 full `navtest` 上，结果显著高于同目录中先前的 baseline `constant_velocity_agent` 结果（`0.3233531830113996`）。

### NavSim v2 RL 统一采样口径后重启记录（2026-03-16）
- 目的：
  - 使当前 NavSim v2 RL run 与新统一口径一致：
    - 训练：`do_sample=true, temperature=1.0, top_p=1.0, top_k=0`
    - 推理/评测：`temperature=0.2, top_p=1.0, top_k=20`
- 使用配置：
  - `/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-sft20260312e4.yaml`
- 停止前旧 run 状态：
  - tmux：`rl_navsimv2_sft20260312e4_ip190`
  - 旧日志：
    - `/data/liushiqi/AutoVLA/logs/train/rl_navsimv2_sft20260312e4_ip190_2026-03-16_10-02-41.log`
  - 停止前进度：
    - `step ~3327/12000`
  - 停止前健康状态摘要：
    - `sample_action_tokens_len` 持续非零，约 `8.6~10.0`
    - `group_reward_std` 大多数 step 非零
    - `group_adv_fallback` 大多数 step 为 `0.0`
- 新 run 启动信息：
  - 节点：
    - `10.199.7.190`
  - tmux：
    - `rl_navsimv2_sft20260312e4_ip190`
  - 新 `RUN_ID`：
    - `grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06`
  - 新日志：
    - `/data/liushiqi/AutoVLA/logs/train/rl_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06.log`
  - 启动命令：
```bash
ssh -p 2289 root@10.199.7.190 '
cd /data/liushiqi/AutoVLA &&
TS=2026-03-16_13-23-06 &&
SESSION=rl_navsimv2_sft20260312e4_ip190 &&
LOG=/data/liushiqi/AutoVLA/logs/train/${SESSION}_${TS}.log &&
tmux new-session -d -s ${SESSION} "
cd /data/liushiqi/AutoVLA &&
env CONDA_ENV=autolsqv2 \
PYTHON_BIN=/data/miniconda/envs/autolsqv2/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
RFT_CONFIG=training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-sft20260312e4 \
RFT_RUN_ID=grpo_navsimv2_sft20260312e4_ip190_${TS} \
stdbuf -oL -eL bash scripts/run_rft.sh 2>&1 | tee -a ${LOG}
"
'
```
- 新 run 前几步观测：
  - `step 1`：
    - `sample_action_tokens_len=9.880`
    - `group_reward_std=1.000`
    - `group_adv_fallback=0.000`
  - `step 5`：
    - `sample_action_tokens_len=10.00`
    - `group_reward_std=0.604`
    - `group_adv_fallback=0.000`
  - `step 8`：
    - `sample_action_tokens_len=7.620`
    - `group_reward_std=0.000`
    - `group_adv_fallback=1.000`
    - `loss=-10.0`
    - `policy_loss=-10.0`
- 当前结论：
  - 新 run 已成功重启并进入训练循环，不是空跑。
  - `sample_action_tokens_len` 仍为非零，说明动作仍在生成。
  - 组内 reward 同质化现象仍会偶发出现，但不是持续全程发生。
