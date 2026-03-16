# Navsim v2 标准 EPDMS 本地模板任务

## 目标
- 在当前仓库新增“标准 v2 EPDMS 口径”本地模板入口，不覆盖现有 AutoVLA 定制评测链路。
- 明确并隔离两条评测口径：`标准口径（upstream）` vs `定制口径（AutoVLA）`。

## 问题清单（已确认）
1. 当前 `AutoVLA/navsim` 的默认 `run_pdm_score.py` / `pdm_score.py` 为定制实现，不能直接代表标准 v2 EPDMS。
2. `run_pdm_score_cot.py` 也复用当前仓库的定制 `pdm_score`，存在同样口径漂移风险。
3. 缺少“防混用”硬校验，历史上出现过 `score_proposals` 参数不匹配的混用错误。
4. 缺少 metric cache v2 schema 前置检查，旧 cache 会导致评测中途失败。
5. 标准入口与定制入口缺乏显式命名和操作指引，容易误用。

## 实施方案（双入口并存）
- 新增标准模板脚本：`tools/eval/run_navsimv2_epdms_standard.py`
  - 强制注入 upstream 路径：`/data/liushiqi/navsim`
  - 清理已加载的 `navsim*` 模块，避免进程内混用。
  - 评测前做 cache schema 抽样检查，缺字段直接 fail-fast。
  - 输出“口径指纹”：`navsim.__file__`、`run_pdm_score.py` 路径、`pdm_score` 签名、`score_proposals` 签名、`scorer/agent` 实例类型。
  - 对 `navtest` 这类不含 `reactive_all_mapping` 的 one-stage split，自动走 wrapper fallback 汇总，绕过 upstream `run_pdm_score.py` 的已知收尾 bug。
  - 支持 `--override k=v` 透传到 upstream hydra 入口；支持 `--dry-run`。
- 新增模板配置：`config/eval/navsimv2_epdms_standard.yaml`
  - 固定标准默认口径：`scorer=pdm_scorer` + `agent=constant_velocity_agent`
  - 默认指向 v2 cache：`/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49`
- 保留并不改动以下定制文件：
  - `navsim/navsim/planning/script/run_pdm_score.py`
  - `navsim/navsim/evaluate/pdm_score.py`

## 命令模板
- Dry-run（口径验证）：
  - `python tools/eval/run_navsimv2_epdms_standard.py --dry-run`
- Navtest 冒烟（20 场景）：
  - `python tools/eval/run_navsimv2_epdms_standard.py --override train_test_split.scene_filter.max_scenes=20 --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_smoke_20`
- Navtest 全量：
  - `python tools/eval/run_navsimv2_epdms_standard.py --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_full`
- Navhard two-stage（需要完整 v2 cache）：
  - `python tools/eval/run_navsimv2_epdms_standard.py --override train_test_split=navhard_two_stage --override metric_cache_path=<navhard_v2_cache> --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navhard_two_stage_full`

## 执行记录
- `2026-03-14`：初始化任务卡，落地标准模板脚本与 YAML。
- `2026-03-14`：T1 dry-run 通过。
  - 命令：`/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --dry-run`
  - 指纹结论：
    - `navsim.__file__=/data/liushiqi/navsim/navsim/__init__.py`
    - `run_pdm_score_file=/data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py`
    - `agent_type=navsim.agents.constant_velocity_agent.ConstantVelocityAgent`
    - `scorer_type=navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer.PDMScorer`
    - `has_reactive_all_mapping=false`
    - `Execution mode=wrapper_one_stage_fallback`
  - 指纹文件：`/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/fingerprint_2026-03-14_13-14-34.json`
- `2026-03-14`：定位到 upstream `run_pdm_score.py` 的 one-stage bug。
  - 直接跑 upstream `train_test_split=navtest` 时，会在收尾阶段访问不存在的 `train_test_split.reactive_all_mapping`，随后触发 `all_mappings` / `score_stage_one` 未定义错误。
  - 结论：标准模板不能简单 shell-out 到 upstream main；需要在 wrapper 内对 one-stage split 做本地汇总。
- `2026-03-14`：5 场景 smoke 通过。
  - 命令：`/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --override train_test_split.scene_filter.max_scenes=5 --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_smoke_5_2026-03-14_13-14-45`
  - 结果：`successful=5`，`failed=0`，`score_mean=0.7177973851622648`
  - CSV：`/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_smoke_5_2026-03-14_13-14-45/2026.03.14.13.15.33.csv`
- `2026-03-14`：20 场景 smoke 通过。
  - 命令：`/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --override train_test_split.scene_filter.max_scenes=20 --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_smoke_20_2026-03-14_13-15-53`
  - 结果：`successful=20`，`failed=0`，`score_mean=0.7672044916600478`
  - CSV：`/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_smoke_20_2026-03-14_13-15-53/2026.03.14.13.17.27.csv`
- `2026-03-14`：T2 通过。
  - 未出现 `score_proposals takes X but Y were given`
  - 失败根因已从“混用错误”转为“upstream one-stage 汇总 bug”，并已在 wrapper 内兼容处理。

## 验收矩阵
- T1 口径一致性：
  - `--dry-run` 输出 `navsim.__file__` 指向 `/data/liushiqi/navsim/...`
  - `scorer_type` 为 `PDMScorer`，`agent_type` 为 `ConstantVelocityAgent`
  - `pdm_score` 签名包含 `traffic_agents_policy`
- T2 标准冒烟（navtest 20）：
  - 成功产出 csv，无 `score_proposals takes X but Y were given`
- T3 标准全量（navtest）：
  - 产出平均 `score` 与 `valid` 统计
- T4 navhard two-stage（有 cache 时）：
  - stage1+stage2 可跑通，无 schema 报错
- T5 回归隔离：
  - `tools/eval/run_navhard_two_stage_autovla.py` 行为不受影响

## 当前验收状态（2026-03-16 更新）
- T1：
  - 已完成
- T2：
  - 已完成
- T3：
  - 已完成
  - baseline 标准口径 full `navtest` 已有结果：
    - `score_mean=0.3233531830113996`
    - 结果文件：
      - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms/navtest_full_2026-03-14_13-22-03/2026.03.14.14.54.12.csv`
- T4：
  - 标准模板本身对 `navhard two-stage` 的能力已具备，但本任务文档未单独记录一条“由标准 wrapper 直接完成的 navhard full run”。
  - 当前替代验证来自独立任务：
    - `/data/liushiqi/AutoVLA/task/navhard_two_stage_8gpu_eval_task.md`
  - 结论：
    - `navhard` 当前主评测链路已跑通，且 fresh 单卡与 8 卡结果严格一致：
      - `final_extended_pdm_score=0.14937031924317998`
      - `score_diff=0.0`
- T5：
  - 已完成
  - 证据：
    - `tools/eval/run_navhard_two_stage_autovla.py` 在后续 `navhard` 单卡/8卡任务中持续可用，未被标准 wrapper 改造破坏。

## AutoVLA 模型评测排查（2026-03-14）
- 目标：
  - 在 `navtest + v2 cache` 上直接跑 `AutoVLA` 模型，而不是 `constant_velocity_agent` 基线。
- 已执行：
  - 先做最小复现，确认 `run_pdm_score_cot.py + agent=autovla_agent + metric_cache_v2/navtest_full` 是否可跑。
  - 再起 8 卡 tmux 全量任务：
    - `RUN_ID=navtest_autovla_v2cache_tmux8_20260314_152344`
    - 输出目录：`/data/liushiqi/AutoVLA/logs/eval/navtest_autovla_v2cache_tmux8_20260314_152344`
    - 启动脚本：`/tmp/run_navtest_autovla_v2cache_tmux8_2026-03-14_15-23-44.sh`
- 根因 1：环境混用会让定制评测入口直接 import 失败。
  - 若未显式设置
    - `PYTHONPATH=/data/liushiqi/AutoVLA:/data/liushiqi/AutoVLA/navsim`
  - 则 `run_pdm_score_cot.py` 可能混到 upstream `navsim`，报：
    - `ImportError: cannot import name 'plot_cameras_frame_with_bev_agent_cot' from navsim.visualization.plots`
  - 结论：
    - 旧的 `run_pdm_score_cot.py` 链路对 `PYTHONPATH` 顺序敏感，不能和 upstream 标准入口混用。
- 根因 2：即便修正到本地 `AutoVLA/navsim` 环境，当前定制 scorer 也不兼容 v2 cache。
  - 最小复现与 8 卡日志都稳定出现：
    - `AssertionError: PDMObservation: index 41 out of range!`
  - 典型栈：
    - `run_pdm_score_cot.py -> navsim/evaluate/pdm_score.py -> pdm_scorer.py::_calculate_ttc -> pdm_observation.py::__getitem__`
  - 说明：
    - 当前仓库里的定制 `pdm_score/pdm_scorer` 不能正确消费 `metric_cache_v2/navtest_full_2026-03-07_15-40-49`。
    - 这不是模型输出质量问题，而是评测实现与 v2 cache schema / horizon 不匹配。
- 伴随问题：
  - `shard_0` 还出现了额外的 GPU OOM：
    - `OutOfMemoryError: Tried to allocate 1.16 GiB`
  - 这是并行加载大模型时的资源竞争问题，但不是主阻塞；主阻塞仍是上面的 `index 41 out of range`。
- 当前结论：
  - `AutoVLA + 旧定制 run_pdm_score_cot.py` 这条链不能作为 `navsim v2` 正确评测入口。
  - 要得到“AutoVLA 模型 + 标准 v2 EPDMS”结果，必须绕开当前定制 scorer，改为：
    - 本地/新 wrapper 直接调用 `AutoVLA` 生成轨迹；
    - 然后走 upstream 标准 `pdm_score` 内核打分。

## AutoVLA 标准入口落地（2026-03-14）
- 已新增标准 AutoVLA one-stage 入口能力，复用现有标准 wrapper：
  - 脚本：`tools/eval/run_navsimv2_epdms_standard.py`
  - baseline 配置：`config/eval/navsimv2_epdms_standard.yaml`
  - AutoVLA 配置：`config/eval/navsimv2_epdms_standard_autovla.yaml`
- 关键实现：
  - 新增 `evaluation.mode` 分发：
    - `standard_baseline`
    - `autovla_one_stage`
  - 新增 `AutoVLA one-stage` 执行路径：
    - 直接用 `AutoVLAPredictor + _scene_to_autovla_payload`
    - 打分仍调用 upstream `navsim.evaluate.pdm_score`
    - 不再经过旧的定制 `run_pdm_score_cot.py / pdm_score.py`
  - 新增结果标准化：
    - 若 upstream 返回 `pdm_score` 列而不是 `score` 列，wrapper 会自动映射到 `score`
    - one-stage summary 统一输出：
      - `successful`
      - `failed`
      - `invalid_sum`
      - `score_mean`
  - 新增运行环境注入：
    - `run.env_vars` 可在 yaml 中自动设置：
      - `NUPLAN_MAPS_ROOT`
      - `OPENSCENE_DATA_ROOT`
      - `NUPLAN_MAP_VERSION`
- 单测：
  - 文件：`tools/eval/tests/test_run_navsimv2_epdms_standard.py`
  - 当前：`6 passed`

## AutoVLA 标准入口验证记录（2026-03-14）
- Dry-run（AutoVLA 配置）：
  - 命令：
    - `/data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --config config/eval/navsimv2_epdms_standard_autovla.yaml --dry-run`
  - 结论：
    - 标准 upstream 指纹正确
    - 进入 `autovla_one_stage`
    - `traffic_agents.mode=non_reactive` 已可正确实例化
    - 自动枚举 `num_tokens=12146`
- 首次真实 smoke（1 scene）：
  - 命令：
    - `CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --config config/eval/navsimv2_epdms_standard_autovla.yaml --override train_test_split.scene_filter.max_scenes=1 --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla/navtest_smoke_1_gpu2_retry_2026-03-14_15-55-00`
  - 结果：
    - `successful=1`
    - `failed=0`
    - `invalid_sum=0`
    - 说明：
      - 已成功走完整的 `AutoVLA -> upstream standard pdm_score`
      - 旧的 `index 41 out of range` 未再出现
    - 当时暴露出 wrapper 收尾统计只认 `score`、未兼容 `pdm_score` 的问题，随后已修复
- 修复后真实 smoke（1 scene）：
  - 命令：
    - `CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navsimv2_epdms_standard.py --config config/eval/navsimv2_epdms_standard_autovla.yaml --override train_test_split.scene_filter.max_scenes=1 --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla/navtest_smoke_1_gpu2_scorefix_2026-03-14_15-57-00`
  - 最终结果：
    - `successful=1`
    - `failed=0`
    - `invalid_sum=0`
    - `score_mean=1.0`
    - CSV：
      - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla/navtest_smoke_1_gpu2_scorefix_2026-03-14_15-57-00/2026.03.14.15.59.01.csv`

## 当前限制 / 注意事项
- 当前机器上 GPU0 显存碎片和占用较重，默认 `device=cuda` 会优先落到 GPU0，1-scene 冒烟时触发过：
  - `torch.OutOfMemoryError: Tried to allocate 1.16 GiB`
- 在当前共享机器状态下，建议显式指定较空闲卡，例如：
  - `CUDA_VISIBLE_DEVICES=2`
- `run_navsimv2_epdms_standard_autovla.yaml` 目前默认 checkpoint：
  - `/tmp/sft_epoch4_loss09352.ckpt`
  - 如需切换模型，只改该 yaml 即可。

## 190 机器全量评测启动记录（2026-03-14）
- 本次使用的独立配置：
  - `config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4.yaml`
- 模型 checkpoint：
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- 启动机器：
  - `root@10.199.7.190:2289`
- 运行方式：
  - `tmux` 后台
  - 显式绑定：`CUDA_VISIBLE_DEVICES=7`
  - 额外设置：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- tmux session：
  - `navtest_epdms_2026_03_14_16_06_37`
- 输出目录：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4/navtest_full_2026-03-14_16-06-37_190`
- stdout 日志：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4/navtest_full_2026-03-14_16-06-37_190/launcher_stdout.log`
- 模板主日志：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4/run_navsimv2_epdms_standard_autovla_sft_20260312_epoch4.log`
- 启动命令：
```bash
ssh -p 2289 root@10.199.7.190 "
tmux new-session -d -s navtest_epdms_2026_03_14_16_06_37 '
cd /data/liushiqi/AutoVLA &&
CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/miniconda/envs/autolsqv2/bin/python \
/data/liushiqi/AutoVLA/tools/eval/run_navsimv2_epdms_standard.py \
  --config /data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4.yaml \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4/navtest_full_2026-03-14_16-06-37_190 \
  2>&1 | tee /data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4/navtest_full_2026-03-14_16-06-37_190/launcher_stdout.log
'"
```
- 当前状态：
  - 已通过 dry-run
  - 全量任务已启动
  - 已进入 `Loading checkpoint shards` 阶段，确认不是空跑

## 190 机器保守配置重启记录（2026-03-15）
- 背景：
  - 上一版全量任务 `navtest_epdms_2026_03_14_16_06_37` 长时间无新日志、无 `summary.json/csv`。
  - 非中断采样显示主线程长期停在：
    - `AutoVLA.predict -> transformers.generate -> Qwen2.5-VL forward`
  - 因此停止旧任务，改用评测专用保守解码配置重启。
- 新增评测专用模型配置：
  - `config/eval/qwen2.5-vl-3B-navsimv2-autovla-epdms-conservative-model.yaml`
  - 关键参数：
    - `max_new_tokens: 256`
    - `temperature: 0.2`
    - `top_k: 20`
    - `top_p: 0.2`
- 新增标准评测配置：
  - `config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml`
- 已停止旧任务：
  - 旧 tmux session：`navtest_epdms_2026_03_14_16_06_37`
- 新任务：
  - tmux session：`navtest_epdms_cons_2026_03_15_04_48_25`
  - 输出目录：
    - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190`
  - stdout：
    - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/launcher_stdout.log`
- 重启命令：
```bash
ssh -p 2289 root@10.199.7.190 "
tmux new-session -d -s navtest_epdms_cons_2026_03_15_04_48_25 '
cd /data/liushiqi/AutoVLA &&
CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/miniconda/envs/autolsqv2/bin/python \
/data/liushiqi/AutoVLA/tools/eval/run_navsimv2_epdms_standard.py \
  --config /data/liushiqi/AutoVLA/config/eval/navsimv2_epdms_standard_autovla_sft_20260312_epoch4_conservative.yaml \
  --override output_dir=/data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190 \
  2>&1 | tee /data/liushiqi/AutoVLA/logs/eval/navsimv2_standard_epdms_autovla_sft_20260312_epoch4_conservative/navtest_full_2026-03-15_04-48-25_190/launcher_stdout.log
'"
```
- 当前观察（重启后前 1 分钟）：
  - 已完成 `Loading checkpoint shards`
  - `GPU7` 利用率约 `71%`
  - 进程状态：`R`
  - 暂未产出 `summary.json/csv`，但比旧任务明显更健康

## 190 机器保守配置全量结果补录（2026-03-16）
- 对应配置：
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
- 结果解读：
  - `AutoVLA 模型 + 标准 v2 EPDMS wrapper + upstream pdm_score` 这条链已经完成 full `navtest`。
  - 与当前标准 baseline full `navtest`（`0.3233531830113996`）相比，当前 SFT 最终 ckpt 明显更优。

## 总结结论
- 这份任务的核心目标已经达成：
  - 标准 v2 EPDMS 本地模板入口已落地；
  - AutoVLA 已可绕开旧定制 scorer，直接走 upstream 标准打分内核；
  - `navtest` baseline full 与 AutoVLA full 都已有最终结果。
- 当前仍未在本任务内单独补一条“标准 wrapper 直接完成的 navhard full 运行记录”，但 `navhard` 能力与并行一致性已由独立任务验证完成。
