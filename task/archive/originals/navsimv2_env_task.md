# Navsimv2 环境可用性验证任务（autolsqv2）

## 任务目标
- 在 `autolsqv2`（Python 3.9.23）环境中完成官方 upstream `navsim v2.0` 的环境可用性验证。
- 不修改 AutoVLA 主仓业务代码，仅进行环境层安装与验证。

## 约束与默认
- 数据根目录沿用现有约定：`/data/dataset/navsim`
- 当前阶段只做环境可用性，不跑训练任务。
- 采用方案 A：原地隔离旧版 `navsim` 并安装官方 `v2.0`。

## 里程碑
- [x] M1 前置盘点（版本、包来源、环境变量、可回滚信息）
- [x] M2 依赖隔离（移除旧 `navsim 1.1.0` 绑定）
- [x] M3 安装官方 upstream `navsim v2.0`
- [x] M4 可用性验收（T1~T4）
- [x] M5 结果报告与回滚建议

## 验收项（T1~T4）
- [x] T1 包来源一致性：`navsim` 指向官方 v2.0 安装路径，不再指向 `/data/hezeyu/...`
- [x] T2 导入测试：`python -c "import navsim"` 成功
- [x] T3 入口测试：`run_pdm_score` / `metric_caching` 帮助信息可拉起
- [x] T4 配置测试：可读取 navtest 配置并完成参数展开（不跑全量）

## 执行日志
### 2026-03-07
- 初始化任务卡。
- M1 盘点结果：
  - `autolsqv2` Python: `3.9.23`
  - 当前 `navsim` 来源：`/data/hezeyu/AutoVLA/navsim/navsim/__init__.py`
  - 当前 `navsim` 发行版本：`1.1.0`
  - `pip show` 快照：`task/navsimv2_pip_show_pre.txt`
  - `pip freeze` 快照：`task/navsimv2_pip_freeze_pre.txt`
  - 环境变量快照：`task/navsimv2_envvars_pre.txt`（当前会话为空）
  - 观察到 warning：`Ignoring invalid distribution -orch`（后续安装需关注）
- M2 依赖隔离：
  - 已执行 `pip uninstall -y navsim`
  - 移除旧绑定来源：`/data/hezeyu/AutoVLA/navsim`（`1.1.0`）
- M3 安装 v2.0：
  - 外网到 GitHub 下载速度过慢，改用本机镜像仓库 `/data/liushiqi/navsim`
  - 已执行 `pip install -e /data/liushiqi/navsim --no-deps --no-warn-conflicts`
  - 镜像仓库版本：`2.0.0`
- M4 验收进展：
  - T1 通过：`pip show navsim` 的 editable location 指向 `/data/liushiqi/navsim`
  - T2 通过：`import navsim` 成功，`navsim.__file__` 指向 `/data/liushiqi/navsim/navsim/__init__.py`
  - T3/T4 补充验证（在 `autolsqv2` 下，且导出 NAVSIM/NUPLAN/OPENSCENE 相关环境变量）：
    - `run_metric_caching.py train_test_split=navtest --help` 返回码 `0`
    - `run_pdm_score.py train_test_split=navtest --cfg job` 返回码 `0`
    - 临时日志：`/tmp/navsimv2_t3_1772895191.log`、`/tmp/navsimv2_t4_1772895191.log`

## 当前结论（最终）
- `autolsqv2` 内 `navsim` 已切到官方 upstream 代码系的 `2.0.0`（本机镜像），基础导入和主入口帮助信息可用。
- 环境可用性验证（T1~T4）已全部通过，可进入下一阶段（例如最小数据子集的真实评测冒烟）。

## 回滚建议
1. 若需回到旧环境，先在 `autolsqv2` 执行 `pip uninstall -y navsim`。
2. 再按旧来源执行可编辑安装（若仍需 v1.1.0 工作流）。
3. 若后续出现依赖冲突，建议启用方案 C：单独建立 `navsimv2` 子环境，避免与现有链路互相影响。

## M6：v2 性能上界评测（human/gt + idm）
- 口径定义（与 `task/reproduce_autovla_task.md` 一致）：
  - `human/gt`：`agent=human_agent`
  - `idm`：`agent=constant_velocity_agent`
- 评测脚本：`/data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py`
- 运行环境：`autolsqv2` + navsim upstream `2.0.0`
- 计划状态：`Running（已切换到 v2 cache）`

### 计划命令（v2）
1. human/gt:
   - `python run_pdm_score.py train_test_split=navtest agent=human_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache/navtest output_dir=<logs/eval/..._csv>`
2. idm:
   - `python run_pdm_score.py train_test_split=navtest agent=constant_velocity_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache/navtest output_dir=<logs/eval/..._csv>`

### 执行记录（持续更新）
- `2026-03-07`：已确认定义与命令口径，准备启动后台任务并回填 PID/日志路径。
- `2026-03-07 15:08 UTC`：已启动 v2 baseline 任务（后台）：
  - human/gt:
    - PID: `3207699`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_2026-03-07_15-08-12.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_2026-03-07_15-08-12_csv`
  - idm:
    - PID: `3207878`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_2026-03-07_15-08-14.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_2026-03-07_15-08-14_csv`
  - 进程检查：`pgrep -af run_pdm_score.py ...` 已确认两条命令均在运行。
- `2026-03-07 15:10~15:11 UTC`：发现首轮 `nohup` 启动不稳定（僵尸进程、未稳定驻留），切换到 `setsid -f` 方式重启。
- `2026-03-07 15:11 UTC`：重启后稳定运行（当前有效任务）：
  - human/gt:
    - wrapper PID: `3209917`
    - python PID: `3209921`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_2026-03-07_15-11-18.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_2026-03-07_15-11-18_csv`
  - idm:
    - wrapper PID: `3210176`
    - python PID: `3210180`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_2026-03-07_15-11-21.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_2026-03-07_15-11-21_csv`
- `2026-03-07 15:18 UTC`：进度监控
  - 两条任务已进入 `Processing stage one reactive scenario` 阶段。
  - 基于 `run_pdm_score.log` 的粗略 ETA：
    - human/gt：预计约 `15:58 UTC` 完成
    - idm：预计约 `15:59 UTC` 完成
  - 观测到少量 `Agent failed for token ...` 警告，需以最终汇总的 `valid`/失败场景计数为准。
- `2026-03-07 15:29 UTC`：根因定位与处置
  - v2 运行日志出现大量同类异常（human 与 idm 一致）：
    - `AttributeError: 'MetricCache' object has no attribute 'current_tracked_objects'`
    - 异常位置：`navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py`
  - 抽样验证：
    - `/data/dataset/navsim/metric_cache/navtest/.../metric_cache.pkl` 缺少 `current_tracked_objects`
    - `/data/liushiqi/recogdrive/exp/metric_cache/.../metric_cache.pkl` 也缺少该字段（现有旧缓存同样是旧 schema）
  - 对照验证：
    - `task/reproduce_autovla_task.md` 中 v1 跑分日志未检出该异常（`Agent failed/current_tracked_objects/AttributeError` 均为 0）
  - 结论：
    - 问题是“旧版 metric cache schema”与“v2 scorer/reactive traffic policy”不兼容，不是 human/idm 定义错误。
  - 处理：
    - 已停止当前两条无效跑分进程（`3209921`、`3210180`）。
- `2026-03-08 02:12 UTC`：使用新 v2 cache 重新启动 M6（后台）
  - cache: `/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49`
  - human/gt:
    - PID: `3583386`（python `3583390`）
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19_csv`
  - idm:
    - PID: `3583584`（python `3583588`）
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22_csv`
  - 启动后检查：
    - 两条 python 进程均在运行
    - `run_pdm_score.log` 已生成
    - 尚未检出 `AttributeError: 'MetricCache' object has no attribute 'current_tracked_objects'`

## M7：v2 metric cache 重建计划（采用 RecogDrive 快速参数）
- 状态：`Running (navtrain) / navtest done`
- 目标：生成与 v2 代码匹配的新 cache（含 `current_tracked_objects`），再重跑 M6。

### 计划原则
1. 不复用当前旧 cache（`/data/dataset/navsim/metric_cache/navtest`、`/data/liushiqi/recogdrive/exp/metric_cache`）。
2. 采用 RecogDrive 脚本的并行参数思想（`single_machine_thread_pool + process_pool + max_workers`）加速。
3. 先 smoke 再 full，避免再次全量跑空。

### M7 执行步骤
1. `Smoke`（navtest 小样本）
   - 使用 v2 代码路径：`/data/liushiqi/navsim/navsim/planning/script/run_metric_caching.py`
   - 关键参数参考 RecogDrive：
     - `worker=single_machine_thread_pool`
     - `worker.use_process_pool=True`
     - `worker.max_workers=64`（可按机器负载回调）
   - 额外加：`train_test_split.scene_filter.max_scenes=<小值>`
   - 输出到新路径：`/data/dataset/navsim/metric_cache_v2/navtest`
2. `Schema 验证`
   - 抽样 pickle，确认存在 `current_tracked_objects` / `past_detections_tracks`
3. `Full navtest`
   - 去掉 smoke 限制，完整生成 v2 navtest cache
4. `重跑 M6`
   - 使用 `metric_cache_path=/data/dataset/navsim/metric_cache_v2/navtest` 重跑 human/gt 与 idm
   - 记录最终 `valid` 数、失败比例和平均 `score`

### M7 执行记录（持续更新）
- `2026-03-07 15:39 UTC`：已启动 `navtest` smoke 缓存任务（v2）
  - PID: `3223264`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navtest_smoke_2026-03-07_15-39-11.log`
  - CACHE_ROOT: `/data/dataset/navsim/metric_cache_v2/navtest_smoke_2026-03-07_15-39-11`
  - 命令关键参数：
    - `train_test_split=navtest`
    - `metric_cache_path=/data/dataset/navsim/metric_cache_v2/navtest_smoke_2026-03-07_15-39-11`
    - `worker=single_machine_thread_pool`
    - `worker.max_workers=64`
    - `worker.use_process_pool=True`
    - `train_test_split.scene_filter.max_scenes=20`
  - 启动确认：
    - 日志已出现 `Starting Metric Caching...`
    - 新目录下已产出首个 `metric_cache.pkl`
- `2026-03-07 15:39 UTC`：smoke 结束
  - 日志结论：`Completed dataset caching! All 20 features and targets were cached successfully.`
  - 产出数量：`20` 个 `metric_cache.pkl`
  - schema 抽样验证（3 个样本）：
    - `has_current_tracked_objects=True`
    - `has_past_detections_tracks=True`
    - `has_future_tracked_objects=True`
  - 结论：v2 缓存 schema 验证通过，可进入 full。
- `2026-03-07 15:40 UTC`：已启动 full `navtest` v2 缓存任务（后台）
  - PID: `3224515`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navtest_full_2026-03-07_15-40-49.log`
  - CACHE_ROOT: `/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49`
  - 启动确认：
    - 已出现 `Starting metric caching of 136 files...`
- `2026-03-07 15:45 UTC`：速度与 ETA 估算
  - 当前进度（实时采样）：`8163 / 12146`
  - 当前吞吐：约 `24.28 cache/s`
  - 预计完成时间：约 `15:49 UTC`
  - 30 秒窗口增量采样：`+848`（`749 -> 1597`）
- 与 RecogDrive 历史记录对比（`/data/liushiqi/recogdrive/metric_caching.log`）：
  - 历史 navtest 完整缓存：`12146` 条，用时约 `1271s`（`05:57:48 -> 06:18:59`）
  - 历史平均吞吐：约 `9.56 cache/s`
  - 当前 v2 full 任务（本次）明显更快（约 `2.5x`）。
- `2026-03-07 15:49 UTC`：按需求并行启动 `navtrain` v2 缓存
  - PID: `3249714`（python 主进程 `3249718`）
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navtrain_full_2026-03-07_15-49-21.log`
  - CACHE_ROOT: `/data/dataset/navsim/metric_cache_v2/navtrain_full_2026-03-07_15-49-21`
  - 关键确认：
    - 使用的是 v2 代码路径：`/data/liushiqi/navsim/navsim/planning/script/run_metric_caching.py`
    - `train_test_split=navtrain`
    - 日志已出现：`Starting metric caching of 1192 files...`
  - 当前短窗测速（30s）：`+80`，约 `2.67 cache/s`
  - 时长估算（仅按当前并行负载的保守估计）：
    - 若维持当前速度，`navtrain` 剩余时长约 `9~12 小时`（区间值）
    - 待 `navtest` 结束释放资源后，预计可进一步缩短。
- `2026-03-08 02:10 UTC`：`navtest` full v2 cache 完成
  - 结果：`12146` 个 `metric_cache.pkl`
  - 标记：日志已出现 `Completed dataset caching! All 12146 features and targets were cached successfully.`
- `2026-03-08 02:2x UTC`：navtrain 缓存状态复核（用户关注“是否 v2 + 还要多久”）
  - 结论：当前 navtrain 任务是 **v2 缓存**（非 v1）
    - 运行脚本：`/data/liushiqi/navsim/navsim/planning/script/run_metric_caching.py`
    - 缓存路径：`/data/dataset/navsim/metric_cache_v2/navtrain_full_2026-03-07_15-49-21`
    - 参数：`train_test_split=navtrain`, `worker.max_workers=32`, `worker.use_process_pool=True`
  - 进度与速度采样：
    - 当前缓存文件数：`98195`（`metric_cache.pkl`）
    - 60 秒窗口：`+78`（约 `1.30 cache/s`）
    - 30 秒窗口：`+38`（约 `1.27 cache/s`）
  - 时长估算（只给“多久”）：
    - 参考 navtest 完整缓存比例（`12146 features / 136 files ≈ 89.31 features/file`）推算 navtrain 总量约 `10.6 万` features
    - 按当前采样速度估计，**剩余约 `1.5 ~ 2.5 小时`**（受并行负载与磁盘抖动影响）
- `2026-03-08 02:2x UTC`：M6（human/idm）继续监控
  - 运行状态：两条进程均在运行（未结束）
  - 关键异常复查：
    - `current_tracked_objects` / `AttributeError` / `Agent failed for token` 在主日志与 `run_pdm_score.log` 中当前均为 `0`
    - 说明：此前 v2 与旧 cache schema 不兼容的报错，当前未复现
  - 产物状态：两侧 `*_csv` 目录尚未产出最终 `.csv`（仍在计算中）

## M8：剩余时长统计脚本与技巧（新增）

### 目标
- 把 `run_pdm_score.log` 的 `scenario x / y in thread_id=...` 进度汇总为全局 `done/total`。
- 用固定时间窗（默认 60 秒）采样吞吐，直接估算 ETA（还需多久）。

### 脚本位置
- `/data/liushiqi/AutoVLA/task/navsim_eta_from_log.sh`

### 使用方式
1. human 日志 ETA：
   - `bash /data/liushiqi/AutoVLA/task/navsim_eta_from_log.sh /data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19_csv/run_pdm_score.log 60`
2. idm 日志 ETA：
   - `bash /data/liushiqi/AutoVLA/task/navsim_eta_from_log.sh /data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22_csv/run_pdm_score.log 60`

### 输出字段说明
- `done=a/b`：已完成 / 总量（自动从每个 thread 的最新计数汇总）。
- `rate=x/s`：采样窗口内的全局吞吐。
- `remaining`：剩余 scenario 数。
- `eta_sec/eta_hr`：按当前吞吐估算的剩余时长。

### 技巧
- 机器负载变化大时，建议连跑 3 次，取中位数作为对外汇报值。
- 若 `delta<=0`，脚本会输出 `eta=inf`，表示采样窗口太短或任务处于切换阶段，可把窗口调到 `120` 秒再测。
- 自测样例（5 秒窗口）：
  - `done=819/12146 ... rate=0.4000/s ... eta_hr=7.87`（说明脚本可正常解析并计算）

## M9：持续监控（navtrain + human/idm）
- 监控脚本：`/data/liushiqi/AutoVLA/task/monitor_navsimv2_status.sh`
- 已启动后台监控（5 分钟采样）：
  - 进程：`3608822`（wrapper: `3608819`）
  - 日志：`/data/liushiqi/AutoVLA/logs/debug/monitor_navsimv2_2026-03-08_03-17-07.log`
- 首次打点：
  - `navtrain_done=0 cache_count=100806`
  - `human_done=0 human_progress=1034/12146`
  - `idm_done=0 idm_progress=1026/12146`

## M10：阶段结论与结果位置（2026-03-09 更新）

### 结论
- `navtrain` v2 缓存：**已完成**
  - 完成标记：
    - `logs/eval/metric_cache_v2_navtrain_full_2026-03-07_15-49-21.log:43`
    - `Completed dataset caching! All 103288 features and targets were cached successfully.`
- `human/idm` v2 评测：**未产出最终结果（失败退出）**
  - 运行过程里 `stage one reactive scenario` 已推进到 `12146/12146`，但最终汇总阶段抛异常后退出。
  - 两个 wrapper 进程现为僵尸态（`[bash] <defunct>`），python 评测进程已结束。

### 失败原因（两条一致）
- 异常类型：`TypeError: 'NoneType' object is not iterable`
- 触发位置：`run_pdm_score.py:137`
  - `tokens_to_evaluate_stage_two = list(set(scene_loader_tokens_stage_two) & set(metric_cache_loader.tokens))`
- 证据：
  - `logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19.log:737`
  - `logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22.log:454`

### 结果文件位置说明
- v2 本轮 **没有最终 CSV**（`*_csv` 目录下无 `.csv` 文件）。
- 可用的产物仅为过程日志：
  - `logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19.log`
  - `logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19_csv/run_pdm_score.log`
  - `logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22.log`
  - `logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22_csv/run_pdm_score.log`

### 监控状态
- 监控日志显示：
  - `2026-03-08 23:37:28 UTC` 起 `human/idm` worker 均为 `0`，进度保持 `12146/12146` 不再变化。
- 后台监控已停止（避免空转），保留监控日志用于追溯：
  - `logs/debug/monitor_navsimv2_2026-03-08_03-17-07.log`

## M11：v2 指标测试顺序调整（2026-03-09 新增）

### 问题确认（用户提问）
- `navhard_two_stage` 不是“只能提交”，可以本地跑分。
- 但 `navhard_two_stage` 本地评测前，必须先完成该 split 的 **metric cache**（不能直接复用 `navtest` cache）。
- cache 前置依赖：
  - `${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs`
  - `${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles`

### 当前盘点结果（2026-03-09）
- 现有 v2 cache 仅包含：
  - `/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49`
  - `/data/dataset/navsim/metric_cache_v2/navtrain_full_2026-03-07_15-49-21`
- 暂未发现 `navhard_two_stage` 对应 cache。
- 数据路径复核（用户补充）：
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs` 存在
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles` 存在
  - `/readOnly/df_l2.9/navsim/warmup_two_stage`、`/readOnly/df_l2.9/navsim/private_test_hard_two_stage` 也存在
  - 结论：`navhard_two_stage` 数据已具备，可直接进入缓存阶段（无需先下载）

### 新执行顺序（先 navtest，再 navhard_two_stage）
1. `S1`：先完成 v2 `navtest` 指标复测（human/gt + idm），确认修复后能产出最终 CSV。
2. `S2`：检查并准备 `navhard_two_stage` 数据目录（若缺失则先下载/同步）。
3. `S3`：执行 `navhard_two_stage` 的 v2 metric caching（新路径，不复用 navtest cache）。
4. `S4`：在 `navhard_two_stage` 上跑 human/gt + idm，并记录 `extended_pdm_score_*`。

### 命令模板（待执行）
- `S1 navtest 复测（human）`
  - `python /data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=human_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49 output_dir=/data/liushiqi/AutoVLA/logs/eval/<navtest_human_csv_dir>`
- `S1 navtest 复测（idm）`
  - `python /data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=constant_velocity_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49 output_dir=/data/liushiqi/AutoVLA/logs/eval/<navtest_idm_csv_dir>`
- `S3 navhard_two_stage 缓存`
  - `OPENSCENE_DATA_ROOT=/readOnly/df_l2.9/navsim python /data/liushiqi/navsim/navsim/planning/script/run_metric_caching.py train_test_split=navhard_two_stage worker=single_machine_thread_pool worker.use_process_pool=True worker.max_workers=32 metric_cache_path=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_<timestamp>`
- `S4 navhard_two_stage 评测（human）`
  - `OPENSCENE_DATA_ROOT=/readOnly/df_l2.9/navsim python /data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py train_test_split=navhard_two_stage agent=human_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_<timestamp> synthetic_sensor_path=/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs synthetic_scenes_path=/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles output_dir=/data/liushiqi/AutoVLA/logs/eval/<navhard_human_csv_dir>`
- `S4 navhard_two_stage 评测（idm）`
  - `OPENSCENE_DATA_ROOT=/readOnly/df_l2.9/navsim python /data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py train_test_split=navhard_two_stage agent=constant_velocity_agent worker=single_machine_thread_pool max_number_of_workers=16 metric_cache_path=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_<timestamp> synthetic_sensor_path=/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs synthetic_scenes_path=/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles output_dir=/data/liushiqi/AutoVLA/logs/eval/<navhard_idm_csv_dir>`

### S1 执行记录（2026-03-09 03:29 UTC）
- 已按“先测试 v2 test 指标”启动 `navtest` 复测：
  - human/gt
    - shell PID: `4069769`
    - python PID: `4069773`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2fix_2026-03-09_03-29-16.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2fix_2026-03-09_03-29-16_csv`
    - 启动标记：`Starting pdm scoring of 12146 scenarios...`
  - idm
    - shell PID: `4070039`
    - python PID: `4070043`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2fix_2026-03-09_03-29-16.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2fix_2026-03-09_03-29-16_csv`
    - 启动标记：`Starting pdm scoring of 12146 scenarios...`

### S3 执行记录（2026-03-09 03:37 UTC）
- 已按要求在 `10.199.7.33` 启动 `navhard_two_stage` v2 缓存（后台）：
  - host: `10.199.7.33`（`ssh -p 2289 root@10.199.7.33`）
  - python PID: `1984573`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navhard_full_2026-03-09_03-37-22_n733.log`
  - CACHE_ROOT: `/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`
  - 关键参数（路径显式覆盖，适配 `/readOnly/df_l2.9/navsim` 目录结构）：
    - `train_test_split=navhard_two_stage`
    - `navsim_log_path=/readOnly/df_l2.9/navsim/test_navsim_logs/test`
    - `original_sensor_path=/readOnly/df_l2.9/navsim/test_sensor_blobs/test`
    - `synthetic_sensor_path=/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`
    - `synthetic_scenes_path=/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles`
    - `worker=single_machine_thread_pool`
    - `worker.use_process_pool=True`
    - `worker.max_workers=32`
  - 启动状态：
    - 日志已出现 `Starting Metric Caching...`
    - 已进入 `Loading synthetic scenes` 阶段（持续推进）

### 进度与耗时快照（2026-03-09 04:33 UTC）
- `navtest` 复测 ETA（基于 `task/navsim_eta_from_log.sh` 的 30s 窗口）：
  - human/gt:
    - `done=1038/12146`
    - `rate=0.3000/s`
    - `remaining=11108`
    - `eta_hr=10.29`
  - idm:
    - `done=1036/12146`
    - `rate=0.3333/s`
    - `remaining=11110`
    - `eta_hr=9.26`
  - 说明：窗口测速波动较大，建议按 `9~11 小时` 区间看待。
- `navhard_two_stage` 缓存（10.199.7.33）：
  - 完成标记：`Completed dataset caching! All 5912 features and targets were cached successfully.`
  - 日志：`/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navhard_full_2026-03-09_03-37-22_n733.log`
  - 用时：约 `9 分钟`（`03:38:35 -> 03:47:29`）

### S4 执行记录（2026-03-09 04:47 UTC）
- 已在 `10.199.7.33` 启动 `navhard_two_stage` 对比评测（后台）：
  - human/gt
    - shell PID: `2000729`
    - python PID: `2000733`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_human_gt_n733_2026-03-09_04-47-09.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_human_gt_n733_2026-03-09_04-47-09_csv`
  - idm
    - shell PID: `2000748`
    - python PID: `2000752`
    - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_n733_2026-03-09_04-47-09.log`
    - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_n733_2026-03-09_04-47-09_csv`
  - 关键参数：
    - `train_test_split=navhard_two_stage`
    - `metric_cache_path=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`
    - `navsim_log_path=/readOnly/df_l2.9/navsim/test_navsim_logs/test`
    - `original_sensor_path=/readOnly/df_l2.9/navsim/test_sensor_blobs/test`
    - `synthetic_sensor_path=/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`
    - `synthetic_scenes_path=/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles`
  - 当前状态：
    - 两条任务均已进入 `Loading logs` 与 `Loading synthetic scenes` 阶段。

### 进度与耗时快照（2026-03-09 06:33 UTC）
- `navtest` v2 复测 ETA（使用 `task/navsim_eta_from_log.sh`，120s 窗口平滑）：
  - human/gt: `done=2687/12146`, `rate=0.2417/s`, `remaining=9459`, `eta_hr=10.87`
  - idm: `done=2725/12146`, `rate=0.2417/s`, `remaining=9421`, `eta_hr=10.83`
  - 结论：当前剩余时间约 `10.8~10.9 小时`（按最近 120s 平滑速率）。
- `navhard_two_stage`（`10.199.7.33`）当前仍在数据加载阶段（尚未进入 `Processing stage one reactive scenario`）：
  - human 最新加载进度：`5184/5462`, `1.55s/it`（日志显示 `95%`）
    - 仅加载阶段剩余约 `278 * 1.55s ~= 7.2 分钟`
  - idm 最新加载进度：`5398/5462`, `1.17s/it`（日志显示 `99%`）
    - 仅加载阶段剩余约 `64 * 1.17s ~= 1.3 分钟`
  - 说明：`navhard` 主评测阶段 ETA 需等日志出现 `Processing stage one reactive scenario` 后再用同脚本估算。

### 进度与耗时快照（2026-03-09 07:08 UTC）
- `navhard_two_stage`（`10.199.7.33`）已完成 synthetic scenes 加载并进入主评测（`run_pdm_score.log` 已出现 `Processing stage one reactive scenario`）。
- 60s 窗口 ETA（`task/navsim_eta_from_log.sh`）：
  - human/gt: `done=2703/3008`, `rate=1.6833/s`, `remaining=305`, `eta_sec=181`, `eta_hr=0.05`
    - 预计剩余约 `3 分钟`
  - idm: `done=1009/3008`, `rate=0.4333/s`, `remaining=1999`, `eta_sec=4613`, `eta_hr=1.28`
    - 预计剩余约 `1.3 小时`
- 结论：若当前速率稳定，`navhard` 全部完成时间主要由 `idm` 决定，约还需 `1.2~1.4 小时`。

### 诊断快照（2026-03-09 07:54 UTC）
- `navhard human/gt` 存在明确异常，且已提前退出（无 CSV 产出）：
  - 进程状态：`human` 进程已结束，`idm` 仍在运行。
  - 失败规模：`human_agent_failed=5462`（全部 token 失败），`idm_agent_failed=0`。
  - 关键堆栈：
    - `human_agent.py -> scene.get_future_trajectory(...)` 抛出 `IndexError: list index out of range`。
    - 随后聚合阶段出现 `AssertionError: Invalid interval nan`（two-frame comfort）。
    - 最终在 `calculate_individual_mapping_scores` 触发 `TypeError: Could not convert [array([nan...])] to numeric`，导致流程在写 CSV 前中断。
  - 直接现象：
    - `logs/eval/navsimv2_navhard_human_gt_n733_2026-03-09_04-47-09_csv/` 下无 `*.csv`。
- `navhard idm` 当前正常推进（无 `Agent failed`）：
  - 最新 60s ETA：`done=2216/4800`, `rate=0.3833/s`, `remaining=2584`, `eta_hr=1.87`。

### Root Cause 复盘（2026-03-09 08:xx UTC）
- 结论：`navhard_two_stage + human_agent` 失败是**数据形态与 agent 假设不匹配**，不是前面 `NoneType` 修复回归。
- 证据链：
  - 失败 token `b19a34b568608122f` 对应 synthetic scene 文件：
    - `/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles/e14d437202167c5e6.pkl`
  - 该 synthetic scene 元数据：
    - `num_history_frames=4`
    - `num_future_frames=0`
    - `num_frames=4`（仅历史帧）
  - 全量统计（`navhard_two_stage/synthetic_scene_pickles`）：
    - `total=5462`
    - `num_future_frames_dist={0: 5462}`
    - `num_frames_dist={(4): 5462}`
  - `human_agent` 代码固定取未来轨迹：
    - `human_agent.py:39` -> `scene.get_future_trajectory(...)`
    - `dataclasses.py:369-370` 会访问 `frames[start : start+num_poses]`
    - 需要未来帧，但 synthetic 只有 4 帧，触发 `IndexError: list index out of range`
- 连锁报错原因：
  - stage-two 大量无效后，聚合出现 `Invalid interval nan`，随后在 `calculate_individual_mapping_scores` 求均值触发
    - `TypeError: Could not convert [array([nan, ...])] to numeric`
  - 导致流程在写 CSV 前终止（所以 human 输出目录无结果 CSV）。
- 影响范围：
  - `navhard_two_stage` 上 `human_agent` 不可用（按当前数据与脚本实现）。
  - `constant_velocity_agent`（idm）不依赖 scene future trajectory，故可正常运行。

### 进度与耗时快照（2026-03-09 09:50 UTC）
- `navtest`（本机）60s 窗口 ETA：
  - human/gt: `done=4853/12146`, `rate=0.1833/s`, `remaining=7293`, `eta_hr=11.05`
  - idm: `done=4914/12146`, `rate=0.1500/s`, `remaining=7232`, `eta_hr=13.39`
  - 结论：当前 `navtest` 两条预计约 `11~13.5 小时` 完成（短窗口有波动）。
- `navhard_two_stage`（10.199.7.33）：
  - human/gt: 已失败结束（不再计时；根因见上节）。
  - idm: `done=4893/5462`, `rate=0.3770/s`, `remaining=569`, `eta_hr=0.42`
    - 预计约 `25 分钟`。

### 结果快照（2026-03-09 10:08 UTC）
- `navhard_two_stage / idm` 已完成：
  - 完成日志：
    - `Finished running evaluation.`
    - `Number of successful scenarios: 5912`
    - `Number of failed scenarios: 0`
    - `Final extended pdm score of valid results: 0.11481608441587043`
  - 结果 CSV：
    - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_n733_2026-03-09_04-47-09_csv/2026.03.09.10.08.31.csv`
  - 汇总行（CSV）：
    - `extended_pdm_score_stage_one = 0.2895878390216141`
    - `extended_pdm_score_stage_two = 0.34246385161185255`
    - `extended_pdm_score_combined = 0.11481608441587043`

### 口径与环境更正（2026-03-09 11:14 UTC）
- 用户确认：`v2` 指标评测应统一使用 `autolsqv2` 环境（不应使用 `autovla_codeclean`）。
- 本次执行记录：
  - 我曾误尝试用 `autovla_codeclean` 启动一组 `IDM no-quant/quantized` 对比。
  - 这两条仅产生僵尸 shell（`bash <defunct>`），未形成有效评测进程、无有效日志路径落地，不影响当前 v2 主任务。
- 后续约束：
  - 任何 `navsim v2` 新评测（含 IDM 离散化对比）均改为 `autolsqv2 + v2 cache` 口径执行。

## M12：IDM action codebook 对比（v2 口径，进行中）
### 目标
- 在 `navtest + v2 cache + autolsqv2` 口径下，跑 `IDM no-quant` 与 `IDM quantized`。
- 输出：
  - `S_idm_ref`
  - `S_idm_q`
  - `z_idm_retain = S_idm_q / S_idm_ref`
  - `delta_idm_abs = S_idm_ref - S_idm_q`
  - `z_idm_rel_penalty = 1 - z_idm_retain`

### 关键变更（2026-03-09 11:2x~11:3x UTC）
- 为恢复 `quantization.*` 配置支持，已在当前分支恢复两处文件（来源：`19ff3b2`）：
  - `navsim/navsim/planning/script/run_pdm_score.py`
  - `navsim/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`
- 为兼容 v2 metric cache pickle，补回兼容定义：
  - `navsim/navsim/common/enums.py`：恢复 `SceneFrameType`
  - `navsim/navsim/planning/metric_caching/metric_cache.py`：恢复 `MapParameters`，并在 `MetricCache` 中加 `map_parameters: Optional[MapParameters] = None`

### 启动记录（稳定版本）
- 由于本地执行器会回收后台子进程，最终改为 `10.199.7.33` 远端 `nohup` 常驻启动。
- no-quant：
  - PID: `2040353`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_noquant_codebookcmp_n733_fixcompat_2026-03-09_11-39-45.log`
  - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_noquant_codebookcmp_n733_fixcompat_2026-03-09_11-39-45_csv`
- quantized：
  - PID: `2040355`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_quantized_codebookcmp_n733_fixcompat_2026-03-09_11-39-45.log`
  - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_quantized_codebookcmp_n733_fixcompat_2026-03-09_11-39-45_csv`

### 运行状态快照（2026-03-09 11:40 UTC）
- 进程存活：`2040353`、`2040355` 均在运行。
- 兼容性错误计数：
  - `SceneFrameType`：`0`
  - `MapParameters`：`0`
- 当前失败计数（早期窗口）：
  - no-quant：`Agent failed = 1`
  - quantized：`Agent failed = 0`
- 观察到的失败类型（no-quant 少量）：`AssertionError: PDMObservation: index 41 out of range!`
  - 先持续观察失败是否扩散；若只占极少 token，则可继续并以有效样本汇总。

### ETA 快照（脚本：`task/navsim_eta_from_log.sh`）
- no-quant（20s 窗口）：`done=293/12146 rate=4.3000/s eta_hr=0.77`
- quantized（60s 窗口）：`done=179/12146 rate=2.2833/s eta_hr=1.46`
- 结论：以当前速率估算，两条大约 `0.8~1.5 小时` 完成（短窗口会波动）。

### M12 修正（2026-03-09 12:xx UTC）
- 上述 `fixcompat` 两条任务已判定为**无效口径**并停止：
  - 终止 PID：`2040353`、`2040355`
  - 原因：`AutoVLA/navsim` 的旧版 scorer 与官方 v2 reactive 评分流程不一致，出现大量
    `AssertionError: PDMObservation: index 41 out of range!`
- 我将 `run_pdm_score.py / default_run_pdm_score.yaml` 回滚回 `19ff3b2` 状态，避免留在中间试验态。

### M12 新方案（官方 v2 scorer + 本地量化 agent）
- 运行入口切换为官方 upstream v2：
  - `/data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py`
- 环境与路径：
  - env: `autolsqv2`
  - `PYTHONPATH=/data/liushiqi/navsim:/data/liushiqi/AutoVLA`
  - split: `navtest`
  - cache: `/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49`
- 新增本地扩展 agent（用于“IDM 后量化”）：
  - `/data/liushiqi/AutoVLA/navsim_ext/quantized_constant_velocity_agent.py`
  - 类名：`navsim_ext.quantized_constant_velocity_agent.QuantizedConstantVelocityAgent`

### M12 正式任务（进行中，10.199.7.33）
- no-quant（官方 IDM）：
  - PID: `2051523`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_noquant_extv2_n733_2026-03-09_12-39-49.log`
  - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_noquant_extv2_n733_2026-03-09_12-39-49_csv`
- quantized（IDM + codebook）：
  - PID: `2051527`
  - LOG: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_quantized_extv2_n733_2026-03-09_12-39-49.log`
  - OUT: `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_quantized_extv2_n733_2026-03-09_12-39-49_csv`

### M12 当前快照（2026-03-09 12:45 UTC）
- 两条进程存活，已进入 `Processing stage one reactive scenario`：
  - no-quant 计数：`128`
  - quantized 计数：`130`
- 当前 `Agent failed`：`0 / 0`（两条）
- 当前 ETA（20s 窗口，波动较大）：
  - no-quant：`done=129/12146 rate=0.0500/s eta_hr=66.76`
  - quantized：`done=137/12146 rate=0.1000/s eta_hr=33.36`
- 说明：
  - 该窗口处于早期低速阶段（大量 worker 刚进入分段日志加载），需继续监控 10~20 分钟后再给稳定 ETA。

### M12 监控快照（2026-03-09 15:13 UTC）
- 进程状态（10.199.7.33）：
  - no-quant PID `2051523`：运行中（`ELAPSED=02:33:16`）
  - quantized PID `2051527`：运行中（`ELAPSED=02:33:16`）
- 当前进度（stage one 计数）：
  - no-quant：`2254`
  - quantized：`2225`
- 当前失败计数：
  - no-quant：`0`
  - quantized：`0`
- 最新 ETA（30s 窗口，波动大）：
  - no-quant：`done=2262/12146 rate=0.2000/s eta_hr=13.73`
  - quantized：`done=2233/12146 rate=0.1000/s eta_hr=27.54`
- 结论：
  - 两条任务仍在稳定推进，但当前窗口速率偏慢；需继续观察更长窗口后再给最终稳定完赛时间。

### M12 监控快照（2026-03-10 02:09 UTC）
- 进程状态（10.199.7.33）：
  - no-quant PID `2051523`：运行中（`ELAPSED=13:29:30`）
  - quantized PID `2051527`：运行中（`ELAPSED=13:29:30`）
- 统计状态：
  - `Finished running evaluation`：`0 / 0`（均未完成）
  - `Agent failed`：`0 / 0`（当前仍为 0）
  - `stage two reactive` 计数：`0 / 0`（尚未进入 stage two）
- stage one 进度采样（30s 窗口）：
  - no-quant：`done=8425/12146`，`delta=2/30s`，`rate=0.0667/s`，`eta_hr=15.50`
  - quantized：`done=8362/12146`，`delta=5/30s`，`rate=0.1667/s`，`eta_hr=6.31`
- 说明：
  - 上述 ETA 仅基于当前 stage one 窗口速率，且波动大；最终总耗时还需叠加 stage two 阶段时间。

### M12 终态复盘（2026-03-12 06:18 UTC）
- 任务状态：
  - 两条进程均已退出为僵尸态（`<defunct>`），未正常完赛。
  - `Finished running evaluation`：`0 / 0`
  - 输出目录均无最终 CSV。
- 直接报错链（两条一致）：
  - `omegaconf.errors.ConfigAttributeError: Key 'reactive_all_mapping' is not in struct`
  - 随后在主流程中触发：
    - `UnboundLocalError: local variable 'all_mappings' referenced before assignment`
- 根因定位（官方 v2 scorer）：
  - `/data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py:381` 读取 `cfg.train_test_split.reactive_all_mapping`
  - `/data/liushiqi/navsim/navsim/planning/script/run_pdm_score.py:417` 在 except 后仍使用 `all_mappings`
  - 当 split 配置缺失 `reactive_all_mapping` 时，会触发上述未定义变量错误并终止。

## M13：navhard IDM 离散化对比（v2，2026-03-12）
### 目标
- 在 `navhard_two_stage` 上补齐 `IDM quantized` 结果，并与已有 `IDM no-quant` 结果对比。

### 启动记录（10.199.7.33）
- 启动时间：`2026-03-12 06:56:50 UTC`
- split：`navhard_two_stage`（two-stage 口径）
- 环境：`autolsqv2` + `PYTHONPATH=/data/liushiqi/navsim:/data/liushiqi/AutoVLA`
- 关键参数：
  - `agent=constant_velocity_agent`
  - `agent._target_=navsim_ext.quantized_constant_velocity_agent.QuantizedConstantVelocityAgent`
  - `+agent.codebook_path=/data/liushiqi/AutoVLA/codebook_cache/agent_vocab.pkl`
  - `metric_cache_path=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`
- 进程：
  - shell PID：`2397857`
  - python PID：`2397864`
- 路径：
  - LOG：`/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_quantized_extv2_n733_2026-03-12_06-56-50.log`
  - OUT：`/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_quantized_extv2_n733_2026-03-12_06-56-50_csv`

### 对比基线（已存在）
- `IDM no-quant`：
  - `/data/liushiqi/AutoVLA/logs/eval/navsimv2_navhard_idm_n733_2026-03-09_04-47-09_csv/2026.03.09.10.08.31.csv`
  - `extended_pdm_score_combined = 0.11481608441587043`

### 进度与 ETA（2026-03-12 07:03 UTC）
- 当前状态：
  - 进程存活，CPU 活跃；日志未出现 `Traceback/Exception`。
  - 已进入 `Starting pdm scoring of 5912 scenarios`，worker 已全部拉起。
  - 尚未出现 `Processing stage one reactive scenario`，因此无法使用 `navsim_eta_from_log.sh` 进行实时速率 ETA。
- 历史推算 ETA（临时）：
  - 参考同口径历史 `IDM no-quant` 完赛时长约 `5h21m`（`2026-03-09 04:47:09 -> 10:08:31`）。
  - 量化 agent 预计存在额外计算开销，按 `1.1x~1.3x` 粗估总时长约 `5.9~7.0 小时`。
  - 本次已运行约 `6分51秒`，剩余约 `5.8~6.9 小时`（待进入 stage one 后改用实测速率更新）。

## M8：AutoVLA 新增 navhard_two_stage 场景驱动评测流（不改原脚本）
- 状态：`Smoke Passed (2+2)`
- 时间：`2026-03-13 09:09 UTC`

### 新增文件
1. `tools/eval/run_navhard_two_stage_autovla.py`
2. `config/eval/navhard_two_stage_autovla.yaml`

### 目标
- 不依赖预先固化 json，直接基于 upstream `SceneLoader`（original + synthetic）构造 AutoVLA 输入并评测 two-stage navhard。
- 兼容 `traffic_agents.mode`（`reactive`/`non_reactive`）。

### 本次关键修复
1. 兼容 `score_proposals` 签名差异：
   - 当前环境 `pdm_score` 会给 `human_past_trajectory`，但部分 scorer（如 `PDMTrafficScorer`）不接收该参数。
   - 新脚本已增加 fallback：按 `scorer.score_proposals` 实际签名动态传参。
2. 子集 smoke 的 two-stage 聚合映射过滤：
   - 子集场景下，先按 `available_tokens` 过滤 `reactive_all_mapping`，避免 `Missing token in score_df`。
3. 配置对齐：
   - `navhard_two_stage_autovla.yaml` 的 scorer 从 `pdm_and_traffic_scorer` 调整为 `pdm_scorer`（前者在当前环境触发 pandas 数值聚合异常）。

### smoke 命令（autolsqv2）
```bash
NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
/data/miniconda/envs/autolsqv2/bin/python \
/data/liushiqi/AutoVLA/tools/eval/run_navhard_two_stage_autovla.py \
  --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml \
  --set eval.max_stage_one_scenarios=2 \
  --set eval.max_stage_two_scenarios=2
```

### smoke 结果
- `num_successful_scenarios`: `4`
- `num_failed_scenarios`: `0`
- `final_extended_pdm_score`: `0.6536081396836846`
- `csv_path`: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/run_2026-03-13_09-07-03/2026.03.13.09.08.52.csv`
- `summary.json`: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/run_2026-03-13_09-07-03/summary.json`
- 说明：本次是 2+2 子集 smoke，映射覆盖日志 `raw=225 filtered=0`，属预期（子集不具备完整 two-stage mapping）。

### M8 全量 EPDMS 运行（后台）
- 启动时间：`2026-03-13 09:14:45 UTC`
- 运行方式：`tmux` 后台
- Session：`navhard_epdms_full_2026-03-13_09-14-45`
- Log：`/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/navhard_epdms_full_2026-03-13_09-14-45.log`
- 命令：
```bash
cd /data/liushiqi/AutoVLA
NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
/data/miniconda/envs/autolsqv2/bin/python \
/data/liushiqi/AutoVLA/tools/eval/run_navhard_two_stage_autovla.py \
  --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml
```
- 启动后关键日志：
  - `Eval setup: stage1_tokens=450 stage2_tokens=5462 traffic_mode=reactive metric_tokens=5912 scene_tokens=5912`
  - 已进入 `Stage1 token i/450` 逐场景评测。

### M8 运行中状态更新（2026-03-13 09:20:05 UTC）
- 当前进度：`Stage1 token 25/450`（最新日志时间 `2026-03-13 09:20:12 UTC`）。
- 当前阶段：仅 Stage1，尚未进入 `Stage2 token 1/5462`。
- 结论：当前还没有最终 EPDMS（`final_extended_pdm_score`）；需待两阶段全部完成并聚合后产出。

### M8 全量完成（2026-03-13 13:57:39 UTC）
- 运行完成：`Stage1 450/450` + `Stage2 5462/5462`，两阶段均已跑完。
- two-stage 映射覆盖：`raw=225 filtered=225 available_tokens=5912`（完整命中）。
- 最终结果（EPDMS）：
  - `num_successful_scenarios=5912`
  - `num_failed_scenarios=0`
  - `final_extended_pdm_score=0.13153159316313934`
  - `elapsed_sec=16867.3907`（约 `4h 41m`）
- 产物路径：
  - `summary.json`: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/run_2026-03-13_09-15-08/summary.json`
  - `csv`: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/run_2026-03-13_09-15-08/2026.03.13.13.57.39.csv`
  - `run log`: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/navhard_epdms_full_2026-03-13_09-14-45.log`
- /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/navhard_epdms_full_2026-03-13_09-14-45.log
- navtest 之前确实有指标
  但“标准 baseline 的 full 指标”有，值是 0.3233531830113996
  “AutoVLA 标准口径 full 指标”当前这条 190 的任务还没出最终结果
  “AutoVLA 定制链路的 navtest full 指标”以前也跑过，但不是标准 v2 EPDMS 口径
