# Navsimv2 环境可用性验证任务

状态：已完成

## 目标
- 在 `autolsqv2` 环境中完成 upstream `navsim v2.0` 的安装、切换与可用性验证。
- 不修改 AutoVLA 主训练逻辑，只完成环境层验收。

## 最终结论
- `autolsqv2` 中的 `navsim` 已切换到 upstream `2.0.0`。
- 基础导入、`run_metric_caching.py`、`run_pdm_score.py` 的入口检查均通过。
- NavSim v2 环境已满足后续标准评测与训练实验要求。

## 关键结果
- `navsim.__file__` 指向：`/data/liushiqi/navsim/navsim/__init__.py`
- 版本：`2.0.0`
- 入口检查：
  - `run_metric_caching.py train_test_split=navtest --help` 通过
  - `run_pdm_score.py train_test_split=navtest --cfg job` 通过

## 后续影响
- 后续 `NavSim v2` 标准评测应优先使用该 upstream 环境。
- 历史 v1.1 / vendored navsim 链路不得再混用为标准 v2 口径。

## 原版归档
- 详细执行日志、安装过程、M6/M7 扩展记录：
  - `/data/liushiqi/AutoVLA/task/archive/originals/navsimv2_env_task.md`
