# Navtest 8GPU Standard EPDMS And Progress Logging Design

## Goal
为 NavSim v2 标准 `navtest` 评测提供正式的 8GPU 分片入口，并让单 shard 日志在长时间评测过程中持续输出 token 级进度。

## Scope
- 新增 `navtest` 标准 EPDMS 8GPU 分片准备、执行、合并入口。
- 保持现有 `run_navsimv2_epdms_standard.py` 单进程入口可用。
- 为 `autovla_one_stage` 路径增加可配置的中间进度日志。
- 不改 `navhard` 现有分片链路。

## Design
- 复用 `navhard` 现有 `prepare -> shard runner -> merge -> shell launcher` 结构，避免为 `navtest` 继续维护超长临时 bash 命令。
- `prepare_navtest_epdms_shards.py` 负责解析 config、加载 scene/cache token、求交集、按 round-robin 切 shard，并生成 manifest。
- `run_navtest_epdms_standard_shard.py` 只负责读取单 shard manifest，并调用现有标准 wrapper 的 `autovla_one_stage` 路径完成评测。
- `merge_navtest_epdms_shards.py` 负责汇总各 shard 的 csv/summary，输出 merged summary/csv。
- `run_navsimv2_epdms_standard.py` 在 token 主循环内按固定间隔打印进度日志，字段包含 `done/total success failed elapsed eta last_token`。

## Constraints
- 输出目录结构应与 `navhard` 分片保持一致，便于运维和后续脚本复用。
- 进度日志默认每 50 个 token 打一次，避免日志爆炸。
- 单 shard 失败时 launcher 必须返回非零，避免 merge 误以为成功。
