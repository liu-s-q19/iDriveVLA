# Repository Guidelines

## Project Structure & Module Organization
AutoVLA’s training code sits in `models/` (core VLA architectures and wrappers) and `dataset_utils/` (SFT datasets, token builders). Experiment configs live under `config/` with `training/`, `eval/`, and dataset-specific YAML; keep new configs additive and referenceable from scripts. Automation entrypoints live in `tools/` (`run_sft.py`, `run_rft.py`, eval utilities) and `scripts/` (preprocessing pipelines for nuPlan, Waymo, nuScenes). `navsim/` mirrors the upstream nuPlan devkit; avoid modifying it without mirroring upstream changes. Large assets (pretrained weights, vocabularies) are cached in `codebook_cache/` and external download scripts in `scripts/download_*.sh`.

## Environment Setup
All commands assume the active shell uses `conda activate autovla_codeclean`, the canonical contributor environment. If it does not exist, create it from `environment.yml` (`conda env create -f environment.yml -n autovla_codeclean`) and reinstall editable dependencies via `pip install -e . --no-warn-conflicts` followed by `bash install.sh`. Keep CUDA/cuDNN versions aligned with Torch 2.4.0 and document any divergent package pins in `requirements.txt`.

## Full Access Security Baseline / 全权限安全基线
当前项目在 Full Access（全权限）模式下运行。为保障宿主机与数据安全，所有 Agent/开发者必须遵守以下规则：

1. **代码来源审计（Code Provenance）**
   - 禁止在未审查源码时，直接执行公网下载的 shell/Python 脚本或二进制文件。
   - 执行 `pip install` / `npm install` 前必须核对包名，防止 typosquatting。
   - 对未经审核来源的远程 `pickle` / `.pth` 模型文件保持高警惕，避免加载潜在恶意载荷。

2. **环境隔离建议（Environment Isolation）**
   - 文件写入应限制在 `/data/liushiqi/AutoVLA/` 及其子目录。
   - 严禁向系统敏感目录写入（如 `/etc/`、`/usr/bin/`、`/root/`）。
   - 定期检查异常后台进程（如未授权反弹 shell、挖矿进程等）。

3. **网络与资源边界（Network & Resource Boundary）**
   - 非训练/推理必要场景，禁止建立外部长连接或开启未授权监听端口。
   - 调用 GPU 时需控制显存分配与释放，避免泄漏、过载导致驱动崩溃或硬件过热。

4. **数据破坏风险（Data Destruction Risk）**
   - 自动化脚本中严禁使用破坏性命令（如 `rm -rf /`、`mkfs`），删除类操作必须做空值校验。
   - 在 `approval_policy = "never"` 场景下，执行大规模清理前必须先做增量备份。

### Local Dataset Layout & Environment Variables
- The navsim + nuPlan bundle is staged under `/data/dataset/navsim` on this machine with `maps/`, `sensor_blobs/`, `navsim_logs/`, and cached zip archives. Point all nuPlan/navsim config paths into this tree (e.g., `sensor_blobs=/data/dataset/navsim/sensor_blobs`).
- Export the following before preprocessing, training, or evaluation (adjust map version if you keep `nuplan-maps-v1.1` unzipped):
  ```bash
  export NAVSIM_DEVKIT_ROOT=/data/liushiqi/AutoVLA/navsim
  export NAVSIM_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
  export NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs
  export OPENSCENE_DATA_ROOT=/data/dataset/navsim
  export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
  export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
  ```
- Cache pretrained Qwen2.5-VL checkpoints under `/data/AutoVLA` or another fast SSD and update `model.pretrained_model_path` in configs accordingly.

### Default Path Quick Map / 默认路径速览
- `NAVSIM_ROOT=/data/dataset/navsim`
- `SENSOR_BLOBS_ROOT=/data/dataset/navsim/sensor_blobs`
- `PREPROCESS_FULL_OUT=/data/dataset/navsim/preprocessed/navtrainval_nocot`
- `PREPROCESS_SMOKE_OUT=/data/dataset/navsim/preprocessed/navtest_nocot`
- `METRIC_CACHE_ROOT=/data/dataset/navsim/metric_cache`
- `RUNS_ROOT=/data/liushiqi/AutoVLA/runs`

### Run Modes / 运行模式
- `Production (full)`:
  - `CONFIG=dataset/qwen2.5-vl-72B-trainval`
  - `OUTPUT_DIR=/data/dataset/navsim/preprocessed/navtrainval_nocot`
  - 用于正式 SFT/GRPO 数据准备。
- `Debug (smoke)`:
  - `CONFIG=dataset/qwen2.5-vl-72B-nuplan`
  - `OUTPUT_DIR=/data/dataset/navsim/preprocessed/navtest_nocot`
  - 仅用于快速联调与脚本回归验证。

## Build, Test, and Development Commands
```bash
# run inside the autovla_codeclean environment
bash scripts/run_nuplan_preprocessing.sh                         # nuPlan preprocessing (+CoT when INCLUDE_COT=1)
python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft  # supervised fine-tuning
bash scripts/run_rft.sh                                          # GRPO reinforcement fine-tuning
python tools/eval/nusc_eval.py --config config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml \
    --checkpoint runs/sft/.../epoch=XX-loss=YY.ckpt --seg_data_path <seg_dir>
```
Document custom paths inside configs before committing.

## Long Job Policy / 长任务运行规范
- 除非是快速测试、语法检查或用户明确要求前台运行，所有长任务默认后台执行。
- 后台任务必须写入独立日志文件，禁止只依赖终端滚动输出。
- 日志必须按任务类型放入分类子目录：`logs/preprocess/`、`logs/sft/`、`logs/grpo/`、`logs/eval/`、`logs/debug/`（至少使用一级任务分类，禁止所有日志平铺在 `logs/` 根目录）。
- 空日志（0 bytes）和失败日志必须及时清理，避免日志目录堆积无效文件。
- 启动后台任务后，必须立即输出：
  - 进程 PID
  - 日志绝对路径（便于 `tail -f` 查看）
- 推荐启动模板：
  ```bash
  mkdir -p logs/{preprocess,sft,grpo,eval,debug}
  TASK_TYPE="preprocess"  # preprocess|sft|grpo|eval|debug
  RUN_TAG="nuplan_preprocess_$(date +%F_%H-%M-%S)"
  LOG_DIR="/data/liushiqi/AutoVLA/logs/${TASK_TYPE}"
  LOG_PATH="${LOG_DIR}/${RUN_TAG}.log"
  mkdir -p "$LOG_DIR"
  nohup bash scripts/run_nuplan_preprocessing.sh >"$LOG_PATH" 2>&1 &
  PID=$!
  sleep 2
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$LOG_PATH"   # 启动失败，立即删除失败/空日志
    echo "FAILED_TO_START PID=$PID"
  else
    echo "PID=$PID LOG=$LOG_PATH"
  fi
  ```
- 结果检查示例：
  ```bash
  tail -f /data/liushiqi/AutoVLA/logs/<task_type>/<run_tag>.log
  ```
- 日志清理示例（每次任务结束后执行）：
  ```bash
  # 删除空日志
  find /data/liushiqi/AutoVLA/logs -type f -name '*.log' -size 0 -delete

  # 删除最近一次判定失败的日志（按实际 run_tag 替换）
  rm -f /data/liushiqi/AutoVLA/logs/<task_type>/<run_tag>.log
  ```

## Preflight Checklist / 运行前检查
1. `conda env` 已切换到 `autovla_codeclean`。
2. `NAVSIM_DEVKIT_ROOT/NUPLAN_DATA_ROOT/NUPLAN_MAPS_ROOT` 等环境变量已导出。
3. `config/dataset/*.yaml` 中的 `dataset_path`、`scene_filter` 与目标 split 一致。
4. 输入数据目录存在且可读：`navsim_logs/<split>`、`sensor_blobs/<split>`。
5. 输出目录存在且可写：例如 `/data/dataset/navsim/preprocessed/navtrainval_nocot`。

## Training & Reproduction Workflow
1. **Data preprocessing**
   - nuPlan/navsim CoT vs. No-CoT: edit `scripts/run_nuplan_preprocessing.sh` to set `INCLUDE_COT`, `CONFIG` (e.g., `dataset/qwen2.5-vl-72B-trainval`), and `OUTPUT_DIR` (recommend `/data/dataset/navsim/preprocessed/navtrainval_nocot`; use `/data/dataset/navsim/preprocessed/navtest_nocot` only for quick smoke tests). The script dispatches to `tools/preprocessing/{cot_sample_generation,nocot_sample_generation}.py`, which expect the dataset paths defined inside `config/dataset/*.yaml`. Ensure `dataset_path` and `scene_filter` there match the split you stored under `/data/dataset/navsim`.
   - Waymo and nuScenes preprocessing follow `scripts/run_waymo_e2e_{image_extraction,preprocessing}.sh` and `scripts/run_nuscenes_preprocessing.sh` respectively; point `--nuscenes_path`/`--output_dir`/`--drivelm_path` to your mirrors.
   - Generated JSON samples are consumed by both SFT and GRPO configs as `data.*.json_dataset_path`. Keep a simple directory naming scheme (`navtrainval_nocot`, `navtrainval_cot`; `navtest_*` for smoke tests only) and add them to git-ignored storage such as `/data/dataset/navsim/preprocessed`.

2. **Action codebook creation**
   - Run `python tools/action_token/action_token_cluster.py --data_path /data/dataset/navsim/preprocessed/navtrainval_nocot --output codebook_cache/agent_vocab.pkl --num_cluster 2048`.
   - The resulting pickle is read by `models/action_tokenizer.py`; commit only small vocabularies, but keep the high-resolution version in `codebook_cache/`.

3. **Supervised fine-tuning (SFT)**
   - Choose or duplicate a YAML under `config/training/` (currently `qwen2.5-vl-3B-mix-sft.yaml`) and edit:
     - `model.pretrained_model_path` to your local Qwen checkpoint.
     - `data.train.json_dataset_path`/`sensor_data_path` lists to reference the preprocessed JSON and raw sensor blobs under `/data/dataset/navsim`.
     - `training` section for batch size, epochs, gradient accumulation, and optional `train_sample_size`.
     - `model.use_cot` toggles dual-mode prompting. Set to `false` if you fed only No-CoT JSON.
   - Launch `python tools/run_sft.py --config training/<name-without-.yaml>` (the script uses Lightning + FSDP). Checkpoints and CSV logs land in `runs/sft/<timestamp>/`.

4. **Reinforcement fine-tuning (GRPO)**
   - Start from `config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` and update:
     - `model.sft_model_path` to the best `.ckpt` from SFT.
     - `data.*.json_dataset_path`, `sensor_data_path`, and `metric_cache_path` (precompute PDMS caches with navsim or reuse `/data/dataset/navsim/*_metric_cache` once generated).
     - `training.devices` to match the visible GPUs (script defaults to `[0,1]`).
     - Optional LoRA knobs under `model.lora`.
   - Run `CUDA_VISIBLE_DEVICES=0,1 python tools/run_rft.py --config training/qwen2.5-vl-3B-nuplan-grpo-cot`. FSDP + GroupSampler ensure each process evaluates every scene for reward computation.

5. **Evaluation**
   - Navsim PDMS: use `navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh` as a template. Update `TRAIN_TEST_SPLIT`, `CHECKPOINT`, all path variables (point them to `/data/liushiqi/AutoVLA` and `/data/dataset/navsim`), and keep `PYTHONPATH` including `$NAVSIM_DEVKIT_ROOT`. The script runs `navsim/planning/script/run_pdm_score_cot.py agent=autovla_agent`.
   - nuScenes open-loop: `python tools/eval/nusc_eval.py --config config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml --checkpoint /path/to/runs/sft/...ckpt --seg_data_path /path/to/nusc_eval_seg`. The evaluator reuses `dataset_utils.sft_dataset.SFTDataset`, so its config’s `data.val.*` must point to a valid JSON + sensor blob pair.

6. **Logging and artifacts**
   - Lightning checkpoints follow `runs/{sft,grpo}/<timestamp>/epoch=E-loss=L.ckpt` or `rft-step####-reward####.ckpt`.
   - CSV logs drop under the same `runs/` directory; TensorBoard logs are enabled for GRPO. Capture evaluation tables from `tools/eval` and hydra outputs (`navsim/exp/...`) for PR attachments.

## Common Failure Triage / 常见故障速查
1. `ModuleNotFoundError: qwen_vl_utils`:
   - 原因：解释器环境不对。
   - 处理：确保使用 `autovla_codeclean`，或显式设置 `PYTHON_BIN=/data/miniconda/envs/autovla_codeclean/bin/python`。
2. `FileNotFoundError` 指向相机帧:
   - 原因：`sensor_blobs` 缺帧或读取了错误 split。
   - 处理：核对 `dataset_path` 和 `scene_filter`，确认目录是 `trainval` 而不是 `trainval_ini`。
3. 预处理路径检查失败:
   - 原因：`navsim_logs` / `sensor_blobs` 目录不存在或变量未导出。
   - 处理：按上文环境变量重新导出，并检查目录是否挂载。
4. 进程很卡或机器负载过高:
   - 原因：DataLoader workers 与磁盘吞吐不匹配。
   - 处理：优先减少 `--num_workers`，先保证稳定跑完再调速。

## Coding Style & Naming Conventions
Write Python 3.9+ code with 4-space indentation, snake_case APIs, and descriptive module names (e.g., `action_token_cluster.py`). Favor dataclasses or typed dicts for structured data; keep TorchLightning modules deterministic by seeding (`seed_everything`) like in `tools/run_sft.py`. Configuration keys should mirror YAML hierarchy (`model.use_cot`, `training.batch_size`). Run `python -m compileall <dir>` or `python -m pytest` if/when unit tests are added to catch syntax regressions, and lint with `ruff` or `flake8` locally if available.

## Testing & Evaluation
Regression checks rely on scenario-level metrics. Before pushing, rerun the relevant preprocessing script, then execute the matching evaluation script in `tools/eval/` with a held-out checkpoint and segmentation cache. Capture PrettyTable outputs for NuScenes and store under `runs/eval/` or attach to PRs. For new datasets, extend `PlanningMetric` with any extra KPIs and document expected values.

## Commit & Pull Request Guidelines
Keep commits small with imperative subjects (`add nusc eval sanity check`), mirroring existing history (`git log`). Reference issue IDs in the body, summarize data/command changes, and note required environment variables (e.g., `NUPLAN_MAPS_ROOT`). PRs should include: problem statement, config diff, command logs, evaluation tables, and screenshots when visualizing trajectories. Request review before merging any navsim or data schema edits to avoid breaking downstream pipelines.

### Commit Scope Guardrails / 提交范围约束
- `Whitelist (prefer commit)`: `config/`, `scripts/`, `tools/`, `models/`, `dataset_utils/`, `AGENTS.md`
- `Blacklist (do not commit)`: `logs/`, `task/`, `runs/`, `/data/dataset/navsim/preprocessed/*`, 临时调试输出
- 提交前执行一次 `git status --short`，确认无中间产物混入。

### Author Change Summaries (新增要求)
每次作者在本仓库修改代码或文档后，都需要在提交描述或评审说明中自我总结：
- 做了哪些修改：列出核心改动点及其目标。
- 中间遇到什么问题：描述遇到的阻塞、错误或需要权衡的地方，以及成因。
- 如何解决以及结果如何：解释最终的处理方式、为什么这样做、测试/验证结果如何。
保持这一三段式复盘，帮助后来者理解决策过程与经验教训。

模板示例 / Template:
```text
1) 做了哪些修改
- ...

2) 中间遇到什么问题
- ...

3) 如何解决以及结果如何
- 处理方式: ...
- 验证结果: ...
```

## Current Progress & Next Steps / 当前进展与后续计划
1. **nuPlan preprocessing** – Rerun `scripts/run_nuplan_preprocessing.sh` to populate `/data/dataset/navsim/preprocessed/navtrainval_nocot` and ensure PDMS metric caches exist (or regenerate them via navsim工具).  
   重新运行预处理脚本，生成 `navtrainval_nocot` JSON，并确认/生成 PDMS 指标缓存。
2. **Qwen checkpoint setup + SFT** – 下载或同步 Qwen 权重到 `/data/liushiqi/AutoVLA/`，确保 `model.pretrained_model_path` 可用，然后执行 `python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft`。  
   Download and place Qwen checkpoints locally before launching the SFT job above.
3. **GRPO + Navsim PDMS eval** – 将最优 SFT `.ckpt` 写入 GRPO 配置，运行 `tools/run_rft.py`，之后用 navsim PDMS 脚本做评估。  
   Feed the SFT checkpoint into GRPO and finish with the PDMS evaluation run.
4. **Author summaries** – 每个里程碑按“修改/问题/结果”三段在提交或说明里复盘。  
   Follow the author-summary rule at every milestone (what changed, issues, outcomes).

### Communication Preference / 沟通偏好
默认回复使用简体中文。如需其它语言或双语，请由用户显式提出。
- 每次启动后台长任务后，回复末尾必须附上日志绝对路径。

## Maintenance & Improvement Ideas
1. Add a `Makefile` (targets for `env`, `lint`, `sft`, `eval`) so future contributors can discover workflows faster.
2. Wire up `pre-commit` (ruff, black, yaml-lint) and document the hook versions to reduce formatting churn.
3. Publish minimal synthetic fixtures under `tests/` for dataset loaders to enable CI smoke tests without proprietary assets.
4. Capture secrets or dataset paths via `.env.example` plus `hydra` defaults to avoid hard-coded absolute paths in configs.
