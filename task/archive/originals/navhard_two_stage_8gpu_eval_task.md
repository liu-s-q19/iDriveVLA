# Navhard Two-Stage 8-GPU Eval Task

## Goal
- Add a shardable 8-GPU evaluation path for `tools/eval/run_navhard_two_stage_autovla.py`.
- Keep final merged scoring semantics aligned with the existing single-process `extended_pdm_score_combined` path.

## Design
- Do not convert the current two-stage evaluator into `torch.distributed`.
- Split work outside the core evaluator:
  - `prepare_navhard_two_stage_shards.py`: resolve config and generate shard manifests.
  - `run_navhard_two_stage_autovla_shard.py`: run one shard on one GPU and save raw partial rows as pickle.
  - `merge_navhard_two_stage_shards.py`: validate shard coverage and run the same final score aggregation used by the single-run path.
- Preserve the current single-run entrypoint for baseline/debug use.

## Files
- Shared helper: `tools/eval/navhard_two_stage_sharded.py`
- Single-run entrypoint updated to reuse helper merge logic: `tools/eval/run_navhard_two_stage_autovla.py`
- Shard prepare: `tools/eval/prepare_navhard_two_stage_shards.py`
- Shard worker: `tools/eval/run_navhard_two_stage_autovla_shard.py`
- Shard merge: `tools/eval/merge_navhard_two_stage_shards.py`
- Launcher: `scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`

## Output Layout
- Plan root: `<eval.output_dir>/sharded_run_<timestamp>`
- Plan file: `shard_plan.json`
- Manifests: `manifests/shard_00.json` ... `manifests/shard_07.json`
- Partial outputs per shard:
  - `partials/shard_XX/partial_rows.pkl`
  - `partials/shard_XX/partial_rows.csv`
  - `partials/shard_XX/summary.json`
  - `partials/shard_XX/runner.log`
  - `partials/shard_XX/launcher_stdout.log`
- Final merged outputs:
  - `merged/<timestamp>.csv`
  - `merged/summary.json`
  - `merged/merge.log`

## Commands
- Prepare only:
  - `NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps OPENSCENE_DATA_ROOT=/data/dataset/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 /data/miniconda/envs/autolsqv2/bin/python tools/eval/prepare_navhard_two_stage_shards.py --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml --num-shards 8`
- Run one shard:
  - `NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps OPENSCENE_DATA_ROOT=/data/dataset/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla_shard.py --plan-dir <plan_dir> --shard-index 0`
- Merge shards:
  - `/data/miniconda/envs/autolsqv2/bin/python tools/eval/merge_navhard_two_stage_shards.py --plan-dir <plan_dir>`
- Launch 8 shards end-to-end:
  - `bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`
  - Optional GPU pinning: `GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`
  - Optional custom plan root: `PLAN_DIR=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/custom_run bash scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`

## Safety Checks
- Shard merge validates `(stage_name, token)` coverage before aggregation.
- Partial rows are stored as pickle to preserve object-valued scoring columns used by pseudo closed-loop aggregation.
- Merge recomputes final `extended_pdm_score_*` summary rows only once, after all shard rows are combined.
- `navhard` two-stage scene loading requires `NUPLAN_MAPS_ROOT`, `OPENSCENE_DATA_ROOT`, and `NUPLAN_MAP_VERSION`; the launcher now exports sane defaults for these.

## Verification
- `python -m pytest tools/eval/tests/test_navhard_two_stage_sharded.py tools/eval/tests/test_run_navsimv2_epdms_standard.py -v`
- `python tools/eval/prepare_navhard_two_stage_shards.py --help`
- `python tools/eval/run_navhard_two_stage_autovla_shard.py --help`
- `python tools/eval/merge_navhard_two_stage_shards.py --help`
- `bash -n scripts/eval/run_navhard_two_stage_autovla_8gpu.sh`

## Root Causes Found
- Stage2 token order was unstable because upstream `SceneLoader.reactive_tokens_stage_two` / `non_reactive_tokens_stage_two` use unordered `set` intersection.
- Even after token alignment, single-run and sharded scores still drifted because eval inference used `AutoVLA.predict()` with `do_sample=True` and no per-token seed, so the same token could decode differently across processes / GPUs.

## Fixes Applied
- Added `select_stage_two_tokens()` to preserve `scene_filter` order while filtering by available synthetic scenes and metric cache tokens.
- Updated both `tools/eval/run_navhard_two_stage_autovla.py` and `tools/eval/prepare_navhard_two_stage_shards.py` to stop relying on upstream unordered stage2 token properties.
- Added `stable_token_seed()` and made `AutoVLAPredictor` default to `model.prediction_seed_mode=token_hash` with `prediction_seed_base=0`, so the same token gets the same sampling seed in single-run and sharded eval.
- Used `DictConfig.get()` for optional `reactive/non_reactive_synthetic_initial_tokens` access because `navhard_two_stage` scene filter only defines the reactive list.

## Validation Record
- Pytest:
  - `/data/miniconda/envs/autolsqv2/bin/python -m pytest tools/eval/tests/test_navhard_two_stage_sharded.py tools/eval/tests/test_run_navsimv2_epdms_standard.py -v`
  - Result: `12 passed`
- Single vs sharded smoke:
  - Single:
    - `CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla.py --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml --set eval.output_dir=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_smoke_compare_2026-03-15_seedfix/single --set eval.max_stage_one_scenarios=1 --set eval.max_stage_two_scenarios=1`
  - Prepare:
    - `/data/miniconda/envs/autolsqv2/bin/python tools/eval/prepare_navhard_two_stage_shards.py --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml --num-shards 2 --plan-dir /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_smoke_compare_2026-03-15_seedfix/sharded --set eval.max_stage_one_scenarios=1 --set eval.max_stage_two_scenarios=1`
  - Shards:
    - `CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla_shard.py --plan-dir /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_smoke_compare_2026-03-15_seedfix/sharded --shard-index 0`
    - `CUDA_VISIBLE_DEVICES=1 /data/miniconda/envs/autolsqv2/bin/python tools/eval/run_navhard_two_stage_autovla_shard.py --plan-dir /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_smoke_compare_2026-03-15_seedfix/sharded --shard-index 1`
  - Merge:
    - `/data/miniconda/envs/autolsqv2/bin/python tools/eval/merge_navhard_two_stage_shards.py --plan-dir /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_smoke_compare_2026-03-15_seedfix/sharded`
- Smoke result:
  - Single `final_extended_pdm_score = 0.4449532898712633`
  - Sharded merged `final_extended_pdm_score = 0.4449532898712633`
  - `score_diff = 0.0`

## Notes
- The dominant startup cost is still each worker reloading and indexing all `5462` synthetic scenes. The current 8-GPU path improves per-token evaluation throughput but does not eliminate this repeated initialization cost.

## Historical Comparison
- Historical single-card full run:
  - Summary: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/run_2026-03-13_09-15-08/summary.json`
  - Log: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla/navhard_epdms_full_2026-03-13_09-14-45.log`
  - Result:
    - `num_successful_scenarios = 5912`
    - `num_failed_scenarios = 0`
    - `final_extended_pdm_score = 0.13153159316313934`
    - `elapsed_sec = 16867.390713214874` (`~4.69h`)
- Current remote 8-GPU full run:
  - Summary: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47/merged/summary.json`
  - Log: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47.log`
  - Result:
    - `num_successful_scenarios = 5912`
    - `num_failed_scenarios = 0`
    - `final_extended_pdm_score = 0.14937031924317998`
    - Per-shard wall time from shard summaries/logs: about `2085s` to `2168s` (`~34.8-36.1 min`)
- Comparison conclusion:
  - Coverage is consistent: both runs completed all `5912` scenarios with `0` failures.
  - Scores are not numerically identical: `0.13153159316313934` vs `0.14937031924317998`.
  - This is not an apples-to-apples mismatch in the sharding logic itself. The old single-card full run happened before the later eval fixes:
    - stage2 token selection order stabilization
    - deterministic per-token sampling seed (`prediction_seed_mode=token_hash`)
  - After these fixes, the fresh smoke validation on current code showed single-run and sharded merged results exactly match (`score_diff = 0.0`), so the current 8-GPU path is considered aligned with the current single-run implementation, but not directly equal to the older historical single-card number.

## Fresh Single-Card Full Run Launch
- Goal:
  - Run a fresh single-card full `navhard_two_stage` eval on the current fixed code path.
  - Compare against the completed remote 8-GPU full run for:
    - score consistency
    - total wall time
- Machine:
  - `root@10.199.7.33:2289`
- GPU:
  - `CUDA_VISIBLE_DEVICES=0`
- Initial tmux attempt:
  - `navhard_single_2026_03_15_07_12_56`
  - Session did not stay alive on remote host, so actual launch was switched to `nohup` to avoid further shell-layer failure.
- Output root:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56`
- Model checkpoint:
  - `/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt`
- Launch script:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launch.sh`
- Intended command:
```bash
ssh -p 2289 root@10.199.7.33 "
tmux new-session -d -s navhard_single_2026_03_15_07_12_56 '
cd /data/liushiqi/AutoVLA &&
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
OPENSCENE_DATA_ROOT=/data/dataset/navsim \
NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
/data/miniconda/envs/autolsqv2/bin/python \
tools/eval/run_navhard_two_stage_autovla.py \
  --config /data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml \
  --set model.checkpoint_path=/data/liushiqi/AutoVLA/runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt \
  --set eval.output_dir=/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56 \
  2>&1 | tee /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launcher_stdout.log
'"
```
- Actual background launch:
```bash
ssh -p 2289 root@10.199.7.33 "
cd /data/liushiqi/AutoVLA &&
nohup bash /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launch.sh \
  > /data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/nohup_wrapper.log 2>&1 &
"
```
- Actual run status:
  - Eval PID: `2651550`
  - Run dir: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/run_2026-03-15_07-15-08`
  - Stdout log: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launcher_stdout.log`
  - Wrapper log: `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/nohup_wrapper.log`
- Expected comparison target:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47/merged/summary.json`

## Fresh Single-Card Full Run Result
- Summary:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/run_2026-03-15_07-15-08/summary.json`
- Log:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_single_2026-03-15_07-12-56/launcher_stdout.log`
- Result:
  - `num_successful_scenarios = 5912`
  - `num_failed_scenarios = 0`
  - `final_extended_pdm_score = 0.14937031924317998`
  - `elapsed_sec = 16216.207909345627` (`~4.50h`)

## Fresh Single vs 8-GPU Comparison
- Comparison target:
  - `/data/liushiqi/AutoVLA/logs/eval/navhard_two_stage_autovla_remote33_2026-03-15_06-07-47/merged/summary.json`
- Metric consistency:
  - Fresh single-card `final_extended_pdm_score = 0.14937031924317998`
  - 8-GPU merged `final_extended_pdm_score = 0.14937031924317998`
  - `score_diff = 0.0`
- Coverage consistency:
  - Both runs completed all `5912` scenarios with `0` failures.
- Time comparison:
  - Fresh single-card total wall time: `16216.207909345627s` (`~4.50h`)
  - 8-GPU shard wall time: about `2085s` to `2168s` (`~34.8-36.1 min`)
  - Using the slowest shard as end-to-end wall clock proxy, the 8-GPU run is about `7.5x` faster than the fresh single-card run.
- Conclusion:
  - On the current fixed code path, single-card and 8-GPU merged results are numerically identical.
  - The remaining difference between modes is runtime, not metric semantics.
