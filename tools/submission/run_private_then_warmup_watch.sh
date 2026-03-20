#!/usr/bin/env bash
set -euo pipefail

cd /data/liushiqi/AutoVLA/.worktrees/navsimv2-submission

PRIVATE_RUN_ID="2026-03-18_private8pose_sharded"
PRIVATE_OUT="/data/liushiqi/AutoVLA/logs/submission/navsimv2_autovla_private_rft_step6000_test_2026-03-18_8pose_sharded"
PRIVATE_VALID="/data/liushiqi/AutoVLA/logs/submission_validation/navsimv2_autovla_private_rft_step6000_test_2026-03-18_8pose"

WARMUP_OUT="/data/liushiqi/AutoVLA/logs/submission/navsimv2_autovla_warmup_rft_step6000_test_2026-03-18_8pose_auto"
WARMUP_VALID="/data/liushiqi/AutoVLA/logs/submission_validation/navsimv2_autovla_warmup_rft_step6000_test_2026-03-18_8pose_auto"

CKPT="/data/liushiqi/AutoVLA/runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt"
METRIC_CACHE="/data/dataset/navsim/metric_cache_v2/warmup_two_stage_full_local_2026-03-17"

while ps -eo args | grep "run_navsimv2_autovla_submission.py" | grep "${PRIVATE_RUN_ID}" | grep -v grep >/dev/null; do
  sleep 30
done

/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_autovla_submission.py \
  --config config/submission/navsimv2_autovla_private_test_hard.yaml \
  --set run.run_id="${PRIVATE_RUN_ID}" \
  --set run.num_shards=8 \
  --set run.merge_only=true \
  --set model.checkpoint_path="${CKPT}" \
  --set submission.team_name=test_rft_step6000_8pose_private \
  --set submission.authors=test \
  --set submission.email=test@example.com \
  --set submission.institution=AutoVLA \
  --set submission.country=CN \
  --set run.output_dir="${PRIVATE_OUT}" \
  --set run.log_path="${PRIVATE_OUT}/merge.log"

/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_submission_validation.py \
  --config config/submission/navsimv2_autovla_private_test_hard.yaml \
  --submission-file "${PRIVATE_OUT}/run_${PRIVATE_RUN_ID}/submission.pkl" \
  --set validation.output_dir="${PRIVATE_VALID}" \
  --set run.log_path="${PRIVATE_VALID}/validation.log"

CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_autovla_submission.py \
  --config config/submission/navsimv2_autovla_warmup.yaml \
  --set model.checkpoint_path="${CKPT}" \
  --set submission.team_name=test_rft_step6000_8pose_warmup \
  --set submission.authors=test \
  --set submission.email=test@example.com \
  --set submission.institution=AutoVLA \
  --set submission.country=CN \
  --set data.metric_cache_path="${METRIC_CACHE}" \
  --set run.output_dir="${WARMUP_OUT}" \
  --set run.log_path="${WARMUP_OUT}/run_navsimv2_autovla_submission.log"

WARMUP_RUN="$(find "${WARMUP_OUT}" -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1)"

/data/miniconda/envs/autolsqv2/bin/python tools/submission/run_navsimv2_submission_validation.py \
  --config config/submission/navsimv2_autovla_warmup.yaml \
  --submission-file "${WARMUP_RUN}/submission.pkl" \
  --set data.metric_cache_path="${METRIC_CACHE}" \
  --set validation.output_dir="${WARMUP_VALID}" \
  --set run.log_path="${WARMUP_VALID}/validation.log"
