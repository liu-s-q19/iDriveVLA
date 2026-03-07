#!/bin/bash
export PYTHONPATH=/data/liushiqi/AutoVLA/navsim:$PYTHONPATH

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/data/liushiqi/AutoVLA/runs/sft/latest.ckpt"
NAVSIM_DEVKIT_ROOT="/data/liushiqi/AutoVLA/navsim"
CACHE_PATH="/data/dataset/navsim/metric_cache/navtest"
JSON_DATA_PATH="/data/dataset/navsim/preprocessed/navtest_nocot"
SENSOR_DATA_PATH="/data/dataset/navsim/sensor_blobs/test"
CONFIG_PATH="/data/liushiqi/AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
LORA=false


CUDA_VISIBLE_DEVICES=0 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  agent=autovla_agent \
  +agent.config_path="$CONFIG_PATH" \
  +agent.checkpoint_path="$CHECKPOINT" \
  +agent.sensor_data_path="$SENSOR_DATA_PATH" \
  +agent.lora_conf.use_lora=$LORA \
  metric_cache_path=$CACHE_PATH \
  json_data_path=$JSON_DATA_PATH \
  experiment_name=autovla_agent\
