#!/usr/bin/env bash
set -u

INTERVAL_SEC="${1:-300}"

NAVTRAIN_LOG="/data/liushiqi/AutoVLA/logs/eval/metric_cache_v2_navtrain_full_2026-03-07_15-49-21.log"
NAVTRAIN_CACHE_DIR="/data/dataset/navsim/metric_cache_v2/navtrain_full_2026-03-07_15-49-21"

HUMAN_LOG="/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19.log"
HUMAN_RUNLOG="/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19_csv/run_pdm_score.log"

IDM_LOG="/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22.log"
IDM_RUNLOG="/data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_idm_v2cache_2026-03-08_02-12-22_csv/run_pdm_score.log"

get_done_total() {
  local runlog="$1"
  if [ ! -f "$runlog" ]; then
    echo "NA/NA"
    return 0
  fi
  perl -ne '
    if(/scenario (\d+) \/ (\d+) in thread_id=([^,]+)/){
      $done{$3}=$1;
      $total{$3}=$2;
    }
    END{
      $sum_done=0; $sum_total=0;
      for $k (keys %total){
        $sum_done += $done{$k};
        $sum_total += $total{$k};
      }
      if($sum_total > 0){ print "$sum_done/$sum_total\n"; }
      else { print "NA/NA\n"; }
    }
  ' "$runlog" 2>/dev/null
}

is_navtrain_done() {
  rg -q "Completed dataset caching! All .* cached successfully" "$NAVTRAIN_LOG" 2>/dev/null
}

is_eval_done() {
  local log_path="$1"
  rg -q "Final average score of valid results|Finished running evaluation" "$log_path" 2>/dev/null
}

get_cache_count() {
  timeout 20s rg --files "$NAVTRAIN_CACHE_DIR" 2>/dev/null | rg 'metric_cache\.pkl$' | wc -l
}

echo "monitor_start=$(date -u +'%Y-%m-%d %H:%M:%S UTC') interval_sec=${INTERVAL_SEC}"
echo "paths navtrain_log=${NAVTRAIN_LOG} human_log=${HUMAN_LOG} idm_log=${IDM_LOG}"

while true; do
  ts="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"

  navtrain_done=0
  human_done=0
  idm_done=0

  if is_navtrain_done; then navtrain_done=1; fi
  if is_eval_done "$HUMAN_LOG"; then human_done=1; fi
  if is_eval_done "$IDM_LOG"; then idm_done=1; fi

  navtrain_pids="$(pgrep -f 'run_metric_caching.py train_test_split=navtrain' 2>/dev/null || true)"
  human_pids="$(pgrep -f 'run_pdm_score.py train_test_split=navtest agent=human_agent' 2>/dev/null || true)"
  idm_pids="$(pgrep -f 'run_pdm_score.py train_test_split=navtest agent=constant_velocity_agent' 2>/dev/null || true)"

  navtrain_workers="$(echo "$navtrain_pids" | awk 'NF>0{c++} END{print c+0}')"
  human_workers="$(echo "$human_pids" | awk 'NF>0{c++} END{print c+0}')"
  idm_workers="$(echo "$idm_pids" | awk 'NF>0{c++} END{print c+0}')"

  navtrain_first="$(echo "$navtrain_pids" | head -n 1)"
  human_first="$(echo "$human_pids" | head -n 1)"
  idm_first="$(echo "$idm_pids" | head -n 1)"

  cache_cnt="$(get_cache_count)"
  human_prog="$(get_done_total "$HUMAN_RUNLOG")"
  idm_prog="$(get_done_total "$IDM_RUNLOG")"

  echo "[${ts}] navtrain_done=${navtrain_done} cache_count=${cache_cnt} human_done=${human_done} human_progress=${human_prog} idm_done=${idm_done} idm_progress=${idm_prog}"
  echo "[${ts}] workers navtrain=${navtrain_workers} human=${human_workers} idm=${idm_workers} sample_pid navtrain=${navtrain_first:-NA} human=${human_first:-NA} idm=${idm_first:-NA}"

  if [ "$navtrain_done" -eq 1 ] && [ "$human_done" -eq 1 ] && [ "$idm_done" -eq 1 ]; then
    echo "all_done=$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    break
  fi

  sleep "$INTERVAL_SEC"
done
