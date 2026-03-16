#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <run_pdm_score.log> [window_sec]"
  echo "Example: $0 /data/liushiqi/AutoVLA/logs/eval/navsimv2_navtest_human_gt_v2cache_2026-03-08_02-12-19_csv/run_pdm_score.log 60"
  exit 1
fi

LOG_PATH="$1"
WINDOW_SEC="${2:-60}"

if [ ! -f "$LOG_PATH" ]; then
  echo "ERROR: log file not found: $LOG_PATH"
  exit 2
fi

get_done_total() {
  perl -ne '
    if(/scenario (\d+) \/ (\d+) in thread_id=([^,]+)/){
      $done{$3}=$1;
      $total{$3}=$2;
    }
    END{
      $sum_done=0; $sum_total=0; $n=0;
      for $k (keys %total){
        $n++;
        $sum_done += $done{$k};
        $sum_total += $total{$k};
      }
      print "$sum_done $sum_total $n\n";
    }
  ' "$LOG_PATH"
}

read -r done1 total1 threads1 <<<"$(get_done_total)"
ts1="$(date -u +%s)"
sleep "$WINDOW_SEC"
read -r done2 total2 threads2 <<<"$(get_done_total)"
ts2="$(date -u +%s)"

elapsed=$((ts2 - ts1))
delta=$((done2 - done1))

if [ "$total2" -le 0 ] || [ "$threads2" -le 0 ]; then
  echo "STATUS=NO_PROGRESS_DATA done=${done2} total=${total2} threads=${threads2}"
  exit 3
fi

ratio=$(awk -v d="$done2" -v t="$total2" 'BEGIN{printf "%.4f", d/t}')
rate=$(awk -v x="$delta" -v s="$elapsed" 'BEGIN{if(s>0){printf "%.4f", x/s}else{print "0"}}')
remaining=$((total2 - done2))

if [ "$delta" -le 0 ]; then
  echo "done=${done2}/${total2} ratio=${ratio} threads=${threads2} delta=${delta}/${elapsed}s rate=${rate}/s eta=inf (delta<=0)"
  exit 0
fi

eta_sec=$(awk -v r="$remaining" -v v="$rate" 'BEGIN{if(v>0){printf "%.0f", r/v}else{print 0}}')
eta_hr=$(awk -v e="$eta_sec" 'BEGIN{printf "%.2f", e/3600}')

echo "done=${done2}/${total2} ratio=${ratio} threads=${threads2} delta=${delta}/${elapsed}s rate=${rate}/s remaining=${remaining} eta_sec=${eta_sec} eta_hr=${eta_hr}"
