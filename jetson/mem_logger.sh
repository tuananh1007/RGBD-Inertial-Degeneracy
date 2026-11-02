#!/bin/bash
# Resident memory of the estimator node, one sample per second: t_s,rss_mb  (inside the container; exact process name,
# no pattern matching).  Usage: mem_logger.sh OUT.csv [rgbd_mapping_node]
OUT=$1; NAME=${2:-rgbd_mapping_node}; echo "t_s,rss_mb" > "$OUT"; T0=$(date +%s)
while true; do
  PID=$(pidof -s "$NAME" 2>/dev/null) || PID=""
  [ -n "$PID" ] && echo "$(( $(date +%s) - T0 )),$(awk '{printf "%.1f", $1/1024}' <(ps -o rss= -p "$PID"))" >> "$OUT"
  sleep 1
done
