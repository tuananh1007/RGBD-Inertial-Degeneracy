#!/bin/bash
# Runner of the experiments: plays each sequence in real time into the estimator container.
#
# Usage:
#   run_sequences.sh OUT_ROOT DATA_DIR IMAGE "VARIANT ..." RUNS JOBS [SEQ ...]
#
#   VARIANT  names CONFIG_DIR/<VARIANT>.yaml, passed as override_path
#   RUNS     repeated playback runs per (seq, variant), indexed r1..rN
#   JOBS     parallel containers; slot k is pinned to CORES_PER_JOB cores (from CORE_OFFSET, env, default 0)
#   SEQ      default: every <seq>_d455.bag in DATA_DIR
#
#   BASE_CFG / CAM_CFG (env) override the per-sequence tum_fr*.yaml and
#   camera_tum_fr*.yaml, e.g. BASE_CFG=openloris_tf.yaml CAM_CFG=camera_openloris.yaml
#   LAUNCH / LOG_DIR_IN (env): another system with the same launch arguments, e.g. upstream FAST-LIVO2
#   (LAUNCH="depth_bridge depth_mapping.launch" LOG_DIR_IN=/ws/src/fast_livo_upstream/Log/result)
#   COPY_STATE=1 (env) also keeps Log/mat_pre.txt and mat_out.txt, the state before and after each update
#
# Writes OUT_ROOT/<VARIANT>_r<run>/<seq>_traj.txt and <seq>.log, and appends
# one row per run to OUT_ROOT/runs.csv.  Existing trajectories are skipped.
set -uo pipefail

OUT_ROOT=$1; DATA_DIR=$2; IMAGE=$3; read -r -a VARIANTS <<< "$4"; RUNS=$5; JOBS=$6
shift 6
CONFIG_DIR=${CONFIG_DIR:-$(cd "$(dirname "$0")/../config" && pwd)}
CORES_PER_JOB=${CORES_PER_JOB:-5}
TIMEOUT_S=${TIMEOUT_S:-900}
LAUNCH=${LAUNCH:-rgbd_vio mapping_d455.launch}
LOG_DIR_IN=${LOG_DIR_IN:-/ws/src/rgbd_vio/Log/result}

if [ $# -gt 0 ]; then SEQS=("$@"); else
  SEQS=(); for b in "$DATA_DIR"/*_d455.bag; do s=$(basename "$b" _d455.bag); SEQS+=("$s"); done
fi
mkdir -p "$OUT_ROOT"
RUNS_CSV="$OUT_ROOT/runs.csv"
[ -s "$RUNS_CSV" ] || echo "seq,variant,run,poses,wall_s,exit,data_dir,image_id,slot" > "$RUNS_CSV"
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}' | cut -c8-19)

QUEUE="$OUT_ROOT/.queue"; LOCK="$OUT_ROOT/.queue.lock"
: > "$QUEUE"
for r in $(seq 1 "$RUNS"); do for v in "${VARIANTS[@]}"; do for s in "${SEQS[@]}"; do
  echo "$s $v $r" >> "$QUEUE"; done; done; done
echo "[info] $(wc -l < "$QUEUE") runs queued, $JOBS job(s), image $IMAGE ($IMAGE_ID)"

next_task() {   # atomically pop the first queued task
  flock "$LOCK" bash -c "head -1 '$QUEUE'; sed -i '1d' '$QUEUE'"
}

run_one() {
  local SEQ=$1 VARIANT=$2 RUN=$3 SLOT=$4
  local OUT="$OUT_ROOT/${VARIANT}_r${RUN}"
  local TRAJ="$OUT/${SEQ}_traj.txt"
  local CPUS="$((SLOT * CORES_PER_JOB + ${CORE_OFFSET:-0}))-$((SLOT * CORES_PER_JOB + CORES_PER_JOB - 1 + ${CORE_OFFSET:-0}))"   # CORE_OFFSET (env): first core of slot 0
  mkdir -p "$OUT"
  if [ -s "$TRAJ" ]; then echo "[skip] $SEQ $VARIANT r$RUN"; return 0; fi
  local T0=$(date +%s)
  timeout "$TIMEOUT_S" docker run --rm --cpuset-cpus "$CPUS" \
    -v "$DATA_DIR":/ws/datasets/tum:ro \
    -v "$CONFIG_DIR":/ws/config_host:ro \
    -v "$OUT":/ws/results \
    --entrypoint bash "$IMAGE" -c "
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
LOG_DIR=$LOG_DIR_IN
mkdir -p \$LOG_DIR && rm -f \$LOG_DIR/*.txt
roscore >/dev/null 2>&1 &
sleep 2
roslaunch $LAUNCH rviz:=false \
  config_path:=/ws/config_host/${BASE_CFG:-tum_${SEQ%%_*}.yaml} \
  camera_path:=/ws/config_host/${CAM_CFG:-camera_tum_${SEQ%%_*}.yaml} \
  override_path:=/ws/config_host/${VARIANT}.yaml > /ws/results/${SEQ}.log 2>&1 &
sleep 3
rosbag play --clock -r 1.0 /ws/datasets/tum/${SEQ}_d455.bag > /dev/null 2>&1
# drain: a loaded machine can leave the estimator behind real time; wait until the
# trajectory stops growing (2 s without new poses, at most 120 s)
N0=-1; for i in \$(seq 1 60); do sleep 2; N=\$(cat \$LOG_DIR/*.txt 2>/dev/null | wc -l); [ \"\$N\" = \"\$N0\" ] && break; N0=\$N; done
FOUND=\$(find \$LOG_DIR -name '*.txt' -type f | head -1)
[ -n \"\$FOUND\" ] && cp \"\$FOUND\" /ws/results/${SEQ}_traj.txt
for C in \$LOG_DIR/*.cov; do [ -f \"\$C\" ] && cp \"\$C\" /ws/results/${SEQ}_cov.txt; done   # evo/cov_output_en
[ -n \"${COPY_STATE:-}\" ] && cp \$LOG_DIR/../mat_pre.txt /ws/results/${SEQ}_mat_pre.txt && cp \$LOG_DIR/../mat_out.txt /ws/results/${SEQ}_mat_out.txt
killall -9 roslaunch rosmaster > /dev/null 2>&1 || true
chown $(id -u):$(id -g) /ws/results/${SEQ}* 2>/dev/null || true
" > /dev/null 2>&1
  local EXIT=$?
  local POSES=0; [ -s "$TRAJ" ] && POSES=$(wc -l < "$TRAJ" | tr -d ' ')
  flock "$LOCK" bash -c "echo '$SEQ,$VARIANT,$RUN,$POSES,$(( $(date +%s) - T0 )),$EXIT,$DATA_DIR,$IMAGE_ID,$SLOT' >> '$RUNS_CSV'"
  echo "[done] $SEQ $VARIANT r$RUN slot$SLOT poses=$POSES exit=$EXIT"
}

worker() {
  local SLOT=$1 TASK
  while TASK=$(next_task) && [ -n "$TASK" ]; do run_one $TASK "$SLOT"; done
}

START=$(date +%s)
for k in $(seq 0 $((JOBS - 1))); do worker "$k" & done
wait
echo "[timing] total $(( ($(date +%s) - START) / 60 )) min"
rm -f "$QUEUE" "$LOCK"
