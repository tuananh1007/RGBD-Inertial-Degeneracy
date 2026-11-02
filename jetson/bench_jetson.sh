#!/bin/bash
# Jetson benchmark: Ours (s_ours_lin), Baseline (B), and the simpler depth-stratified
# sampler in the same fusion pipeline (lin_noitps), one real-time run per sequence on the same 8 sequences as the
# desktop timing.  Per run it records the per-frame processing log (summarize_timing.py), the sensor-to-pose latency
# and queue backlog (latency_logger.py), the node's resident memory (mem_logger.sh), tegrastats power/RAM/temperature,
# the trajectory, and a runs.csv row per dataset so that evaluate_runs.py scores ATE/RPE on the desktop:
#   python3 scripts/evaluate_runs.py OUT_DIR/PLATFORM/utmm  data/utmm   (etc., see README)
#
#   BAGS_DIR=/path/to/bags OUT_DIR=$PWD/timing PLATFORM=orin_nx_16gb bash bench_jetson.sh
#
# BAGS_DIR holds <seq>_d455.bag for the sequences below (copy them from the desktop).  Config files are read
# from this checkout (config/), so config changes need no image rebuild.  NO_CLOCKS=1 skips
# nvpmodel/jetson_clocks (by default the board is set to its maximum-performance mode first).  RUNS=n repeats.
# The image is CPU-only (Ubuntu 20.04 + ROS Noetic), so it runs without the NVIDIA runtime on JetPack 5 or 6 hosts.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/.." && pwd)
BAGS_DIR=${BAGS_DIR:?set BAGS_DIR}; OUT_DIR=${OUT_DIR:-$HOME/timing}; PLATFORM=${PLATFORM:-jetson}
IMAGE=${IMAGE:-rgbd-inertial:v1.1}; VARIANTS=${VARIANTS:-"s_ours_lin B lin_noitps"}; RUNS=${RUNS:-1}
OUT=$OUT_DIR/$PLATFORM; mkdir -p "$OUT"
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}' | cut -c8-19)
# dataset  sequence  base config  camera config
SEQS="utmm ego-centric-1 utmm.yaml camera_utmm.yaml
utmm square-1 utmm.yaml camera_utmm.yaml
openloris office1-2 openloris_tf.yaml camera_openloris.yaml
openloris office1-4 openloris_tf.yaml camera_openloris.yaml
fastlivo2_retail Retail_Street fastlivo2_retail.yaml camera_fl2_retail.yaml
fastlivo2_hku CBD_Building_02 fastlivo2_hku.yaml camera_fl2_hku.yaml
site_a outside_1 field_site_a.yaml camera_field_d455.yaml
site_b Bolter_2 field_site_b.yaml camera_field_d455.yaml"

if [ "${NO_CLOCKS:-0}" != 1 ] && command -v nvpmodel > /dev/null; then
  # the maximum-performance mode by name (MAXN, or MAXN_SUPER on JetPack 6.2): its ID differs between JetPacks and modules
  M=$(grep -oE 'POWER_MODEL ID=[0-9]+ NAME=MAXN[A-Z_]*' /etc/nvpmodel.conf 2>/dev/null | head -1 | grep -oE 'ID=[0-9]+' | cut -d= -f2)
  sudo nvpmodel -m "${M:-0}" && sudo jetson_clocks && echo "[bench] nvpmodel ${M:-0} + jetson_clocks"
fi
{ echo "date: $(date -Is)"; echo "platform: $PLATFORM"; echo "image: $IMAGE ($IMAGE_ID)"; echo "git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) $(git -C "$REPO" status --short 2>/dev/null | wc -l) modified files"
  echo "module: $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"; echo "l4t: $(head -1 /etc/nv_tegra_release 2>/dev/null)"
  echo "kernel: $(uname -r)"; echo "nvpmodel: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"; echo "jetson_clocks: $(sudo jetson_clocks --show 2>/dev/null | grep -c MaxFreq) cores at max"
  echo "container_os: $(docker run --rm "$IMAGE" sh -c '. /etc/os-release; echo $PRETTY_NAME') ROS $(docker run --rm "$IMAGE" bash -c 'echo $ROS_DISTRO')"
  echo "cpus: $(nproc)"; echo "mem_total_mb: $(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo)"; } > "$OUT/platform.txt"

echo "$SEQS" | while read -r DS SEQ BASE CAM; do
  [ -f "$BAGS_DIR/${SEQ}_d455.bag" ] || { echo "[skip] $SEQ (no bag)"; continue; }
  DT=$(awk -F'"' '/depth_topic:/ {print $2; exit}' "$REPO/config/$BASE")   # the Site A bags use /camera/camera/...
  RUNS_CSV="$OUT/$DS/runs.csv"; mkdir -p "$OUT/$DS"
  [ -s "$RUNS_CSV" ] || echo "seq,variant,run,poses,wall_s,exit,data_dir,image_id,slot" > "$RUNS_CSV"
  for R in $(seq 1 "$RUNS"); do for V in $VARIANTS; do
    D="$OUT/$DS/${V}_r$R"; mkdir -p "$D"
    [ -s "$D/${SEQ}_traj.txt" ] && { echo "[skip] $SEQ $V r$R"; continue; }
    bash "$HERE/tegrastats_monitor.sh" start "$D/$SEQ.power.csv" > /dev/null
    T0=$(date +%s)
    docker run --rm --network host \
      -v "$BAGS_DIR":/bags:ro -v "$REPO/config":/ws/config_host:ro -v "$HERE":/bench:ro -v "$D":/out \
      "$IMAGE" bash -c "
source /opt/ros/noetic/setup.bash; source /ws/devel/setup.bash
mkdir -p /ws/src/rgbd_vio/Log/result && rm -f /ws/src/rgbd_vio/Log/result/*.txt
roscore > /dev/null 2>&1 & sleep 3
roslaunch rgbd_vio mapping_d455.launch rviz:=false config_path:=/ws/config_host/$BASE \
  camera_path:=/ws/config_host/$CAM override_path:=/ws/config_host/$V.yaml > /out/$SEQ.log 2>&1 &
sleep 2
python3 /bench/latency_logger.py /out/$SEQ.latency.csv --depth $DT > /out/$SEQ.latency.log 2>&1 &
bash /bench/mem_logger.sh /out/$SEQ.mem.csv rgbd_mapping_node &
sleep 2
rosbag play --clock -r 1.0 /bags/${SEQ}_d455.bag > /dev/null 2>&1
N0=-1
for i in \$(seq 1 60); do sleep 2; N=\$(cat /ws/src/rgbd_vio/Log/result/*.txt 2>/dev/null | wc -l); [ \"\$N\" = \"\$N0\" ] && break; N0=\$N; done
cp /ws/src/rgbd_vio/Log/result/*.txt /out/${SEQ}_traj.txt 2>/dev/null
killall -9 roslaunch rosmaster > /dev/null 2>&1 || true
chown $(id -u):$(id -g) /out/${SEQ}* 2>/dev/null || true"
    EXIT=$?
    bash "$HERE/tegrastats_monitor.sh" stop > /dev/null
    POSES=0; [ -s "$D/${SEQ}_traj.txt" ] && POSES=$(wc -l < "$D/${SEQ}_traj.txt" | tr -d ' ')
    echo "$SEQ,$V,$R,$POSES,$(( $(date +%s) - T0 )),$EXIT,$BAGS_DIR,$IMAGE_ID,0" >> "$RUNS_CSV"
    P=$(awk -F, 'NR>1 {s+=$10; n++; if($10>m) m=$10} END{if(n) printf "mean %.1f W, peak %.1f W", s/n/1000, m/1000}' "$D/$SEQ.power.csv")
    L=$(awk -F, 'NR>1 && $4!="" {print $4}' "$D/$SEQ.latency.csv" 2>/dev/null | sort -g | awk '{a[NR]=$1} END{if (NR) printf "latency p50 %.0f ms, p99 %.0f ms", 1000*a[int(NR/2)+1], 1000*a[int(0.99*NR)+1]; else printf "latency n/a"}')   # no asort: mawk
    echo "[done] $DS $SEQ $V r$R  poses=$POSES  power: $P  $L"
  done; done
done
docker run --rm -v "$REPO/scripts":/scripts:ro -v "$OUT":/t "$IMAGE" python3 /scripts/summarize_timing.py /t | tee "$OUT/timing_summary.txt"
echo "[bench] done; send back $OUT (timing_summary.txt, platform.txt, and the per-run folders) for ATE scoring on the desktop"
