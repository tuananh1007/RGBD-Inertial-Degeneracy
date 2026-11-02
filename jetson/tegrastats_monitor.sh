#!/bin/bash
# tegrastats_monitor.sh — sample Jetson power/temp/clocks during a benchmark.
# Parses `tegrastats --interval 100` output into a flat CSV with per-sample
# RAM, CPU freq, GPU freq, total power (mW), and temp in °C.
#
# Start in background before the VIO run, stop after:
#   ./tegrastats_monitor.sh start /tmp/jetson_benchmark/power.csv
#   ./benchmark_timing.sh ...
#   ./tegrastats_monitor.sh stop
set -uo pipefail

ACTION="${1:-start}"
OUT="${2:-/tmp/jetson_tegrastats.csv}"
PIDFILE=/tmp/tegrastats_monitor.pid

case "$ACTION" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
      echo "[tegrastats] already running (pid=$(cat $PIDFILE))"; exit 0
    fi
    if ! command -v tegrastats >/dev/null 2>&1; then
      echo "[tegrastats] ERR: not a Jetson (tegrastats missing)" >&2; exit 2
    fi
    echo "ts_ms,ram_used_mb,ram_total_mb,cpu_pct,gpu_freq_mhz,soc_temp_c,power_soc_mw,power_cpu_mw,power_gpu_mw,power_total_mw" > "$OUT"
    nohup bash -c "
      T0=\$(date +%s%3N)
      tegrastats --interval 100 | while read line; do
        ts=\$(( \$(date +%s%3N) - T0 ))
        # RAM: '4524/15537MB'
        ram=\$(echo \"\$line\" | grep -oE 'RAM [0-9]+/[0-9]+MB' | head -1 | sed 's/RAM //; s/MB//')
        ram_used=\${ram%%/*}; ram_total=\${ram##*/}
        # CPU util: take mean of per-core pct
        cpu_pct=\$(echo \"\$line\" | grep -oE 'CPU \[[^]]+\]' | head -1 | grep -oE '[0-9]+%' | sed 's/%//' | awk '{s+=\$1; n++} END{if(n>0) printf \"%.0f\", s/n; else print 0}')
        # GPU freq: 'GR3D_FREQ 624%@624'
        gpu_mhz=\$(echo \"\$line\" | grep -oE 'GR3D_FREQ [0-9]+%@[0-9]+' | grep -oE '@[0-9]+' | sed 's/@//')
        # SoC temp
        soc_c=\$(echo \"\$line\" | grep -oE 'soc0@[0-9.]+C' | grep -oE '[0-9.]+' | head -1)
        # Power: 'VDD_IN 8200mW/8200mW'
        p_soc=\$(echo \"\$line\" | grep -oE 'VDD_SOC [0-9]+mW' | grep -oE '[0-9]+')
        p_cpu=\$(echo \"\$line\" | grep -oE 'VDD_CPU[_A-Z0-9]* [0-9]+mW' | head -1 | grep -oE '[0-9]+')
        p_gpu=\$(echo \"\$line\" | grep -oE 'VDD_GPU[_A-Z0-9]* [0-9]+mW' | head -1 | grep -oE '[0-9]+')
        p_tot=\$(echo \"\$line\" | grep -oE 'VDD_IN [0-9]+mW' | grep -oE '[0-9]+')
        # Alt key names on older JetPacks
        [ -z \"\$p_tot\" ] && p_tot=\$(echo \"\$line\" | grep -oE 'POM_5V_IN [0-9]+/' | grep -oE '[0-9]+')
        echo \"\$ts,\${ram_used:-0},\${ram_total:-0},\${cpu_pct:-0},\${gpu_mhz:-0},\${soc_c:-0},\${p_soc:-0},\${p_cpu:-0},\${p_gpu:-0},\${p_tot:-0}\" >> $OUT
      done
    " >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "[tegrastats] started (pid=$!) -> $OUT"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      pid=$(cat "$PIDFILE")
      pkill -P "$pid" tegrastats 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
      rm -f "$PIDFILE"
      echo "[tegrastats] stopped"
    fi
    ;;
  summary)
    if [ -z "${2:-}" ] || [ ! -f "$OUT" ]; then
      echo "Usage: $0 summary <csv>"; exit 1
    fi
    python3 - <<PYEOF
import csv, statistics, sys
with open('$OUT') as f:
    r = csv.DictReader(f)
    rows = list(r)
if not rows:
    print("no samples"); sys.exit(0)
p_tot = [int(x['power_total_mw']) for x in rows if x['power_total_mw'].isdigit() and int(x['power_total_mw'])>0]
p_cpu = [int(x['power_cpu_mw']) for x in rows if x['power_cpu_mw'].isdigit()]
p_gpu = [int(x['power_gpu_mw']) for x in rows if x['power_gpu_mw'].isdigit()]
t    = [float(x['soc_temp_c']) for x in rows if x['soc_temp_c'] not in ('0','')]
ram  = [int(x['ram_used_mb']) for x in rows if x['ram_used_mb'].isdigit()]
def stats(v, name):
    if not v: return
    print(f"  {name:<18}  mean={statistics.mean(v):7.1f}  p95={sorted(v)[int(0.95*len(v))]:7.1f}  max={max(v):7.1f}  (n={len(v)})")
print("Samples:", len(rows))
stats(p_tot, "power_total (mW)")
stats(p_cpu, "power_cpu   (mW)")
stats(p_gpu, "power_gpu   (mW)")
stats(t,     "soc_temp    (C)")
stats(ram,   "ram_used    (MB)")
PYEOF
    ;;
  *)
    echo "Usage: $0 {start <csv> | stop | summary <csv>}"
    exit 1 ;;
esac
