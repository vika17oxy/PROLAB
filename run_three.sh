#!/usr/bin/env bash
set -uo pipefail
DATA="./data"

run_exp() {
    local name=$1; shift
    echo "=== Starting: ${name} ==="
    mkdir -p "${DATA}/${name}"
    rm -f "${DATA}/kf_log.csv" "${DATA}/ekf_log.csv" "${DATA}/pf_log.csv"
    timeout 130 docker compose run --rm experiment \
        ros2 launch prol_filters experiment.launch.py "$@" || true
    for f in kf_log.csv ekf_log.csv pf_log.csv; do
        if [ -f "${DATA}/${f}" ]; then
            mv "${DATA}/${f}" "${DATA}/${name}/${f}"
            echo "  saved ${name}/${f}"
        else
            echo "  MISSING ${f}"
        fi
    done
    echo "=== Done: ${name} ==="
    echo ""
}

run_exp 01_baseline      trajectory:=circle duration:=90.0
run_exp 08_delay_100ms   trajectory:=circle duration:=90.0 delay_ms:=100.0
run_exp 09_delay_500ms   trajectory:=circle duration:=90.0 delay_ms:=500.0

echo "ALL DONE"
