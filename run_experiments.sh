#!/usr/bin/env bash
# run_experiments.sh
#
# Builds the Docker image once, then runs all mandatory experiments in sequence.
# Each experiment auto-exits after `duration` seconds (simulator triggers shutdown).
# CSVs are saved to ./data/<experiment_name>/ on the host.
#
# Usage:
#   bash run_experiments.sh
#
# Requirements:
#   - Docker + Docker Compose v2
#   - ./data/ will be created automatically

set -euo pipefail

DATA="./data"
IMAGE="prol_filters:humble"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

mkdir -p "$DATA"

# ── Build image (once) ───────────────────────────────────────────────────────
info "Building Docker image..."
docker compose build experiment
info "Build complete."

# ── run_experiment <name> [launch args...] ───────────────────────────────────
run_experiment() {
    local name=$1
    shift
    local launch_args="$*"

    info "Starting experiment: ${name}"
    mkdir -p "${DATA}/${name}"

    # Remove any leftover CSVs from a previous partial run
    rm -f "${DATA}/kf_log.csv" "${DATA}/ekf_log.csv" "${DATA}/pf_log.csv"

    # Run the experiment container (blocks until simulator exits, timeout=120s safety net)
    timeout 120 docker compose run --rm experiment \
        ros2 launch prol_filters experiment.launch.py ${launch_args} || true

    # Move CSVs into named subdirectory
    for f in kf_log.csv ekf_log.csv pf_log.csv; do
        if [ -f "${DATA}/${f}" ]; then
            mv "${DATA}/${f}" "${DATA}/${name}/${f}"
            info "  Saved ${name}/${f}"
        else
            warn "  ${f} not found — node may not have logged data"
        fi
    done

    info "Experiment '${name}' complete."
    echo ""
}

# ════════════════════════════════════════════════════════════════════════════
# BASELINE
# ════════════════════════════════════════════════════════════════════════════

# 1. Baseline — circle, default noise params
run_experiment "01_baseline" \
    trajectory:=circle duration:=90.0

# ════════════════════════════════════════════════════════════════════════════
# PROCESS NOISE (Q) VARIATION
# ════════════════════════════════════════════════════════════════════════════

# 4. Low process noise — filter trusts motion model strongly
run_experiment "04_q_low" \
    trajectory:=circle duration:=90.0 q_xy:=0.0001 q_theta:=0.00005

# 5. High process noise — filter trusts motion model weakly
run_experiment "05_q_high" \
    trajectory:=circle duration:=90.0 q_xy:=0.1 q_theta:=0.05

# ════════════════════════════════════════════════════════════════════════════
# MEASUREMENT NOISE (R) VARIATION
# ════════════════════════════════════════════════════════════════════════════

# 6. Low measurement noise — filter trusts landmark range strongly
run_experiment "06_r_low" \
    trajectory:=circle duration:=90.0 r_landmark:=0.005

# 7. High measurement noise — filter trusts landmark range weakly
run_experiment "07_r_high" \
    trajectory:=circle duration:=90.0 r_landmark:=0.5

# ════════════════════════════════════════════════════════════════════════════
# DELAY EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

# 8. Time delay: 100 ms
run_experiment "08_delay_100ms" \
    trajectory:=circle duration:=90.0 delay_ms:=100.0

# 9. Time delay: 500 ms
run_experiment "09_delay_500ms" \
    trajectory:=circle duration:=90.0 delay_ms:=500.0

# ════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ════════════════════════════════════════════════════════════════════════════

info "Running post-processing: metrics + rmse_bar for all experiments..."
for exp_dir in "${DATA}"/*/; do
    exp_name=$(basename "${exp_dir}")
    docker compose --profile evaluate run --rm evaluate \
        python3 /ros2_ws/install/prol_filters/lib/prol_filters/evaluate_filters.py \
        --data-dir "/data/${exp_name}" 2>/dev/null || \
    warn "  evaluate_filters.py failed for ${exp_name} — run manually"
done

info "Running post-processing: full plot suite for baseline (includes Q/R/delay variation)..."
mkdir -p "${DATA}/01_baseline/plots"
docker compose --profile plot run --rm plot \
    python3 /ros2_ws/install/prol_filters/lib/prol_filters/plot_results.py \
    --data-dir /data/01_baseline \
    --out-dir /data/01_baseline/plots \
    --root-dir /data 2>/dev/null || \
warn "plot_results.py failed — run manually after experiments"

info "Running post-processing: trajectory + position error plots for delay experiments..."
for exp_name in "08_delay_100ms" "09_delay_500ms"; do
    if [ -d "${DATA}/${exp_name}" ]; then
        mkdir -p "${DATA}/${exp_name}/plots"
        # No --root-dir here: only generate per-experiment plots (trajectories, pos_error, rmse, cov)
        # Cross-experiment Q/R/delay variation plots live only in 01_baseline/plots/
        docker compose --profile plot run --rm plot \
            python3 /ros2_ws/install/prol_filters/lib/prol_filters/plot_results.py \
            --data-dir "/data/${exp_name}" \
            --out-dir "/data/${exp_name}/plots" 2>/dev/null || \
        warn "  plot_results.py failed for ${exp_name}"
    fi
done

info "════════════════════════════════════════════════════════"
info "All experiments complete. Results in: ${DATA}/"
info "  CSVs:    ${DATA}/<experiment_name>/*.csv"
info "  Summary: ${DATA}/<experiment_name>/evaluation_summary.csv"
info "  Plots:   ${DATA}/<experiment_name>/plots/*.png"
info "  Main plots (all variations): ${DATA}/01_baseline/plots/"
info "════════════════════════════════════════════════════════"
