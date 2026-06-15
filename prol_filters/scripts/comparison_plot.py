#!/usr/bin/env python3
"""
comparison_plot.py — KF vs EKF vs PF from logged CSVs (real C++ filter output).

Reads kf_log.csv / ekf_log.csv / pf_log.csv from --data-dir and produces a
2-panel figure (filter_comparison.png):
  left  : trajectory overlay (ground truth vs each filter estimate + landmark)
  right : 2D position error over time for each filter

Usage:  python3 comparison_plot.py --data-dir /data [--landmark 1.8 3.0]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"KF": "#e74c3c", "EKF": "#2980b9", "PF": "#27ae60"}


def load(path):
    if not os.path.isfile(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    if d is None or d.size == 0 or d.dtype.names is None:
        return None
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def perr(d):
    return np.sqrt((d["x"] - d["gt_x"]) ** 2 + (d["y"] - d["gt_y"]) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--landmark", nargs=2, type=float, default=[1.8, 3.0])
    a = ap.parse_args()
    dd = a.data_dir

    filt = {"KF": load(os.path.join(dd, "kf_log.csv")),
            "EKF": load(os.path.join(dd, "ekf_log.csv")),
            "PF": load(os.path.join(dd, "pf_log.csv"))}

    fig, (axt, axe) = plt.subplots(1, 2, figsize=(14, 5.6))

    # ── Trajectory overlay ────────────────────────────────────────────────────
    gt = next(d for d in filt.values() if d is not None)
    axt.plot(gt["gt_x"], gt["gt_y"], "-", color="black", lw=2.4, label="Ground Truth", zorder=5)
    for name, d in filt.items():
        if d is None:
            continue
        axt.plot(d["x"], d["y"], "--", color=C[name], lw=1.6, alpha=0.9, label=f"{name} estimate")
    axt.plot(gt["gt_x"][0], gt["gt_y"][0], "o", color="0.4", ms=10, label="Start", zorder=6)
    axt.plot(gt["gt_x"][-1], gt["gt_y"][-1], "o", color="orange", ms=10, label="True End", zorder=6)
    axt.plot(*a.landmark, "*", color="#1f77b4", ms=20, label="Known Landmark", zorder=6)
    axt.set_aspect("equal"); axt.grid(True, alpha=0.3)
    axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]")
    axt.set_title("Trajectory: Ground Truth vs Filter Estimates", fontweight="bold")
    axt.legend(loc="best", fontsize=8)

    # ── Error over time ───────────────────────────────────────────────────────
    for name, d in filt.items():
        if d is None:
            continue
        t = d["time_s"] - d["time_s"][0]
        e = perr(d)
        axe.plot(t, e, color=C[name], lw=1.4,
                 label=f"{name}  (RMSE {np.sqrt(np.mean(e**2)):.3f} m)")
    axe.set_xlabel("time [s]"); axe.set_ylabel("2D position error [m]")
    axe.set_title("Position Error over Time", fontweight="bold")
    axe.grid(True, alpha=0.3)
    axe.legend(loc="best", fontsize=9)

    fig.suptitle("KF vs EKF vs PF — S-trajectory (real C++ filter logs)", fontsize=14, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.join(dd, "plots"), exist_ok=True)
    out = os.path.join(dd, "plots", "filter_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
