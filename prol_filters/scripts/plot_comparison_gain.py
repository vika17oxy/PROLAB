#!/usr/bin/env python3
"""
plot_comparison_gain.py — regenerate filter_comparison.png and kalman_gain.png
in the plotting style of Elias Bitsch's PRO-LAB analyze_results.py (filter colour
scheme, legend placed outside the axes, clean rc, landmark-update shading). This
is an independent reimplementation of those conventions — own code — running on
PROL_Vika's own KF/EKF/PF CSV logs.

Usage:  python3 plot_comparison_gain.py --data-dir /data/10_scurve
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Elias-style filter colours (analyze_results.py COLOURS).
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red", "GT": "k"}

plt.rcParams.update({
    "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "0.4", "axes.linewidth": 1.0,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "legend.framealpha": 0.92, "legend.edgecolor": "0.8",
    "lines.linewidth": 1.8,
})


def load(p):
    if not os.path.isfile(p):
        return None
    d = np.genfromtxt(p, delimiter=",", names=True, invalid_raise=False)
    if d is None or d.size == 0 or d.dtype.names is None:
        return None
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def legend_out(ax, **kw):
    # Place the legend outside the axes (upper-left of the right margin), as in
    # Elias's _legend_right; savefig bbox='tight' keeps it in the file.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
              frameon=True, framealpha=0.92, **kw)


def perr(d):
    return np.hypot(d["x"] - d["gt_x"], d["y"] - d["gt_y"])


def t0(d):
    return d["time_s"] - d["time_s"][0]


def plot_filter_comparison(filt, out):
    fig, (axt, axe) = plt.subplots(1, 2, figsize=(14, 5.6))
    gt = next(d for d in filt.values() if d is not None)
    axt.plot(gt["gt_x"], gt["gt_y"], "-", color=C["GT"], lw=2.4,
             label="Ground Truth", zorder=5)
    for n, d in filt.items():
        if d is None:
            continue
        axt.plot(d["x"], d["y"], "--", color=C[n], lw=1.5, alpha=0.9,
                 label=f"{n} estimate")
    axt.set_aspect("equal"); axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]")
    axt.set_title("Trajectory: ground truth vs filters"); legend_out(axt, fontsize=9)

    for n, d in filt.items():
        if d is None:
            continue
        e = perr(d)
        axe.plot(t0(d), e, color=C[n], lw=1.4,
                 label=f"{n}  (RMSE {np.sqrt(np.mean(e ** 2)):.3f} m)")
    axe.set_xlabel("time [s]"); axe.set_ylabel("2D position error [m]")
    axe.set_title("Position error over time"); legend_out(axe, fontsize=10)

    fig.suptitle("KF vs EKF vs PF — filter comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("saved", out)


def plot_trajectories(filt, out):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    gt = next(d for d in filt.values() if d is not None)
    ax.plot(gt["gt_x"], gt["gt_y"], "-", color=C["GT"], lw=2.6,
            label="Ground Truth", zorder=5)
    for n, d in filt.items():
        if d is None:
            continue
        ax.plot(d["x"], d["y"], "--", color=C[n], lw=1.4, alpha=0.9, label=n)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison"); legend_out(ax, fontsize=11)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("saved", out)


def plot_kalman_gain(kf, ekf, out):
    sets = [(d, c, l) for d, c, l in [(kf, C["KF"], "KF"), (ekf, C["EKF"], "EKF")]
            if d and all(k in d for k in ("time_s", "had_update", "k00", "k21"))]
    if not sets:
        print("skip kalman_gain (no gain columns)")
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    rows = [("k00", "k01", r"$K_x$"), ("k10", "k11", r"$K_y$"),
            ("k20", "k21", r"$K_\theta$")]
    d0 = sets[0][0]; tt = t0(d0); upd = d0["had_update"] > 0.5
    for ax, (cr, cb, lbl) in zip(axes, rows):
        for d, col, nm in sets:
            t = t0(d); m = d["had_update"] > 0.5
            ax.plot(t, np.where(m, d[cr], np.nan), color=col, lw=1.8, ls="-",
                    alpha=0.85, label=f"{nm} range")
            ax.plot(t, np.where(m, d[cb], np.nan), color=col, lw=1.8, ls="--",
                    alpha=0.85, label=f"{nm} bearing")
        # Shade landmark-update windows (gold), like the original gain plot.
        inr = False; ts = 0.0
        for i, u in enumerate(upd):
            if u and not inr:
                ts, inr = tt[i], True
            elif not u and inr:
                ax.axvspan(ts, tt[i], alpha=0.06, color="gold", zorder=0); inr = False
        if inr:
            ax.axvspan(ts, tt[-1], alpha=0.06, color="gold", zorder=0)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel(f"{lbl} gain"); legend_out(ax, fontsize=9)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Kalman gain over time  (landmark-update windows shaded)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="/data/10_scurve")
    a = ap.parse_args(); D = a.data_dir
    kf = load(os.path.join(D, "kf_log.csv"))
    ekf = load(os.path.join(D, "ekf_log.csv"))
    pf = load(os.path.join(D, "pf_log.csv"))
    os.makedirs(os.path.join(D, "plots"), exist_ok=True)
    plot_trajectories({"KF": kf, "EKF": ekf, "PF": pf},
                      os.path.join(D, "plots", "trajectories.png"))
    plot_filter_comparison({"KF": kf, "EKF": ekf, "PF": pf},
                           os.path.join(D, "plots", "filter_comparison.png"))
    plot_kalman_gain(kf, ekf, os.path.join(D, "plots", "kalman_gain.png"))


if __name__ == "__main__":
    main()
