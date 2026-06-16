#!/usr/bin/env python3
"""
runtime_plot.py — Runtime / Performance experiment (mandatory).
Reads the per-tick wall-clock time (update_ms column) logged by the KF, EKF and
PF nodes and produces runtime_comparison.png: mean per-update cost (bar, log
scale, with std whiskers) + the per-tick distribution (box). Also prints the RMSE
of each filter from the same run, for the sim<->plot alignment check.

Elias analyze_results.py colour/legend conventions; own code.
Usage:  python3 runtime_plot.py --data-dir /data/11_runtime
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}
plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "axes.axisbelow": True,
    "axes.edgecolor": "0.4", "font.size": 12, "axes.titlesize": 14,
    "axes.titleweight": "bold", "legend.framealpha": 0.92,
})


def load(p):
    if not os.path.isfile(p):
        return None
    d = np.genfromtxt(p, delimiter=",", names=True, invalid_raise=False)
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="/data/11_runtime")
    a = ap.parse_args(); D = a.data_dir
    logs = {"KF": load(os.path.join(D, "kf_log.csv")),
            "EKF": load(os.path.join(D, "ekf_log.csv")),
            "PF": load(os.path.join(D, "pf_log.csv"))}
    us = {}      # per-tick time in microseconds
    rmse = {}
    for n, d in logs.items():
        if d is None or "update_ms" not in d:
            continue
        v = d["update_ms"].astype(float) * 1000.0          # ms -> us
        us[n] = v[np.isfinite(v) & (v >= 0)]
        if "pos_err" in d:
            e = d["pos_err"].astype(float); rmse[n] = float(np.sqrt(np.mean(e[np.isfinite(e)]**2)))

    names = [n for n in ("KF", "EKF", "PF") if n in us]
    fig, (axb, axd) = plt.subplots(1, 2, figsize=(13, 5.6))

    # Per-update asymptotic complexity (d = state dim = 3, m = meas dim = 2,
    # M = particle count = 500). KF/EKF are dominated by the d×d covariance
    # products and the m×m gain inverse -> O(d^3); the PF touches every particle
    # in predict + weight + resample -> O(M·d).
    BIGO = {"KF": r"$O(d^3)$", "EKF": r"$O(d^3)$", "PF": r"$O(M\,d)$"}
    means = [us[n].mean() for n in names]; stds = [us[n].std() for n in names]
    bars = axb.bar(names, means, yerr=stds, capsize=6,
                   color=[C[n] for n in names], alpha=0.85, edgecolor="k", linewidth=0.8)
    for n, m, b in zip(names, means, bars):
        axb.text(b.get_x()+b.get_width()/2, m, f" {m:.2f} µs",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axb.set_xticks(range(len(names)))
    axb.set_xticklabels([f"{n}\n{BIGO[n]}" for n in names])
    axb.set_yscale("log"); axb.set_ylabel("mean time per update [µs]  (log)")
    axb.set_title("Computational cost per filter update")
    axb.text(0.5, -0.22, "d = state dim (3),  m = meas dim (2),  M = particles (500)",
             transform=axb.transAxes, ha="center", fontsize=9, color="0.3")

    axd.boxplot([us[n] for n in names], labels=names, showfliers=False,
                patch_artist=True,
                boxprops=dict(alpha=0.6), medianprops=dict(color="k", lw=1.5))
    for patch, n in zip(axd.findobj(plt.matplotlib.patches.PathPatch), names):
        patch.set_facecolor(C[n])
    axd.set_yscale("log"); axd.set_ylabel("time per update [µs]  (log)")
    axd.set_title("Per-tick runtime distribution")

    fig.suptitle("Runtime / Performance: KF vs EKF vs PF  "
                 f"(real run, {len(us[names[0]])} ticks @ 50 Hz)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(D, "plots", "runtime_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)

    BIGO_T = {"KF": "O(d^3)", "EKF": "O(d^3)", "PF": "O(M*d)"}
    print("\n=== Runtime (per update)   d=3 state, m=2 meas, M=500 particles ===")
    for n in names:
        print(f"  {n:4s} {BIGO_T[n]:7s}: mean {us[n].mean():7.2f} us   median {np.median(us[n]):7.2f} us   "
              f"p95 {np.percentile(us[n],95):7.2f} us   max {us[n].max():7.2f} us")
    print("\n=== RMSE (this real run) ===")
    for n in names:
        if n in rmse:
            print(f"  {n:4s}: {rmse[n]:.4f} m")


if __name__ == "__main__":
    main()
