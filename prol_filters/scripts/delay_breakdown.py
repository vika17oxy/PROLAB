#!/usr/bin/env python3
"""
delay_breakdown.py — "At what delay does each filter break?"

Scientific pipeline: sweep the measurement-processing delay from 0 up to a large
value, run a deterministic offline replica of the KF / EKF / PF (same buffered-
delay model and node-matched parameters as the C++ nodes), average the position
RMSE over several noise realisations for reproducibility, and locate each
filter's BREAKING POINT.

The delay is modelled exactly as in the node: each landmark measurement is taken
when the landmark is physically in range (true geometry + noise) and then
*buffered* and applied D = round(delay/dt) steps later — so for large delays the
correction pulls the estimate toward a stale pose.

Two reference levels are reported:
  • a fixed usability threshold (default 0.50 m), and
  • the open-loop dead-reckoning RMSE (no landmark at all): once a filter's RMSE
    reaches this, the delayed correction no longer helps at all.
The break point is the smallest delay whose mean RMSE exceeds the threshold.

Outputs:  <data>/delay_breakdown.png   and   <data>/delay_breakdown.csv
Usage:    python3 delay_breakdown.py --data /data [--v 0.4] [--max-ms 5000]
          [--step-ms 250] [--seeds 12] [--threshold 0.5]
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
DT = 0.02
GYRO_NOISE = 0.012                 # = node gyro_noise_std
R_R, R_B = 0.005, 0.01             # = node r_landmark, r_bearing
Q_XY, Q_TH = 0.001, 0.0005         # = node q_xy, q_theta
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500   # = node PF params
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}

plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "axes.axisbelow": True,
    "axes.edgecolor": "0.4", "font.size": 12, "axes.titlesize": 14,
    "axes.titleweight": "bold", "legend.framealpha": 0.92,
})


def wrap(a): return math.atan2(math.sin(a), math.cos(a))
def wrap_arr(a): return np.arctan2(np.sin(a), np.cos(a))


def build_path(xs, ys):
    pts = np.array(list(zip(xs, ys)), float)
    P = np.vstack([2*pts[0]-pts[1], pts, 2*pts[-1]-pts[-2]]); al = 0.5; dense = []
    for i in range(1, len(P)-2):
        P0, P1, P2, P3 = P[i-1], P[i], P[i+1], P[i+2]
        t0 = 0.0
        t1 = t0+max(np.linalg.norm(P1-P0)**al, 1e-6)
        t2 = t1+max(np.linalg.norm(P2-P1)**al, 1e-6); t3 = t2+max(np.linalg.norm(P3-P2)**al, 1e-6)
        for k in range(80):
            t = t1+(t2-t1)*k/80.0
            A1 = (t1-t)/(t1-t0)*P0+(t-t0)/(t1-t0)*P1
            A2 = (t2-t)/(t2-t1)*P1+(t-t1)/(t2-t1)*P2
            A3 = (t3-t)/(t3-t2)*P2+(t-t2)/(t3-t2)*P3
            B1 = (t2-t)/(t2-t0)*A1+(t-t0)/(t2-t0)*A2
            B2 = (t3-t)/(t3-t1)*A2+(t-t1)/(t3-t1)*A3
            dense.append((t2-t)/(t2-t1)*B1+(t-t1)/(t2-t1)*B2)
    dense.append(pts[-1]); D = np.array(dense)
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    return D, np.concatenate([[0.0], np.cumsum(seg)])


def make_truth(V, seed, bias):
    # `bias` = systematic gyro error [rad/s] (model error). Without it the gyro is
    # near-perfect over this short path, dead-reckoning barely drifts and the
    # landmark is irrelevant — so a small bias is needed to make the correction
    # (and hence the delay) matter. Same device used by the Q/R study.
    D, cum = build_path(WPX, WPY); n = int(cum[-1]/(V*DT)); gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(seed); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + bias + rng.normal(0, GYRO_NOISE)
    return gt, omega


def rmse(est, gt):
    return float(np.sqrt(np.mean((est[:, 0]-gt[:, 0])**2 + (est[:, 1]-gt[:, 1])**2)))


def run_gauss(jac, gt, omega, V, delay_steps, seed):
    """Buffered-delay KF (jac=False, F=I) / EKF (jac=True). delay_steps<0 => no updates."""
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    x = gt[0].copy(); P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    est = np.zeros((n, 2)); est[0] = gt[0, :2]; pending = []
    for k in range(1, n):
        th = x[2]; x = x + [V*math.cos(th)*DT, V*math.sin(th)*DT, omega[k]*DT]; x[2] = wrap(x[2])
        if jac:
            G = np.eye(3); G[0, 2] = -V*math.sin(th)*DT; G[1, 2] = V*math.cos(th)*DT; P = G@P@G.T + Q
        else:
            P = P + Q
        gx, gy, gth = gt[k]
        if delay_steps >= 0 and math.hypot(lx-gx, ly-gy) < LM_R:          # true-range detection (sensor)
            z_r = math.hypot(lx-gx, ly-gy) + rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth) + rng.normal(0, math.sqrt(R_B))
            pending.append((k, z_r, z_b))
        while pending and (k-pending[0][0]) >= delay_steps:
            _, z_r, z_b = pending.pop(0); dx = lx-x[0]; dy = ly-x[1]; r = math.hypot(dx, dy)
            if r >= 1e-6:
                H = np.array([[-dx/r, -dy/r, 0.0], [dy/r/r, -dx/r/r, -1.0]])
                S = H@P@H.T + R; K = P@H.T@np.linalg.inv(S)
                x = x + K@np.array([z_r-r, wrap(z_b-wrap(math.atan2(dy, dx)-x[2]))]); x[2] = wrap(x[2])
                IKH = np.eye(3)-K@H; P = IKH@P@IKH.T + K@R@K.T            # Joseph form (matches C++)
        est[k] = x[:2]
    return rmse(est, gt)


def run_pf(gt, omega, V, delay_steps, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    Pp = np.tile(gt[0], (NP, 1)).astype(float); w = np.full(NP, 1.0/NP)
    est = np.zeros((n, 2)); est[0] = gt[0, :2]; pending = []
    for k in range(1, n):
        vn = V + rng.normal(0, SIGMA_V, NP); wn = omega[k] + rng.normal(0, SIGMA_W, NP)
        Pp[:, 0] += vn*np.cos(Pp[:, 2])*DT; Pp[:, 1] += vn*np.sin(Pp[:, 2])*DT
        Pp[:, 2] = wrap_arr(Pp[:, 2] + wn*DT)
        gx, gy, gth = gt[k]
        if delay_steps >= 0 and math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy) + rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth) + rng.normal(0, math.sqrt(R_B))
            pending.append((k, z_r, z_b))
        while pending and (k-pending[0][0]) >= delay_steps:
            _, z_r, z_b = pending.pop(0)
            dx = lx-Pp[:, 0]; dy = ly-Pp[:, 1]; rp = np.hypot(dx, dy)
            er = z_r-rp; eb = wrap_arr(z_b-wrap_arr(np.arctan2(dy, dx)-Pp[:, 2]))
            w *= np.exp(-0.5*(er*er/R_R + eb*eb/R_B))
            s = w.sum(); w = w/s if s > 1e-300 else np.full(NP, 1.0/NP)
            pos = (rng.random()+np.arange(NP))/NP
            idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, NP-1)
            Pp = Pp[idx].copy(); w = np.full(NP, 1.0/NP)
        est[k] = [np.average(Pp[:, 0], weights=w), np.average(Pp[:, 1], weights=w)]
    return rmse(est, gt)


def mean_rmse(name, V, delay_steps, seeds, bias):
    vals = []
    for s in range(seeds):
        gt, om = make_truth(V, s, bias)
        if name == "KF":  vals.append(run_gauss(False, gt, om, V, delay_steps, 100+s))
        elif name == "EKF": vals.append(run_gauss(True, gt, om, V, delay_steps, 100+s))
        else: vals.append(run_pf(gt, om, V, delay_steps, 200+s))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--v", type=float, default=0.4)
    ap.add_argument("--max-ms", type=float, default=5000.0)
    ap.add_argument("--step-ms", type=float, default=250.0)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5, help="usability RMSE threshold [m]")
    ap.add_argument("--bias", type=float, default=0.02, help="gyro bias [rad/s] (model error)")
    a = ap.parse_args()

    delays = np.arange(0.0, a.max_ms + 1e-9, a.step_ms)
    names = ["KF", "EKF", "PF"]
    curve = {n: np.array([mean_rmse(n, a.v, int(round(d/1000.0/DT)), a.seeds, a.bias) for d in delays])
             for n in names}
    dead = {n: mean_rmse(n, a.v, -1, a.seeds, a.bias) for n in names}   # open-loop (no landmark)

    def breakpoint(n):
        over = np.where(curve[n] > a.threshold)[0]
        return float(delays[over[0]]) if len(over) else None

    bp = {n: breakpoint(n) for n in names}

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in names:
        ax.plot(delays, curve[n], "o-", color=C[n], lw=2, ms=4, label=f"{n}")
        ax.axhline(dead[n], color=C[n], ls=":", lw=1.2, alpha=0.7)
        if bp[n] is not None:
            ax.axvline(bp[n], color=C[n], ls="--", lw=1.3, alpha=0.8)
            ax.annotate(f"{n} breaks\n{bp[n]:.0f} ms", xy=(bp[n], a.threshold),
                        xytext=(bp[n]+120, a.threshold*1.5), color=C[n], fontsize=9,
                        fontweight="bold")
    ax.axhline(a.threshold, color="0.3", ls="-", lw=1.4, label=f"threshold {a.threshold:.2f} m")
    ax.set_xlabel("measurement-processing delay [ms]")
    ax.set_ylabel("position RMSE [m]")
    ax.set_title(f"Delay breakdown — when each filter breaks  (v = {a.v} m/s)")
    ax.set_ylim(0, max(a.threshold*3, max(curve['KF'].max(), curve['EKF'].max())*1.1))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)
    ax.text(1.01, 0.02, "dotted = open-loop\n(dead-reckoning) RMSE\nper filter",
            transform=ax.transAxes, fontsize=8, color="0.3")
    fig.tight_layout()
    out_png = os.path.join(a.data, "delay_breakdown.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    # ── csv + console table ──────────────────────────────────────────────────
    with open(os.path.join(a.data, "delay_breakdown.csv"), "w") as f:
        f.write("delay_ms," + ",".join(f"rmse_{n}" for n in names) + "\n")
        for i, d in enumerate(delays):
            f.write(f"{d:.0f}," + ",".join(f"{curve[n][i]:.4f}" for n in names) + "\n")

    print(f"\n=== Delay breakdown (v={a.v} m/s, {a.seeds} seeds, threshold {a.threshold} m) ===")
    print(f"{'filter':6s} {'baseline(0ms)':>14s} {'dead-reckoning':>15s} "
          f"{'breaks > thr at':>16s}")
    for n in names:
        b = f"{bp[n]:.0f} ms" if bp[n] is not None else f">{a.max_ms:.0f} ms (never)"
        print(f"{n:6s} {curve[n][0]:>13.3f}m {dead[n]:>14.3f}m {b:>16s}")
    print("\nsaved", out_png)


if __name__ == "__main__":
    main()
