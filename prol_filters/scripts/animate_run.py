#!/usr/bin/env python3
"""
animate_run.py — animate the simulation on the S-slalom: ground truth + KF/EKF/PF
estimates + landmark. A measurement/processing delay is shown the way RViz shows
it — the estimates trail the robot by the delay (lag = v * delay).

Outputs MP4 if ffmpeg is available, otherwise a GIF.
Usage:  python3 animate_run.py --delay-ms 0    --out /data/animations/sim_nodelay.mp4
        python3 animate_run.py --delay-ms 5000 --out /data/animations/sim_delay5s.mp4
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
DT = 0.02
GYRO_NOISE = 0.012
R_R, R_B = 0.005, 0.01
Q_XY, Q_TH = 0.001, 0.0005
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}
# obstacle blocks (3x3 grid), for context
OBX = [0.94, 2.02, 3.10]; OBY = [-0.53, 0.51, 1.59]


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


def make_truth(V, seed=0):
    D, cum = build_path(WPX, WPY); n = int(cum[-1]/(V*DT)); gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(seed); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run_gauss(jac, gt, omega, V, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    x = gt[0].copy(); P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    est = np.zeros((n, 2)); est[0] = gt[0, :2]
    for k in range(1, n):
        th = x[2]; x = x + [V*math.cos(th)*DT, V*math.sin(th)*DT, omega[k]*DT]; x[2] = wrap(x[2])
        if jac:
            G = np.eye(3); G[0, 2] = -V*math.sin(th)*DT; G[1, 2] = V*math.cos(th)*DT; P = G@P@G.T+Q
        else:
            P = P+Q
        gx, gy, gth = gt[k]
        if math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy)+rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth)+rng.normal(0, math.sqrt(R_B))
            dx = lx-x[0]; dy = ly-x[1]; r = math.hypot(dx, dy)
            if r >= 1e-6:
                H = np.array([[-dx/r, -dy/r, 0.0], [dy/r/r, -dx/r/r, -1.0]])
                S = H@P@H.T+R; K = P@H.T@np.linalg.inv(S)
                x = x+K@np.array([z_r-r, wrap(z_b-wrap(math.atan2(dy, dx)-x[2]))]); x[2] = wrap(x[2])
                IKH = np.eye(3)-K@H; P = IKH@P@IKH.T+K@R@K.T
        est[k] = x[:2]
    return est


def run_pf(gt, omega, V, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    Pp = np.tile(gt[0], (NP, 1)).astype(float); w = np.full(NP, 1.0/NP)
    est = np.zeros((n, 2)); est[0] = gt[0, :2]
    for k in range(1, n):
        vn = V+rng.normal(0, SIGMA_V, NP); wn = omega[k]+rng.normal(0, SIGMA_W, NP)
        Pp[:, 0] += vn*np.cos(Pp[:, 2])*DT; Pp[:, 1] += vn*np.sin(Pp[:, 2])*DT
        Pp[:, 2] = wrap_arr(Pp[:, 2]+wn*DT)
        gx, gy, gth = gt[k]
        if math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy)+rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth)+rng.normal(0, math.sqrt(R_B))
            dx = lx-Pp[:, 0]; dy = ly-Pp[:, 1]; rp = np.hypot(dx, dy)
            er = z_r-rp; eb = wrap_arr(z_b-wrap_arr(np.arctan2(dy, dx)-Pp[:, 2]))
            w *= np.exp(-0.5*(er*er/R_R+eb*eb/R_B))
            s = w.sum(); w = w/s if s > 1e-300 else np.full(NP, 1.0/NP)
            pos = (rng.random()+np.arange(NP))/NP
            idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, NP-1)
            Pp = Pp[idx].copy(); w = np.full(NP, 1.0/NP)
        est[k] = [np.average(Pp[:, 0], weights=w), np.average(Pp[:, 1], weights=w)]
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--v", type=float, default=0.4)
    ap.add_argument("--out", default="/data/animations/sim.gif")
    a = ap.parse_args()

    gt, om = make_truth(a.v)
    est = {"KF": run_gauss(False, gt, om, a.v, 42),
           "EKF": run_gauss(True, gt, om, a.v, 42),
           "PF": run_pf(gt, om, a.v, 44)}
    n = len(gt); D = int(round(a.delay_ms/1000.0/DT))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.plot(gt[:, 0], gt[:, 1], "-", color="0.8", lw=2, zorder=1)            # full path (faint)
    for ox in OBX:                                                            # obstacle blocks
        for oy in OBY:
            ax.add_patch(plt.Rectangle((ox-0.12, oy-0.12), 0.24, 0.24, color="0.55", zorder=1))
    ax.plot(*LM, "*", color="tab:blue", ms=20, markeredgecolor="k", zorder=4, label="Landmark")
    ax.plot(gt[0, 0], gt[0, 1], "o", color="green", ms=9, zorder=4, label="Start")

    gt_trail, = ax.plot([], [], "-", color="k", lw=2.4, zorder=5, label="Ground Truth")
    robot, = ax.plot([], [], "o", color="orange", ms=14, markeredgecolor="k", zorder=8, label="Robot")
    dots = {n_: ax.plot([], [], "o", color=C[n_], ms=9, zorder=7, label=f"{n_} estimate")[0]
            for n_ in ("KF", "EKF", "PF")}
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11,
                  fontweight="bold", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)
    ttl = f"S-slalom — {'no delay' if D == 0 else f'{a.delay_ms:.0f} ms delay'}"
    ax.set_title(ttl, fontweight="bold")
    ax.set_xlim(gt[:, 0].min()-0.6, gt[:, 0].max()+0.6)
    ax.set_ylim(gt[:, 1].min()-0.6, max(gt[:, 1].max(), LM[1])+0.4)

    step = max(1, n//240)
    frames = list(range(0, n, step)) + [n-1]

    def update(i):
        gt_trail.set_data(gt[:i+1, 0], gt[:i+1, 1])
        robot.set_data([gt[i, 0]], [gt[i, 1]])
        j = max(0, i-D)                                # estimate shown delayed by D steps
        for n_ in dots:
            dots[n_].set_data([est[n_][j, 0]], [est[n_][j, 1]])
        lag = a.v * a.delay_ms/1000.0
        txt.set_text(f"t = {i*DT:4.1f} s\ndelay = {a.delay_ms:.0f} ms  (lag ~{lag:.1f} m)")
        return [gt_trail, robot, txt, *dots.values()]

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(a.out, writer=FFMpegWriter(fps=20, bitrate=2400))
        print("saved", a.out)
    except Exception as e:
        gif = os.path.splitext(a.out)[0] + ".gif"
        from matplotlib.animation import PillowWriter
        anim.save(gif, writer=PillowWriter(fps=20))
        print(f"ffmpeg unavailable ({type(e).__name__}); saved GIF instead:", gif)


if __name__ == "__main__":
    main()
