# PRO Lab — Probabilistic Robotics: KF / EKF / Particle Filter

Three probabilistic state-estimation filters — **Kalman Filter (KF)**, **Extended
Kalman Filter (EKF)** and **Particle Filter (PF)** — implemented as **ROS 2 C++
nodes** and compared on an obstacle-avoiding S-slalom trajectory, with **Python
(matplotlib)** post-processing. Everything runs in Docker; no real robot needed.

Author: Viktoriia Ovdiienko — FH Technikum Wien.

---

## Repository Structure

```
prol_filters/                       ROS 2 package
├── include/prol_filters/
│   ├── kalman_filter.hpp            Linear KF  (F = I covariance), Eigen, header-only
│   ├── ekf.hpp                      Extended KF (full Jacobian G, Joseph update)
│   └── particle_filter.hpp          Particle filter (MCL, 500 particles, low-variance resample)
├── src/
│   ├── kalman_filter_node.cpp       ROS 2 KF node
│   ├── ekf_node.cpp                 ROS 2 EKF node
│   ├── particle_filter_node.cpp     ROS 2 PF node
│   └── map_publisher_node.cpp       publishes the map.yaml as /map (OccupancyGrid)
├── launch/simulation.launch.py      sim + 3 filters + map (+ RViz if use_rviz:=true)
├── config/{filter_params.yaml, filters.rviz}
├── map/map.yaml
└── scripts/
    ├── imu_simulator.py             drives the trajectory; publishes /imu + /ground_truth
    ├── evaluate_filters.py          RMSE / heading / runtime table + rmse_bar.png
    ├── plot_comparison_gain.py      trajectories.png + filter_comparison.png
    ├── plot_localization.py         EKF localization explainer (3-panel)
    ├── delay_experiment.py          time-delay RMSE sweep (0/100/500 ms)
    ├── delay_trajectories.py        delay trajectory comparison (3-panel)
    ├── qr_experiment.py             Q-variation + R-variation
    ├── runtime_plot.py              per-update runtime comparison
    └── kalman_gain_slalom.py        Kalman gain over time
data/                                logged CSVs + generated plots (per experiment folder)
Dockerfile, docker-compose.yml, entrypoint.sh
map.yaml, map.pgm                    copied into the package at image-build time
ProbRob_Paper_Template_Englisch/     IEEE-style paper (.tex + .pdf)
```

---

## Model

All three filters share the same **3-DOF unicycle state** and motion model:

| Symbol | Description |
|--------|-------------|
| x, y   | position [m] |
| θ      | heading [rad] |

**Motion model** (control `u = [v, ω]`): `v` is a constant forward speed,
`ω` is read from the noisy gyroscope on `/imu`:

```
x ← x + v·cos(θ)·dt
y ← y + v·sin(θ)·dt
θ ← θ + ω·dt
```

**Measurement model** — a single known **landmark** gives range + bearing
(synthesised from ground truth + noise when the robot is within `landmark_radius`):

```
z = [ r , φ ] = [ ‖m − p‖ ,  atan2(m_y − y, m_x − x) − θ ]
```

**Sensors:** `imu_simulator.py` publishes `/imu` (gyro ω) and `/ground_truth`
(`nav_msgs/Odometry`, used only for evaluation and to synthesise the landmark reading).

---

## Filter Comparison

| Feature | KF | EKF | PF |
|---|---|---|---|
| Covariance prediction | `P + Q`  (**F = I**, no Jacobian) | `G·P·Gᵀ + Q` (full Jacobian) | particle spread |
| Landmark update | range/bearing, Joseph form | range/bearing, Joseph form | likelihood weighting + resample |
| Cost / step | ~6 µs | ~7 µs | ~150 µs (N = 500) |
| Weak point | F = I ignores θ→position coupling → drifts on turns | linearisation fails very close to the landmark (1/r² term) | particle degeneracy on symmetric paths |

**Key point:** the KF approximates the covariance prediction with `F = I`, so it
does **not** propagate heading uncertainty into position. The EKF fixes this with
the motion Jacobian `G`, and the PF avoids linearisation entirely — which is why
on the slalom the ordering is **PF < EKF < KF** in RMSE.

---

## Build (Docker)

```bash
docker build -t prol_filters:humble .
```

The Dockerfile builds the package with `colcon` (two-stage build) and copies
`map.yaml` / `map.pgm` into the package.

> Native alternative: drop `prol_filters/` into a ROS 2 Humble workspace and run
> `colcon build --packages-select prol_filters` (deps: `rclcpp`, `sensor_msgs`,
> `geometry_msgs`, `nav_msgs`, `std_msgs`, `visualization_msgs`, `tf2*`, `Eigen3`).

---

## Run

### Interactive (RViz)

```bash
docker compose up simulation
```
Shows the robot driving the S-slalom with the KF/EKF/PF pose estimates, covariance
ellipses, particle cloud, the landmark and the map. (Needs an X11 display — WSL 2 +
WSLg on Windows, or a Linux desktop.)

### Headless logged run (produces the CSVs the plots use)

```bash
docker compose run --rm headless          # writes kf_log.csv / ekf_log.csv / pf_log.csv to ./data
# or directly:
docker run --rm -w /data/11_runtime -v "$PWD/data:/data" prol_filters:humble \
  timeout 35 ros2 launch prol_filters simulation.launch.py \
  trajectory:=waypoints initial_x:=-0.3 initial_y:=-0.5 initial_theta:=0.896 \
  linear_vel:=0.3 landmark_x:=1.8 landmark_y:=3.0 landmark_radius:=6.0 \
  duration:=25.0 log_csv:=true use_rviz:=false
```

`simulation.launch.py` is the **single entry point**. Key arguments:

| Arg | Default | Meaning |
|---|---|---|
| `trajectory` | figure8 | `waypoints` (S-slalom) · `circle` · `figure8` |
| `linear_vel` | 0.3 | forward speed [m/s] |
| `initial_x/y/theta` | — | start pose (forwarded to all filters) |
| `q_xy`, `q_theta` | 0.001 / 0.0005 | process noise (Q) |
| `r_landmark`, `r_bearing` | 0.005 / 0.01 | measurement noise (R) |
| `gyro_noise_std` | 0.012 | gyro noise std |
| `sigma_v`, `sigma_omega` | 0.03 / 0.015 | PF motion noise |
| `num_particles` | 500 | PF particle count |
| `landmark_x/y/radius` | — / — / — | landmark position + detection radius |
| `delay_ms` | 0.0 | artificial measurement-processing delay |
| `duration` | — | auto-shutdown time [s] (flushes CSVs) |
| `log_csv`, `use_rviz` | false | enable CSV logging / RViz |

---

## Mandatory Experiments

The **baseline comparison + runtime** come from a real logged run (above). The
parameter studies are computed with deterministic offline replicas of the same
filter mathematics (reproducible — best-effort ROS QoS drops make live sweeps
noisy). Run them on the logged `data/`:

```bash
EP="docker run --rm --entrypoint python3 -v $PWD/data:/data -v $PWD/prol_filters/scripts:/s prol_filters:humble"

$EP /s/evaluate_filters.py  --data-dir /data/11_runtime   # RMSE table + rmse_bar.png
$EP /s/plot_comparison_gain.py --data-dir /data/11_runtime # trajectories + filter_comparison
$EP /s/runtime_plot.py     --data-dir /data/11_runtime     # runtime_comparison.png
$EP /s/qr_experiment.py    --data /data                    # q_variation.png + r_variation.png
$EP /s/delay_experiment.py --data /data                    # delay_rmse_comparison.png
$EP /s/delay_trajectories.py --out /data/10_scurve/plots/trajectories_delay_comparison.png
$EP /s/plot_localization.py  --out /data/10_scurve/plots/ekf_localization.png
```

---

## Landmark Detection

A single virtual landmark is set via `landmark_x` / `landmark_y` (default off the
path so the EKF linearisation stays well-conditioned). When the robot is within
`landmark_radius`, each filter applies a noisy range + bearing update. The
landmark is drawn in RViz as a coloured marker (green = in view).

---

## Topics

| Topic | Type | By |
|---|---|---|
| `/imu` | `sensor_msgs/Imu` | imu_simulator (gyro) |
| `/ground_truth` | `nav_msgs/Odometry` | imu_simulator |
| `/map` | `nav_msgs/OccupancyGrid` | map_publisher_node |
| `/kf/pose`, `/ekf/pose`, `/pf/pose` | `geometry_msgs/PoseWithCovarianceStamped` | filters |
| `/kf/rmse`, `/ekf/rmse`, `/pf/rmse` | `std_msgs/Float64` | filters |
| `/pf/particles` | `visualization_msgs/MarkerArray` | PF node |
| `/{kf,ekf,pf}/landmark_marker` | `visualization_msgs/Marker` | filters |

Fixed frame: **`map`**.

---

## Results (S-slalom, real run)

| Metric | KF | EKF | PF |
|---|---|---|---|
| Position RMSE [m] | 0.312 | 0.102 | **0.011** |
| Heading RMSE [°] | 7.0 | 2.7 | **1.0** |
| Runtime / step [µs] | 5.4 | 6.8 | 149 |

Full analysis, figures and discussion are in
`ProbRob_Paper_Template_Englisch/ProbRob_Paper_Template_Englisch.pdf`.

---

## Grading Checklist

- [x] KF / EKF / PF as three separate ROS 2 **C++** nodes
- [x] Same input data, coordinate frame (`map`) and trajectory for all filters
- [x] Process-noise (Q) variation — `qr_experiment.py` / `q_xy` arg
- [x] Measurement-noise (R) variation — `qr_experiment.py` / `r_landmark`, `r_bearing`
- [x] Runtime / performance — `runtime_plot.py`, `update_ms` CSV column
- [x] Ground-truth RMSE — `evaluate_filters.py`, `/{kf,ekf,pf}/rmse`
- [x] Landmark detection (own landmark, range + bearing update)
- [x] Time-delayed measurements (`delay_ms`) — specific task
- [x] Visualisation: matplotlib plots **and** RViz 2 (`filters.rviz`)
- [x] Documentation: IEEE-style paper (PDF)

---

## License

MIT
