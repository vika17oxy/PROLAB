# PRO Lab — Probabilistic Robotics: KF / EKF / Particle Filter on TurtleBot4

Three IMU-only probabilistic state estimation filters implemented as **ROS2 C++ nodes**
for TurtleBot4 simulation, with **Python (matplotlib)** post-processing.

---

## Repository Structure

```
prol_filters/
├── CMakeLists.txt
├── package.xml
├── include/prol_filters/
│   ├── kalman_filter.hpp       ← Linear KF  (header-only, Eigen)
│   ├── ekf.hpp                 ← Extended KF (header-only, Eigen)
│   └── particle_filter.hpp     ← Particle Filter (header-only, Eigen)
├── src/
│   ├── kalman_filter_node.cpp  ← ROS2 KF node
│   ├── ekf_node.cpp            ← ROS2 EKF node
│   └── particle_filter_node.cpp← ROS2 PF node
├── launch/
│   └── filters.launch.py
├── config/
│   ├── filter_params.yaml
│   └── filters.rviz
└── scripts/
    ├── plot_results.py         ← matplotlib post-processing
    └── evaluate_filters.py     ← RMSE / MAE metrics table
```

---

## State Space

All three filters share the same **6-DOF state vector**:

| Index | Symbol | Description        |
|-------|--------|--------------------|
| 0     | px     | X position [m]     |
| 1     | py     | Y position [m]     |
| 2     | θ      | Heading [rad]      |
| 3     | vx     | World-frame vx [m/s] |
| 4     | vy     | World-frame vy [m/s] |
| 5     | ω      | Angular rate [rad/s] |

**Sensor: IMU only** (`sensor_msgs/Imu` on `/imu`)
- Gyroscope: `angular_velocity.z` → angular rate ω
- Accelerometer: `linear_acceleration.x/y` → body-frame acceleration

---

## Filter Comparison

| Feature | KF | EKF | PF |
|---|---|---|---|
| Motion model | **Linear** (body-frame acc treated as world-frame) | **Nonlinear** (rotation transform + Jacobian) | **Nonlinear** (Monte-Carlo sampling) |
| Covariance propagation | F·P·Fᵀ + Q | Fⱼ·P·Fⱼᵀ + Q (Jacobian) | Particle spread |
| Landmark update | Linearised Jacobian | Proper Jacobian | Likelihood weighting |
| Computational cost | O(n³) state dim only | O(n³) + Jacobian | O(N·state) — N particles |
| Fails when | Heading deviates from 0° | Strongly nonlinear dynamics | N too small / weight degeneracy |

### Key Design Point
The **KF linearises** the body→world acceleration transform by assuming
`cos(θ)≈1, sin(θ)≈0`. This introduces a systematic velocity estimation error
as the robot turns. The **EKF** and **PF** handle this correctly, which is why
they outperform the KF on curved trajectories.

---

## Build

```bash
# Inside your ROS2 workspace (e.g. ~/ros2_ws/src/)
cd ~/ros2_ws/src
# Copy or clone prol_filters here, then:
cd ~/ros2_ws
colcon build --packages-select prol_filters
source install/setup.bash
```

**Dependencies:** `rclcpp`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`,
`std_msgs`, `visualization_msgs`, `tf2`, `tf2_ros`, `tf2_geometry_msgs`, `Eigen3`

---

## Quick Start

### 1. TurtleBot4 Simulation

Follow the official nav2 guide to launch the TurtleBot4 Ignition simulation:
https://docs.nav2.org/getting_started/index.html

Make sure the `/imu` topic is published.  If the IMU topic differs (e.g.,
`/tb4/imu`) remap it in the launch file.

### 2. Start All Filters

```bash
ros2 launch prol_filters filters.launch.py
```

### 3. Drive the Robot

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# or use nav2 goals in RViz2
```

---

## Mandatory Experiments

### Process Noise Q Variation

```bash
# Low Q — trusts model strongly
ros2 launch prol_filters filters.launch.py q_pos:=0.001 log_csv:=true
# High Q — trusts model weakly
ros2 launch prol_filters filters.launch.py q_pos:=0.1   log_csv:=true
```

### Measurement Noise R Variation

```bash
ros2 launch prol_filters filters.launch.py r_omega:=0.001 log_csv:=true
ros2 launch prol_filters filters.launch.py r_omega:=0.5   log_csv:=true
```

### Time-Delayed Measurements

```bash
ros2 launch prol_filters filters.launch.py delay_ms:=100.0 log_csv:=true
ros2 launch prol_filters filters.launch.py delay_ms:=500.0 log_csv:=true
```

### Particle Count (PF only)

```bash
ros2 launch prol_filters filters.launch.py num_particles:=50
ros2 launch prol_filters filters.launch.py num_particles:=1000
```

---

## Landmark Detection

A virtual landmark is defined in `config/filter_params.yaml` at position
`(landmark_x, landmark_y)`.  When the estimated position enters within
`landmark_radius` metres, the node triggers a **range measurement update**
using a simulated noisy distance reading.  The landmark is visualised as a
coloured cylinder in RViz2 (green = detected, red = not detected).

To move the landmark:
```bash
ros2 param set /kalman_filter_node landmark_x 3.0
ros2 param set /kalman_filter_node landmark_y -1.0
```

---

## Post-Processing (Python / matplotlib)

Enable CSV logging and collect data:
```bash
ros2 launch prol_filters filters.launch.py log_csv:=true
```

After the run, the working directory will contain `kf_log.csv`, `ekf_log.csv`,
`pf_log.csv`.

Generate all plots:
```bash
python3 prol_filters/scripts/plot_results.py --data-dir .
# → ./plots/trajectories.png, rmse_comparison.png, q_variation.png, ...
```

Print metrics table:
```bash
python3 prol_filters/scripts/evaluate_filters.py --data-dir .
```

---

## Topics Reference

| Topic | Type | Published by |
|---|---|---|
| `/imu` | `sensor_msgs/Imu` | TurtleBot4 simulation |
| `/ground_truth` | `nav_msgs/Odometry` | Gazebo (remap as needed) |
| `/kf/pose` | `geometry_msgs/PoseWithCovarianceStamped` | KF node |
| `/kf/odom` | `nav_msgs/Odometry` | KF node |
| `/kf/rmse` | `std_msgs/Float64` | KF node |
| `/kf/landmark_marker` | `visualization_msgs/Marker` | KF node |
| `/ekf/pose` | same pattern | EKF node |
| `/pf/pose` | same pattern | PF node |
| `/pf/particles` | `visualization_msgs/MarkerArray` | PF node |

---

## Parameters Reference

All parameters can be set in `config/filter_params.yaml` or overridden at launch time.

| Parameter | Default | Description |
|---|---|---|
| `q_pos` | 0.01 | Position process noise |
| `q_angle` | 0.001 | Heading process noise |
| `q_vel` | 0.1 | Velocity process noise |
| `q_omega` | 0.01 | Angular rate process noise |
| `r_omega` | 0.01 | Gyroscope measurement noise |
| `measurement_delay_ms` | 0.0 | Artificial pipeline delay [ms] |
| `landmark_x/y` | 2.0 / 2.0 | Landmark position |
| `landmark_radius` | 1.0 | Detection radius [m] |
| `r_landmark` | 0.1 | Landmark range noise |
| `num_particles` | 500 | PF only — particle count |
| `log_csv` | false | Write CSV log files |

---

## Grading Checklist

- [x] KF node (`kalman_filter_node.cpp` + `kalman_filter.hpp`)
- [x] EKF node (`ekf_node.cpp` + `ekf.hpp`)
- [x] PF node (`particle_filter_node.cpp` + `particle_filter.hpp`)
- [x] IMU-only sensor input (all three filters)
- [x] Same topics, coordinate frame (`odom`), launch conditions
- [x] Time-delayed measurements (`measurement_delay_ms` parameter)
- [x] Process noise Q variation (launch arg `q_pos`)
- [x] Measurement noise R variation (launch arg `r_omega`)
- [x] Runtime / performance logging (PF `update_ms` column in CSV)
- [x] Ground truth RMSE (`/kf|ekf|pf/rmse` topics + CSV)
- [x] Landmark detection (virtual landmark with range update + RViz marker)
- [x] Python matplotlib plots (`plot_results.py`)
- [x] RViz2 configuration (`filters.rviz`)
- [x] C++ implementation (all nodes)

---

## License

MIT
