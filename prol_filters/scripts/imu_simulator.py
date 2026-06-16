#!/usr/bin/env python3
"""
imu_simulator.py  —  Synthetic IMU and ground-truth publisher for PROL experiments.

Publishes:
  /imu          (sensor_msgs/Imu)       — noisy body-frame gyro + accelerometer
  /ground_truth (nav_msgs/Odometry)     — exact robot pose (for evaluation)
  /robot_marker (visualization_msgs/Marker)  — TurtleBot visual in RViz2
  /map_marker   (visualization_msgs/Marker)  — occupancy-grid walls in RViz2

Trajectories:
  circle   — constant v, constant ω  (single circle)
  straight — constant v, ω = 0
  figure8  — constant v, alternating ±ω every figure8_period/2 seconds
             (each half-period = one full circle → robot returns to start)
  scurve   — constant v, alternating ±ω every π/ω seconds
             (each half-period = one semicircle → robot progresses forward
              in an S / snake pattern, never returning to start)

The node exits automatically after `duration` seconds.
When used with simulation.launch.py (on_exit=Shutdown()), this terminates
the whole ROS session so CSV logs are flushed.

Parameters:
  trajectory        — circle | straight | figure8 | scurve  (default: figure8)
  duration          — seconds before auto-exit         (default: 73.0)
  imu_rate          — IMU publish rate [Hz]            (default: 50.0)
  linear_vel        — forward speed [m/s]              (default: 0.3)
  angular_vel       — angular rate [rad/s]             (default: 0.5)
  figure8_period    — full figure8 period [s]          (default: 30.0)
  initial_x/y/theta — starting pose                   (default: 0,-2.7,-π/2)
  gyro_noise_std    — gyro noise σ [rad/s]             (default: 0.012)
  accel_noise_std   — accelerometer noise σ [m/s²]     (default: 0.05)
"""

import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import tf2_ros
from tf2_ros import StaticTransformBroadcaster

try:
    import yaml as yaml_mod
except ImportError:
    yaml_mod = None


class ImuSimulator(Node):
    def __init__(self):
        super().__init__('imu_simulator')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('trajectory',     'figure8')
        self.declare_parameter('duration',        73.0)
        self.declare_parameter('imu_rate',        50.0)
        self.declare_parameter('linear_vel',       0.3)
        self.declare_parameter('angular_vel',      0.5)
        self.declare_parameter('figure8_period',  30.0)
        self.declare_parameter('initial_x',        0.0)
        self.declare_parameter('initial_y',       -2.7)
        self.declare_parameter('initial_theta',   -math.pi / 2.0)
        self.declare_parameter('gyro_noise_std',   0.012)
        self.declare_parameter('accel_noise_std',  0.05)
        # 'waypoints' trajectory: smooth slalom through these points (world frame).
        # Default = S-curve weaving through the 3x3 landmark grid, left wall -> right wall.
        self.declare_parameter('waypoints_x', [-0.3, 0.94, 2.02, 3.10, 4.2])
        self.declare_parameter('waypoints_y', [-0.5, 1.05, -0.01, 1.05, 1.3])

        traj         = self.get_parameter('trajectory').value
        self.dur     = self.get_parameter('duration').value
        rate         = self.get_parameter('imu_rate').value
        self.v       = self.get_parameter('linear_vel').value
        self.w       = self.get_parameter('angular_vel').value
        self.f8_per  = self.get_parameter('figure8_period').value
        self.sg      = self.get_parameter('gyro_noise_std').value
        self.sa      = self.get_parameter('accel_noise_std').value
        self.dt      = 1.0 / rate

        self.traj    = traj
        self.px      = self.get_parameter('initial_x').value
        self.py      = self.get_parameter('initial_y').value
        self.theta   = self.get_parameter('initial_theta').value
        self.sim_t   = 0.0
        self.rng     = np.random.default_rng(42)

        # Build the waypoint spline (and snap the start pose to it) if requested.
        self.path_s = 0.0
        if self.traj == 'waypoints':
            wx = list(self.get_parameter('waypoints_x').value)
            wy = list(self.get_parameter('waypoints_y').value)
            self._build_path(wx, wy)
            self.px, self.py, self.theta = self._path_state(0.0)

        # ── Publishers and TF ─────────────────────────────────────────────────
        self.imu_pub        = self.create_publisher(Imu,      '/imu',          10)
        self.gt_pub         = self.create_publisher(Odometry, '/ground_truth', 10)
        self.robot_pub      = self.create_publisher(Marker,   '/robot_marker', 10)
        self.map_pub        = self.create_publisher(Marker,   '/map_marker',   10)
        self.tf_br          = tf2_ros.TransformBroadcaster(self)
        self.static_tf_br   = StaticTransformBroadcaster(self)
        self._map_marker    = None

        self._publish_static_map_tf()
        self._load_map()

        # ── Timers ────────────────────────────────────────────────────────────
        self.sim_timer = self.create_timer(self.dt, self._step)
        self.map_timer = self.create_timer(5.0, self._republish_map)

        self.get_logger().info(
            f'IMU simulator started: trajectory={traj}  '
            f'duration={self.dur:.1f}s  rate={rate:.0f}Hz  '
            f'v={self.v:.2f}m/s  ω={self.w:.2f}rad/s')

    # ── Trajectory: returns (v, omega) at current sim time ───────────────────
    def _controls(self):
        if self.traj == 'straight':
            return self.v, 0.0
        if self.traj == 'figure8':
            half = self.f8_per / 2.0
            omega = self.w if (self.sim_t % self.f8_per) < half else -self.w
            return self.v, omega
        if self.traj == 'scurve':
            # Alternating semicircular arcs: each half-period = π/ω seconds
            # (exactly half a circle), so the robot traces an S/snake path
            # and keeps moving forward rather than looping back to start.
            sc_half = math.pi / self.w       # duration of one semicircle
            sc_full = 2.0 * sc_half
            omega = self.w if (self.sim_t % sc_full) < sc_half else -self.w
            return self.v, omega
        # default: circle
        return self.v, self.w

    # ── Waypoint path (centripetal Catmull-Rom spline) ───────────────────────
    def _build_path(self, xs, ys):
        pts = np.array(list(zip(xs, ys)), dtype=float)
        if len(pts) < 2:
            raise ValueError('waypoints trajectory needs >= 2 waypoints')
        # Reflect end points so the spline reaches the first/last waypoint.
        P = np.vstack([2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]])
        alpha = 0.5  # centripetal -> minimal overshoot near tight turns
        dense = []
        for i in range(1, len(P) - 2):
            P0, P1, P2, P3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            t0 = 0.0
            t1 = t0 + max(np.linalg.norm(P1 - P0) ** alpha, 1e-6)
            t2 = t1 + max(np.linalg.norm(P2 - P1) ** alpha, 1e-6)
            t3 = t2 + max(np.linalg.norm(P3 - P2) ** alpha, 1e-6)
            for k in range(80):
                t  = t1 + (t2 - t1) * k / 80.0
                A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
                A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
                A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
                B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
                B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
                dense.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
        dense.append(pts[-1])
        dense = np.array(dense)
        self.path_x = dense[:, 0]
        self.path_y = dense[:, 1]
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        self.path_cum   = np.concatenate([[0.0], np.cumsum(seg)])
        self.path_total = float(self.path_cum[-1])
        self.get_logger().info(
            f'Waypoint path built: {len(dense)} samples, length {self.path_total:.2f} m')

    def _path_state(self, s):
        """Position (x, y) and tangent heading at arc-length s along the path."""
        s = min(max(s, 0.0), self.path_total)
        x = float(np.interp(s, self.path_cum, self.path_x))
        y = float(np.interp(s, self.path_cum, self.path_y))
        i = int(np.searchsorted(self.path_cum, s))
        i = min(max(i, 1), len(self.path_x) - 2)
        th = math.atan2(self.path_y[i + 1] - self.path_y[i - 1],
                        self.path_x[i + 1] - self.path_x[i - 1])
        return x, y, th

    # ── Simulation step ───────────────────────────────────────────────────────
    def _step(self):
        if self.dur > 0.0 and self.sim_t >= self.dur:
            self.get_logger().info('Simulation complete — shutting down.')
            self.sim_timer.cancel()
            raise SystemExit

        if self.traj == 'waypoints':
            self.path_s += self.v * self.dt
            if self.path_s >= self.path_total:
                self.get_logger().info('Waypoint path complete — shutting down.')
                self.sim_timer.cancel()
                raise SystemExit
            new_x, new_y, new_th = self._path_state(self.path_s)
            v     = self.v
            dth   = math.atan2(math.sin(new_th - self.theta),
                               math.cos(new_th - self.theta))   # wrap to [-pi, pi]
            omega = dth / self.dt
            self.px, self.py, self.theta = new_x, new_y, new_th
        else:
            v, omega = self._controls()
            # Integrate ground-truth pose (exact Euler)
            self.theta += omega * self.dt
            self.px    += v * math.cos(self.theta) * self.dt
            self.py    += v * math.sin(self.theta) * self.dt
        self.sim_t += self.dt

        now  = self.get_clock().now().to_msg()
        half = self.theta / 2.0
        sin_h = math.sin(half)
        cos_h = math.cos(half)

        # ── IMU message (noisy body-frame) ────────────────────────────────────
        imu = Imu()
        imu.header.stamp    = now
        imu.header.frame_id = 'base_link'

        # Gyroscope z (heading rate) + Gaussian noise
        imu.angular_velocity.z = omega + self.rng.normal(0.0, self.sg)

        # Accelerometer: tangential (0 for const. v), centripetal = v*ω (in +y body frame)
        imu.linear_acceleration.x = self.rng.normal(0.0, self.sa)
        imu.linear_acceleration.y = v * omega + self.rng.normal(0.0, self.sa)
        imu.linear_acceleration.z = 9.81   # gravity (not used by filter nodes)

        # Covariance diagonals (for RViz display; not used by our nodes)
        imu.angular_velocity_covariance[8]    = self.sg ** 2
        imu.linear_acceleration_covariance[0] = self.sa ** 2
        imu.linear_acceleration_covariance[4] = self.sa ** 2
        imu.orientation_covariance[0]         = -1.0   # orientation not provided

        self.imu_pub.publish(imu)

        # ── Ground-truth odometry (exact) ─────────────────────────────────────
        gt = Odometry()
        gt.header.stamp         = now
        gt.header.frame_id      = 'odom'
        gt.child_frame_id       = 'base_link'
        gt.pose.pose.position.x = self.px
        gt.pose.pose.position.y = self.py
        gt.pose.pose.orientation.z = sin_h
        gt.pose.pose.orientation.w = cos_h
        gt.twist.twist.linear.x    = v
        gt.twist.twist.angular.z   = omega
        self.gt_pub.publish(gt)

        # ── TF: odom → base_link ──────────────────────────────────────────────
        tf_msg = TransformStamped()
        tf_msg.header.stamp         = now
        tf_msg.header.frame_id      = 'odom'
        tf_msg.child_frame_id       = 'base_link'
        tf_msg.transform.translation.x = self.px
        tf_msg.transform.translation.y = self.py
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z    = sin_h
        tf_msg.transform.rotation.w    = cos_h
        self.tf_br.sendTransform(tf_msg)

        # ── Robot marker (yellow cylinder, TurtleBot4 approx. size) ──────────
        mk = Marker()
        mk.header.stamp    = now
        mk.header.frame_id = 'odom'
        mk.ns   = 'robot'; mk.id = 0
        mk.type   = Marker.CYLINDER
        mk.action = Marker.ADD
        mk.pose.position.x    = self.px
        mk.pose.position.y    = self.py
        mk.pose.position.z    = 0.15
        mk.pose.orientation.z = sin_h
        mk.pose.orientation.w = cos_h
        mk.scale.x = 0.35; mk.scale.y = 0.35; mk.scale.z = 0.30
        mk.color.r = 1.0; mk.color.g = 0.8; mk.color.b = 0.0; mk.color.a = 1.0
        self.robot_pub.publish(mk)

    # ── Static TF: map → odom (identity — no SLAM offset) ────────────────────
    def _publish_static_map_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id  = 'odom'
        t.transform.rotation.w = 1.0
        self.static_tf_br.sendTransform(t)

    # ── Occupancy-grid map marker ─────────────────────────────────────────────
    def _load_map(self):
        """Read map.yaml + map.pgm and build a CUBE_LIST marker for wall cells."""
        if yaml_mod is None:
            self.get_logger().warn('PyYAML not available — skipping map publish')
            return
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg      = get_package_share_directory('prol_filters')
            yaml_path = os.path.join(pkg, 'map', 'map.yaml')
            if not os.path.isfile(yaml_path):
                self.get_logger().warn(f'map.yaml not found: {yaml_path}')
                return
            with open(yaml_path) as f:
                meta = yaml_mod.safe_load(f)
            pgm_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
            if not os.path.isfile(pgm_path):
                self.get_logger().warn(f'map.pgm not found: {pgm_path}')
                return

            pixels, width, height, maxval = self._read_pgm(pgm_path)
            negate     = meta.get('negate', 0)
            occ_thresh = meta.get('occupied_thresh', 0.65)
            resolution = meta.get('resolution', 0.05)
            origin     = meta.get('origin', [0.0, 0.0, 0.0])

            mk = Marker()
            mk.header.frame_id = 'map'
            mk.ns     = 'map_walls'; mk.id = 0
            mk.type   = Marker.CUBE_LIST
            mk.action = Marker.ADD
            mk.scale.x = resolution; mk.scale.y = resolution; mk.scale.z = 0.20
            mk.color.r = 0.65; mk.color.g = 0.65; mk.color.b = 0.65; mk.color.a = 1.0
            mk.pose.orientation.w = 1.0

            n_occ = 0
            for row in range(height):
                for col in range(width):
                    pix  = int(pixels[row, col])
                    prob = pix / maxval if negate else (maxval - pix) / maxval
                    if prob >= occ_thresh:
                        pt = Point()
                        pt.x = origin[0] + (col + 0.5) * resolution
                        pt.y = origin[1] + (row + 0.5) * resolution
                        pt.z = 0.10
                        mk.points.append(pt)
                        n_occ += 1

            self._map_marker = mk
            self._republish_map()
            self.get_logger().info(
                f'Map loaded: {width}×{height} @ {resolution}m/px  ({n_occ} occupied cells)')
        except Exception as exc:
            self.get_logger().warn(f'Map load failed: {exc}')

    def _republish_map(self):
        if self._map_marker is None:
            return
        self._map_marker.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self._map_marker)

    @staticmethod
    def _read_pgm(path):
        """Read binary or ASCII PGM; returns (pixels_array, width, height, maxval)."""
        with open(path, 'rb') as f:
            raw = f.read()
        idx = 0

        def read_line():
            nonlocal idx
            line = b''
            while idx < len(raw) and raw[idx:idx+1] != b'\n':
                line += raw[idx:idx+1]
                idx += 1
            idx += 1
            return line.decode('ascii').strip()

        magic = read_line()
        assert magic in ('P5', 'P2'), f'Unsupported PGM format: {magic}'
        tokens = []
        while len(tokens) < 3:
            ln = read_line()
            if not ln.startswith('#'):
                tokens.extend(ln.split())
        width, height, maxval = int(tokens[0]), int(tokens[1]), int(tokens[2])
        if magic == 'P5':
            pixels = np.frombuffer(raw[idx:idx + width * height],
                                   dtype=np.uint8).reshape((height, width))
        else:
            flat   = list(map(int, raw[idx:].decode('ascii').split()))
            pixels = np.array(flat, dtype=np.uint8).reshape((height, width))
        return pixels, width, height, maxval


def main(args=None):
    rclpy.init(args=args)
    node = ImuSimulator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
