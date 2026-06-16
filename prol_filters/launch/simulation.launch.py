"""
simulation.launch.py  —  the single entry point: IMU simulator + KF + EKF + PF
+ map publisher (+ RViz2 when use_rviz:=true).

Live visualisation:        use_rviz:=true
Headless data collection:  use_rviz:=false log_csv:=true duration:=<seconds>

Usage:
  ros2 launch prol_filters simulation.launch.py
  ros2 launch prol_filters simulation.launch.py trajectory:=figure8
  ros2 launch prol_filters simulation.launch.py trajectory:=straight
  ros2 launch prol_filters simulation.launch.py trajectory:=scurve
  ros2 launch prol_filters simulation.launch.py use_rviz:=false log_csv:=true
  ros2 launch prol_filters simulation.launch.py q_xy:=0.01 r_landmark:=0.5
  ros2 launch prol_filters simulation.launch.py delay_ms:=100.0

Via Docker Compose (single command):
  docker compose up simulation
"""

import os
import launch
import launch.conditions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg         = FindPackageShare('prol_filters')
    params_file = PathJoinSubstitution([pkg, 'config', 'filter_params.yaml'])
    rviz_file   = PathJoinSubstitution([pkg, 'config', 'filters.rviz'])

    # ── Launch arguments ───────────────────────────────────────────────────────
    args = [
        # Simulator
        DeclareLaunchArgument('trajectory',      default_value='figure8',
                              description='circle | straight | figure8 | scurve'),
        DeclareLaunchArgument('duration',        default_value='3600.0',
                              description='Seconds before auto-exit (0 = infinite)'),
        DeclareLaunchArgument('linear_vel',      default_value='0.3',
                              description='Forward speed [m/s]'),
        DeclareLaunchArgument('angular_vel',     default_value='0.5',
                              description='Angular rate [rad/s]'),
        DeclareLaunchArgument('gyro_noise_std',  default_value='0.012'),
        DeclareLaunchArgument('accel_noise_std', default_value='0.05'),
        DeclareLaunchArgument('initial_x',       default_value='0.0'),
        DeclareLaunchArgument('initial_y',       default_value='-2.7'),
        DeclareLaunchArgument('initial_theta',   default_value='-1.5708'),

        # Filter noise (Q and R experiments)
        DeclareLaunchArgument('q_xy',        default_value='0.001',
                              description='Process noise q_xy [m²/step]'),
        DeclareLaunchArgument('q_theta',     default_value='0.0005',
                              description='Process noise q_theta [rad²/step]'),
        DeclareLaunchArgument('r_landmark',  default_value='0.005',
                              description='Landmark range noise R [m²]'),
        DeclareLaunchArgument('r_bearing',   default_value='0.01',
                              description='Landmark bearing noise R [rad²]'),

        # Landmark placement (overridable so it can sit along the trajectory)
        DeclareLaunchArgument('landmark_x',      default_value='0.5'),
        DeclareLaunchArgument('landmark_y',      default_value='-1.2'),
        DeclareLaunchArgument('landmark_radius', default_value='1.5'),

        # Particle Filter
        DeclareLaunchArgument('sigma_v',       default_value='0.03'),
        DeclareLaunchArgument('sigma_omega',   default_value='0.015'),
        DeclareLaunchArgument('num_particles', default_value='500'),

        # Time-delay experiment
        DeclareLaunchArgument('delay_ms', default_value='0.0',
                              description='Artificial measurement delay [ms] — '
                                          'set 100 or 500 for delay experiments'),

        # Misc
        DeclareLaunchArgument('log_csv',  default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
    ]

    # ── IMU simulator ──────────────────────────────────────────────────────────
    sim_node = Node(
        package='prol_filters',
        executable='imu_simulator.py',
        name='imu_simulator',
        parameters=[{
            'trajectory':      LaunchConfiguration('trajectory'),
            'duration':        LaunchConfiguration('duration'),
            'linear_vel':      LaunchConfiguration('linear_vel'),
            'angular_vel':     LaunchConfiguration('angular_vel'),
            'gyro_noise_std':  LaunchConfiguration('gyro_noise_std'),
            'accel_noise_std': LaunchConfiguration('accel_noise_std'),
            'initial_x':       LaunchConfiguration('initial_x'),
            'initial_y':       LaunchConfiguration('initial_y'),
            'initial_theta':   LaunchConfiguration('initial_theta'),
        }],
        output='screen',
        emulate_tty=True,
    )

    # Parameters common to KF, EKF, PF
    # NOTE: initial_x/y/theta MUST be forwarded here too — otherwise overriding
    # them on the command line moves only the simulator (ground truth) while the
    # filters keep their filter_params.yaml default start pose, producing a
    # constant offset that never converges.
    common = {
        'measurement_delay_ms': LaunchConfiguration('delay_ms'),
        'q_xy':                 LaunchConfiguration('q_xy'),
        'q_theta':              LaunchConfiguration('q_theta'),
        'r_landmark':           LaunchConfiguration('r_landmark'),
        'r_bearing':            LaunchConfiguration('r_bearing'),
        'initial_x':            LaunchConfiguration('initial_x'),
        'initial_y':            LaunchConfiguration('initial_y'),
        'initial_theta':        LaunchConfiguration('initial_theta'),
        'initial_vx':           LaunchConfiguration('linear_vel'),
        'landmark_x':           LaunchConfiguration('landmark_x'),
        'landmark_y':           LaunchConfiguration('landmark_y'),
        'landmark_radius':      LaunchConfiguration('landmark_radius'),
        'log_csv':              LaunchConfiguration('log_csv'),
    }

    kf_node = Node(
        package='prol_filters',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        parameters=[params_file, common],
        output='screen',
        emulate_tty=True,
    )

    ekf_node = Node(
        package='prol_filters',
        executable='ekf_node',
        name='ekf_node',
        parameters=[params_file, common],
        output='screen',
        emulate_tty=True,
    )

    pf_node = Node(
        package='prol_filters',
        executable='particle_filter_node',
        name='particle_filter_node',
        parameters=[params_file, common, {
            'num_particles': LaunchConfiguration('num_particles'),
            'sigma_v':       LaunchConfiguration('sigma_v'),
            'sigma_omega':   LaunchConfiguration('sigma_omega'),
        }],
        output='screen',
        emulate_tty=True,
    )

    # ── Map publisher: our map.yaml as a /map OccupancyGrid backdrop ───────────
    map_pub_node = Node(
        package='prol_filters',
        executable='map_publisher_node',
        name='map_publisher',
        parameters=[{
            'map_yaml': os.path.join(
                get_package_share_directory('prol_filters'), 'map', 'map.yaml'),
            'frame_id': 'map',
        }],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file],
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )

    return LaunchDescription(
        args + [sim_node, kf_node, ekf_node, pf_node, map_pub_node, rviz_node])
