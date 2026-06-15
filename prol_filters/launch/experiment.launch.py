"""
experiment.launch.py  —  Headless timed experiment: IMU simulator + KF + EKF + PF.

The simulator exits automatically after `duration` seconds, which triggers
on_exit=Shutdown() and terminates the whole ROS session so all CSV logs are flushed.

Usage:
  # Baseline (no delay):
  ros2 launch prol_filters experiment.launch.py

  # Q variation experiments:
  ros2 launch prol_filters experiment.launch.py q_xy:=0.0001  # low Q
  ros2 launch prol_filters experiment.launch.py q_xy:=0.01    # baseline
  ros2 launch prol_filters experiment.launch.py q_xy:=0.1     # high Q

  # R variation experiments:
  ros2 launch prol_filters experiment.launch.py r_landmark:=0.0005  # low R
  ros2 launch prol_filters experiment.launch.py r_landmark:=0.05    # high R

  # Time-delay experiments (specific task):
  ros2 launch prol_filters experiment.launch.py delay_ms:=100.0
  ros2 launch prol_filters experiment.launch.py delay_ms:=500.0

  # Custom trajectory:
  ros2 launch prol_filters experiment.launch.py trajectory:=circle duration:=60.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg         = FindPackageShare('prol_filters')
    params_file = PathJoinSubstitution([pkg, 'config', 'filter_params.yaml'])

    args = [
        # ── Simulator ─────────────────────────────────────────────────────────
        DeclareLaunchArgument('trajectory',      default_value='figure8',
                              description='circle | straight | figure8 | scurve'),
        DeclareLaunchArgument('duration',        default_value='90.0',
                              description='Experiment duration [s]'),
        DeclareLaunchArgument('linear_vel',      default_value='0.3',
                              description='Forward speed [m/s]'),
        DeclareLaunchArgument('angular_vel',     default_value='0.5',
                              description='Angular rate [rad/s]'),
        DeclareLaunchArgument('gyro_noise_std',  default_value='0.012'),
        DeclareLaunchArgument('accel_noise_std', default_value='0.05'),
        DeclareLaunchArgument('initial_x',       default_value='0.0',
                              description='Initial robot x [m]'),
        DeclareLaunchArgument('initial_y',       default_value='-2.7',
                              description='Initial robot y [m]'),
        DeclareLaunchArgument('initial_theta',   default_value='-1.5708',
                              description='Initial robot heading [rad]'),

        # ── Filter noise (Q variation experiments) ────────────────────────────
        DeclareLaunchArgument('q_xy',       default_value='0.001',
                              description='Process noise q_xy [m²/step]'),
        DeclareLaunchArgument('q_theta',    default_value='0.0005',
                              description='Process noise q_theta [rad²/step]'),

        # ── Measurement noise (R variation experiments) ───────────────────────
        DeclareLaunchArgument('r_landmark', default_value='0.005',
                              description='Landmark range noise [m²]'),
        DeclareLaunchArgument('r_bearing',  default_value='0.01',
                              description='Landmark bearing noise [rad²]'),

        # ── Particle Filter ───────────────────────────────────────────────────
        DeclareLaunchArgument('sigma_v',       default_value='0.03'),
        DeclareLaunchArgument('sigma_omega',   default_value='0.015'),
        DeclareLaunchArgument('num_particles', default_value='500'),

        # ── Time-delay experiment (specific task) ─────────────────────────────
        DeclareLaunchArgument('delay_ms', default_value='0.0',
                              description='Artificial measurement delay [ms]. '
                                          'Baseline=0, test with 100 and 500 ms.'),
    ]

    # ── Simulator (on_exit=Shutdown terminates the whole session) ─────────────
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
        on_exit=Shutdown(),
    )

    common = {
        'measurement_delay_ms': LaunchConfiguration('delay_ms'),
        'q_xy':                 LaunchConfiguration('q_xy'),
        'q_theta':              LaunchConfiguration('q_theta'),
        'r_landmark':           LaunchConfiguration('r_landmark'),
        'r_bearing':            LaunchConfiguration('r_bearing'),
        'initial_vx':           LaunchConfiguration('linear_vel'),
        'initial_x':            LaunchConfiguration('initial_x'),
        'initial_y':            LaunchConfiguration('initial_y'),
        'initial_theta':        LaunchConfiguration('initial_theta'),
        'log_csv':              True,  # always log in experiments
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

    return LaunchDescription(args + [sim_node, kf_node, ekf_node, pf_node])
