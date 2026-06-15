"""
filters.launch.py

Launches all three filter nodes (KF, EKF, PF) with shared parameters from
config/filter_params.yaml plus RViz2 for visualisation.
Requires an external IMU source or imu_simulator.py running separately.

Usage:
  ros2 launch prol_filters filters.launch.py
  ros2 launch prol_filters filters.launch.py delay_ms:=100.0
  ros2 launch prol_filters filters.launch.py use_rviz:=false log_csv:=true
"""

import launch
import launch.conditions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg         = FindPackageShare("prol_filters")
    params_file = PathJoinSubstitution([pkg, "config", "filter_params.yaml"])
    rviz_file   = PathJoinSubstitution([pkg, "config", "filters.rviz"])

    args = [
        DeclareLaunchArgument("delay_ms",      default_value="0.0"),
        DeclareLaunchArgument("q_xy",          default_value="0.01"),
        DeclareLaunchArgument("q_theta",       default_value="0.005"),
        DeclareLaunchArgument("r_landmark",    default_value="0.05"),
        DeclareLaunchArgument("r_bearing",     default_value="0.05"),
        DeclareLaunchArgument("sigma_v",       default_value="0.03"),
        DeclareLaunchArgument("sigma_omega",   default_value="0.015"),
        DeclareLaunchArgument("num_particles", default_value="500"),
        DeclareLaunchArgument("log_csv",       default_value="false"),
        DeclareLaunchArgument("use_rviz",      default_value="true"),
    ]

    common_params = {
        "measurement_delay_ms": LaunchConfiguration("delay_ms"),
        "q_xy":                 LaunchConfiguration("q_xy"),
        "q_theta":              LaunchConfiguration("q_theta"),
        "r_landmark":           LaunchConfiguration("r_landmark"),
        "r_bearing":            LaunchConfiguration("r_bearing"),
        "log_csv":              LaunchConfiguration("log_csv"),
    }

    kf_node = Node(
        package="prol_filters",
        executable="kalman_filter_node",
        name="kalman_filter_node",
        parameters=[params_file, common_params],
        output="screen",
        emulate_tty=True,
    )

    ekf_node = Node(
        package="prol_filters",
        executable="ekf_node",
        name="ekf_node",
        parameters=[params_file, common_params],
        output="screen",
        emulate_tty=True,
    )

    pf_node = Node(
        package="prol_filters",
        executable="particle_filter_node",
        name="particle_filter_node",
        parameters=[params_file, common_params, {
            "num_particles": LaunchConfiguration("num_particles"),
            "sigma_v":       LaunchConfiguration("sigma_v"),
            "sigma_omega":   LaunchConfiguration("sigma_omega"),
        }],
        output="screen",
        emulate_tty=True,
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_file],
        condition=launch.conditions.IfCondition(LaunchConfiguration("use_rviz")),
        output="screen",
    )

    return LaunchDescription(args + [kf_node, ekf_node, pf_node, rviz_node])
