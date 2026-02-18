"""
Launch mission_manager for Stage I contest (pickup -> dropoff -> hub).

Run after:
  - QLabs Plane World + Setup_Competition_Map.py (ENV container)
  - ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py (DEV container)

Usage:
  ros2 launch acc_stage1_mission mission_launch.py

With YOLO detector (recommended):
  ros2 launch acc_stage1_mission mission_launch.py &
  ros2 run qcar2_autonomy yolo_detector
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("acc_stage1_mission")
    default_config = os.path.join(pkg_share, "config", "mission.yaml")

    # Core mission parameters
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to mission.yaml (pickup, dropoff, hub, dwell_s)",
    )
    use_tf_hub_arg = DeclareLaunchArgument(
        "use_tf_hub",
        default_value="true",
        description="Capture hub from TF map->base_link at startup",
    )
    hub_tf_timeout_arg = DeclareLaunchArgument(
        "hub_tf_timeout_s",
        default_value="15.0",
        description="Seconds to wait for map->base_link before using config hub",
    )
    goal_timeout_arg = DeclareLaunchArgument(
        "goal_timeout_s",
        default_value="120.0",
        description="NavigateToPose goal timeout (seconds)",
    )
    max_retries_arg = DeclareLaunchArgument(
        "max_retries_per_leg",
        default_value="3",
        description="Max retries per strategy before trying next recovery strategy",
    )
    enable_led_arg = DeclareLaunchArgument(
        "enable_led",
        default_value="true",
        description="Set qcar2_hardware LED from mission state",
    )
    goal_tol_arg = DeclareLaunchArgument(
        "goal_tol_m",
        default_value="0.35",
        description="Goal tolerance (m); should match Nav2 goal_checker params",
    )

    # Obstacle detection parameters
    enable_obstacle_detection_arg = DeclareLaunchArgument(
        "enable_obstacle_detection",
        default_value="true",
        description="Subscribe to /motion_enable for stop signs/obstacles",
    )
    obstacle_pause_timeout_arg = DeclareLaunchArgument(
        "obstacle_pause_timeout_s",
        default_value="30.0",
        description="Max time to wait for obstacle to clear before recovery",
    )

    # Recovery parameters
    backup_distance_arg = DeclareLaunchArgument(
        "backup_distance_m",
        default_value="0.15",
        description="Distance to back up during recovery (meters)",
    )
    backup_speed_arg = DeclareLaunchArgument(
        "backup_speed",
        default_value="0.1",
        description="Speed for backup maneuver (m/s)",
    )

    mission_node = Node(
        package="acc_stage1_mission",
        executable="mission_manager",
        name="mission_manager",
        output="screen",
        parameters=[
            {"config_file": LaunchConfiguration("config_file")},
            {"use_tf_hub": PythonExpression(["'", LaunchConfiguration("use_tf_hub"), "'.lower() == 'true'"])},
            {"hub_tf_timeout_s": PythonExpression(["float('", LaunchConfiguration("hub_tf_timeout_s"), "')"])},
            {"goal_timeout_s": PythonExpression(["float('", LaunchConfiguration("goal_timeout_s"), "')"])},
            {"max_retries_per_leg": PythonExpression(["int('", LaunchConfiguration("max_retries_per_leg"), "')"])},
            {"enable_led": PythonExpression(["'", LaunchConfiguration("enable_led"), "'.lower() == 'true'"])},
            {"goal_tol_m": PythonExpression(["float('", LaunchConfiguration("goal_tol_m"), "')"])},
            {"enable_obstacle_detection": PythonExpression(["'", LaunchConfiguration("enable_obstacle_detection"), "'.lower() == 'true'"])},
            {"obstacle_pause_timeout_s": PythonExpression(["float('", LaunchConfiguration("obstacle_pause_timeout_s"), "')"])},
            {"backup_distance_m": PythonExpression(["float('", LaunchConfiguration("backup_distance_m"), "')"])},
            {"backup_speed": PythonExpression(["float('", LaunchConfiguration("backup_speed"), "')"])},
        ],
    )

    return LaunchDescription([
        config_file_arg,
        use_tf_hub_arg,
        hub_tf_timeout_arg,
        goal_timeout_arg,
        max_retries_arg,
        enable_led_arg,
        goal_tol_arg,
        enable_obstacle_detection_arg,
        obstacle_pause_timeout_arg,
        backup_distance_arg,
        backup_speed_arg,
        mission_node,
    ])
