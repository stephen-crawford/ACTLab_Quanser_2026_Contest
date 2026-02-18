"""
Launch mission_manager WITH obstacle/sign detection for Stage I contest.

This launch file starts both the mission manager and the traffic system detector
for a complete autonomous solution with stop sign and traffic light handling.

Run after:
  - QLabs Plane World + Setup_Competition_Map.py (ENV container)
  - ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py (DEV container)

Usage:
  ros2 launch acc_stage1_mission mission_with_detection_launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("acc_stage1_mission")
    default_config = os.path.join(pkg_share, "config", "mission.yaml")

    # Core mission parameters
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to mission.yaml",
    )
    use_tf_hub_arg = DeclareLaunchArgument(
        "use_tf_hub",
        default_value="true",
        description="Capture hub from TF at startup",
    )
    goal_timeout_arg = DeclareLaunchArgument(
        "goal_timeout_s",
        default_value="120.0",
        description="NavigateToPose goal timeout (seconds)",
    )

    # Detector type selection
    detector_type_arg = DeclareLaunchArgument(
        "detector_type",
        default_value="traffic_system",
        description="Detection type: 'yolo' (requires GPU), 'traffic_system' (CPU-based), or 'none'",
    )

    # Mission manager node
    mission_node = Node(
        package="acc_stage1_mission",
        executable="mission_manager",
        name="mission_manager",
        output="screen",
        parameters=[
            {"config_file": LaunchConfiguration("config_file")},
            {"use_tf_hub": PythonExpression(["'", LaunchConfiguration("use_tf_hub"), "'.lower() == 'true'"])},
            {"goal_timeout_s": PythonExpression(["float('", LaunchConfiguration("goal_timeout_s"), "')"])},
            {"enable_obstacle_detection": True},
            {"enable_led": True},
            {"max_retries_per_leg": 3},
            {"hub_tf_timeout_s": 15.0},
            {"obstacle_pause_timeout_s": 30.0},
            {"backup_distance_m": 0.15},
            {"backup_speed": 0.1},
        ],
    )

    # Traffic system detector (CPU-based, uses OpenCV)
    # Note: This detector subscribes to /camera/color_image and publishes /motion_enable
    traffic_detector_node = Node(
        package="qcar2_autonomy",
        executable="traffic_system_detector",
        name="traffic_system_detector",
        output="screen",
        condition=LaunchConfiguration("detector_type").perform(None) == "traffic_system",
    )

    return LaunchDescription([
        config_file_arg,
        use_tf_hub_arg,
        goal_timeout_arg,
        detector_type_arg,
        mission_node,
        # Note: Uncomment traffic_detector_node when qcar2_autonomy is available
        # For now, run detector separately: ros2 run qcar2_autonomy traffic_system_detector
    ])
