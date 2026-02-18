#!/usr/bin/env python3
"""
Full Mission Launch File for ACC Competition

This launch file starts all components needed for the autonomous mission:
1. SLAM and Navigation (Nav2)
2. Obstacle Detection (YOLO or color-based)
3. Mission Manager (waypoint navigation)

Usage:
    ros2 launch acc_stage1_mission full_mission_launch.py

With options:
    ros2 launch acc_stage1_mission full_mission_launch.py use_yolo:=true
    ros2 launch acc_stage1_mission full_mission_launch.py debug_viz:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    acc_pkg = get_package_share_directory('acc_stage1_mission')
    qcar2_pkg = get_package_share_directory('qcar2_nodes')

    # Launch arguments
    use_yolo_arg = DeclareLaunchArgument(
        'use_yolo',
        default_value='false',
        description='Use YOLO for detection (requires GPU). Set false for CPU-based color detection.'
    )

    debug_viz_arg = DeclareLaunchArgument(
        'debug_viz',
        default_value='true',
        description='Publish debug visualization images'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(acc_pkg, 'config', 'mission.yaml'),
        description='Path to mission configuration file'
    )

    goal_timeout_arg = DeclareLaunchArgument(
        'goal_timeout_s',
        default_value='120.0',
        description='Navigation goal timeout in seconds'
    )

    obstacle_detection_arg = DeclareLaunchArgument(
        'enable_obstacle_detection',
        default_value='true',
        description='Enable obstacle detection integration'
    )

    # Include SLAM and Nav2 launch
    slam_nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('qcar2_nodes'),
                'launch',
                'qcar2_slam_and_nav_bringup_virtual_launch.py'
            ])
        ])
    )

    # Obstacle detector node
    obstacle_detector_node = Node(
        package='acc_stage1_mission',
        executable='obstacle_detector',
        name='obstacle_detector',
        parameters=[{
            'use_yolo': LaunchConfiguration('use_yolo'),
            'debug_visualization': LaunchConfiguration('debug_viz'),
            'camera_topic': '/camera/color_image',
            'depth_topic': '/camera/depth_image',
        }],
        output='screen',
    )

    # Mission manager node (delayed start to let Nav2 initialize)
    mission_manager_node = Node(
        package='acc_stage1_mission',
        executable='mission_manager',
        name='mission_manager',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'use_tf_hub': True,
            'hub_tf_timeout_s': 15.0,
            'goal_timeout_s': LaunchConfiguration('goal_timeout_s'),
            'max_retries_per_leg': 3,
            'enable_led': True,
            'goal_tol_m': 0.35,
            'enable_obstacle_detection': LaunchConfiguration('enable_obstacle_detection'),
            'obstacle_pause_timeout_s': 30.0,
            'backup_distance_m': 0.15,
            'backup_speed': 0.1,
        }],
        output='screen',
    )

    # Delayed mission start (wait for Nav2 to initialize)
    delayed_mission = TimerAction(
        period=25.0,  # Wait 25 seconds for Nav2
        actions=[
            LogInfo(msg="Starting mission manager..."),
            mission_manager_node,
        ]
    )

    # Delayed detector start (wait for camera topics)
    delayed_detector = TimerAction(
        period=5.0,  # Wait 5 seconds for cameras
        actions=[
            LogInfo(msg="Starting obstacle detector..."),
            obstacle_detector_node,
        ]
    )

    return LaunchDescription([
        # Arguments
        use_yolo_arg,
        debug_viz_arg,
        config_file_arg,
        goal_timeout_arg,
        obstacle_detection_arg,

        # Log startup
        LogInfo(msg="=== ACC Stage 1 Full Mission Launch ==="),
        LogInfo(msg="Starting SLAM and Navigation..."),

        # Launch components
        slam_nav_launch,
        delayed_detector,
        delayed_mission,
    ])
