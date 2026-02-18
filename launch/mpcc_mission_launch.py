"""
MPCC Mission Launch File

Launches the mission with MPCC (Model Predictive Contouring Control)
instead of Nav2's MPPI controller.

Includes:
- odom_from_tf: Converts TF (odom->base_link) to /odom topic
- mpcc_controller: MPCC path following controller (Python or C++)
- mission_manager: Mission waypoint management (MPCC mode)
- sign_detector: C++ HSV traffic sign detector (optional)

Usage:
    ros2 launch acc_stage1_mission mpcc_mission_launch.py
    ros2 launch acc_stage1_mission mpcc_mission_launch.py use_cpp_controller:=true
    ros2 launch acc_stage1_mission mpcc_mission_launch.py use_cpp_sign_detector:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('acc_stage1_mission')

    # Launch arguments - tuned from root cause analysis
    reference_velocity_arg = DeclareLaunchArgument(
        'reference_velocity',
        default_value='0.35',
        description='Target velocity (m/s) - reduced for safe cornering'
    )

    contour_weight_arg = DeclareLaunchArgument(
        'contour_weight',
        default_value='25.0',
        description='Weight for path contouring (lateral) error - PRIMARY weight'
    )

    lag_weight_arg = DeclareLaunchArgument(
        'lag_weight',
        default_value='5.0',
        description='Weight for lag (progress) error - secondary to lane keeping'
    )

    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='20',
        description='MPC prediction horizon (steps)'
    )

    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='30.0',
        description='Road boundary constraint penalty weight'
    )

    use_cpp_controller_arg = DeclareLaunchArgument(
        'use_cpp_controller',
        default_value='false',
        description='Use C++ MPCC controller instead of Python'
    )

    use_cpp_sign_detector_arg = DeclareLaunchArgument(
        'use_cpp_sign_detector',
        default_value='false',
        description='Use C++ HSV sign detector instead of Python YOLO'
    )

    # Odom From TF Node
    odom_from_tf_node = Node(
        package='acc_stage1_mission',
        executable='odom_from_tf',
        name='odom_from_tf',
        output='screen',
        parameters=[{
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'publish_rate': 50.0,
        }]
    )

    # Python MPCC Controller Node (default)
    mpcc_python_node = Node(
        package='acc_stage1_mission',
        executable='mpcc_controller',
        name='mpcc_controller',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_cpp_controller')),
        parameters=[{
            'reference_velocity': LaunchConfiguration('reference_velocity'),
            'contour_weight': LaunchConfiguration('contour_weight'),
            'lag_weight': LaunchConfiguration('lag_weight'),
            'horizon': LaunchConfiguration('horizon'),
        }],
        remappings=[
            ('/odom', '/odom'),
            ('/plan', '/plan'),
            ('/cmd_vel_nav', '/cmd_vel_nav'),
        ]
    )

    # C++ MPCC Controller Node (optional)
    mpcc_cpp_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='mpcc_controller_node',
        name='mpcc_controller_cpp',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_cpp_controller')),
        parameters=[{
            'reference_velocity': LaunchConfiguration('reference_velocity'),
            'contour_weight': LaunchConfiguration('contour_weight'),
            'lag_weight': LaunchConfiguration('lag_weight'),
            'horizon': LaunchConfiguration('horizon'),
            'boundary_weight': LaunchConfiguration('boundary_weight'),
        }],
    )

    # C++ Sign Detector Node (optional)
    sign_detector_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='sign_detector_node',
        name='sign_detector',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_cpp_sign_detector')),
    )

    # Mission Manager Node
    mission_node = Node(
        package='acc_stage1_mission',
        executable='mission_manager',
        name='mission_manager',
        output='screen',
        parameters=[{
            'mpcc_mode': True,
        }],
    )

    return LaunchDescription([
        reference_velocity_arg,
        contour_weight_arg,
        lag_weight_arg,
        horizon_arg,
        boundary_weight_arg,
        use_cpp_controller_arg,
        use_cpp_sign_detector_arg,
        odom_from_tf_node,
        mpcc_python_node,
        mpcc_cpp_node,
        sign_detector_node,
        mission_node,
    ])
