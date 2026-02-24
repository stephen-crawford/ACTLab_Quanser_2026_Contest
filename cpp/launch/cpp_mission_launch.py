"""
All-C++ Mission Launch File

Launches the complete mission stack using only C++ nodes:
- odom_from_tf_node: TF -> /odom bridge
- mpcc_controller_node: MPCC path following controller
- mission_manager_node: Mission state machine (MPCC mode)
- sign_detector_node: HSV traffic sign/light/cone detection

Usage:
    ros2 launch acc_mpcc_controller_cpp cpp_mission_launch.py
    ros2 launch acc_mpcc_controller_cpp cpp_mission_launch.py reference_velocity:=0.30
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    reference_velocity_arg = DeclareLaunchArgument(
        'reference_velocity',
        default_value='0.35',
        description='Target velocity (m/s)'
    )

    contour_weight_arg = DeclareLaunchArgument(
        'contour_weight',
        default_value='25.0',
        description='Weight for path contouring (lateral) error'
    )

    lag_weight_arg = DeclareLaunchArgument(
        'lag_weight',
        default_value='5.0',
        description='Weight for lag (progress) error'
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

    # Odom From TF Node (C++)
    odom_from_tf_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='odom_from_tf_node',
        name='odom_from_tf',
        output='screen',
        parameters=[{
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'publish_rate': 50.0,
        }]
    )

    # MPCC Controller Node (C++)
    mpcc_controller_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='mpcc_controller_node',
        name='mpcc_controller_cpp',
        output='screen',
        parameters=[{
            'reference_velocity': LaunchConfiguration('reference_velocity'),
            'contour_weight': LaunchConfiguration('contour_weight'),
            'lag_weight': LaunchConfiguration('lag_weight'),
            'horizon': LaunchConfiguration('horizon'),
            'boundary_weight': LaunchConfiguration('boundary_weight'),
        }],
    )

    # Sign Detector Node (C++)
    sign_detector_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='sign_detector_node',
        name='sign_detector',
        output='screen',
    )

    # Mission Manager Node (C++)
    mission_manager_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='mission_manager_node',
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
        odom_from_tf_node,
        mpcc_controller_node,
        sign_detector_node,
        mission_manager_node,
    ])
