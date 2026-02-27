"""
All-C++ Mission Launch File (legacy — prefer mpcc_mission_launch.py)

This launch file is installed with the acc_mpcc_controller_cpp package.
The primary launch file is launch/mpcc_mission_launch.py (acc_stage1_mission).

Usage:
    ros2 launch acc_mpcc_controller_cpp cpp_mission_launch.py
    ros2 launch acc_mpcc_controller_cpp cpp_mission_launch.py reference_velocity:=0.40
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments — must match mpcc_mission_launch.py defaults
    reference_velocity_arg = DeclareLaunchArgument(
        'reference_velocity',
        default_value='0.45',
        description='Target velocity (m/s)'
    )

    contour_weight_arg = DeclareLaunchArgument(
        'contour_weight',
        default_value='8.0',
        description='Weight for path contouring (lateral) error'
    )

    lag_weight_arg = DeclareLaunchArgument(
        'lag_weight',
        default_value='15.0',
        description='Weight for lag (progress) error'
    )

    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='MPC prediction horizon (steps)'
    )

    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='0.0',
        description='Road boundary constraint penalty weight (ref has 0)'
    )

    use_state_estimator_arg = DeclareLaunchArgument(
        'use_state_estimator',
        default_value='false',
        description='Use EKF state estimator for controller state input'
    )

    steering_slew_rate_arg = DeclareLaunchArgument(
        'steering_slew_rate',
        default_value='1.0',
        description='Maximum steering command slew rate (rad/s) to reduce oversteer from abrupt command jumps'
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
            'use_state_estimator': LaunchConfiguration('use_state_estimator'),
            'steering_slew_rate': LaunchConfiguration('steering_slew_rate'),
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
        use_state_estimator_arg,
        steering_slew_rate_arg,
        odom_from_tf_node,
        mpcc_controller_node,
        sign_detector_node,
        mission_manager_node,
    ])
