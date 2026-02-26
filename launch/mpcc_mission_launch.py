"""
MPCC Mission Launch File

Launches the full C++ stack for the ACC competition:
- odom_from_tf_node: C++ TF -> /odom bridge
- mpcc_controller_node: C++ MPCC path following controller (SQP solver)
- mission_manager_node: C++ mission state machine + road graph path planning
- sign_detector_node: C++ HSV traffic sign/light/cone detection
- state_estimator_node: C++ EKF state estimator
- obstacle_tracker_node: C++ multi-class Kalman + lidar tracker
- traffic_light_map_node: C++ spatial traffic light mapping

All nodes are C++. No Python fallbacks.

Usage:
    ros2 launch acc_stage1_mission mpcc_mission_launch.py
    ros2 launch acc_stage1_mission mpcc_mission_launch.py reference_velocity:=0.40
    ros2 launch acc_stage1_mission mpcc_mission_launch.py horizon:=15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Launch arguments — tuned via full-mission simulation (Feb 2026)
    reference_velocity_arg = DeclareLaunchArgument(
        'reference_velocity',
        default_value='0.45',
        description='Target velocity (m/s) - lower for tighter tracking'
    )

    contour_weight_arg = DeclareLaunchArgument(
        'contour_weight',
        default_value='20.0',
        description='Weight for path contouring (lateral) error - tighter lane keeping'
    )

    lag_weight_arg = DeclareLaunchArgument(
        'lag_weight',
        default_value='10.0',
        description='Weight for lag (progress) error'
    )

    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='MPC prediction horizon (steps) - match reference (PolyCtrl 2025 K=10)'
    )

    boundary_weight_arg = DeclareLaunchArgument(
        'boundary_weight',
        default_value='0.0',
        description='Road boundary constraint penalty weight (ref has 0; disabled to avoid fighting contour cost on curves)'
    )

    use_direct_motor_arg = DeclareLaunchArgument(
        'use_direct_motor',
        default_value='true',
        description='Publish MotorCommands directly (bypass nav2_qcar_command_convert)'
    )

    use_state_estimator_arg = DeclareLaunchArgument(
        'use_state_estimator',
        default_value='false',
        description='Use EKF state estimator (fuses TF + encoder + odom). Disabled: adds velocity lag not in reference.'
    )

    use_dashboard_arg = DeclareLaunchArgument(
        'use_dashboard',
        default_value='false',
        description='Launch real-time telemetry dashboard (requires display)'
    )

    use_overlay_arg = DeclareLaunchArgument(
        'use_overlay',
        default_value='false',
        description='Launch path overlay map visualizer (requires display)'
    )

    # =========================================================================
    # C++ Nodes — the entire stack
    # =========================================================================

    # Odom From TF Node (C++)
    # broadcast_tf=false: Cartographer already publishes map→odom→base_link TF.
    # odom_from_tf only converts TF to /odom messages; it must NOT re-broadcast
    # TF or it conflicts with Cartographer and creates disconnected TF trees.
    odom_from_tf_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='odom_from_tf_node',
        name='odom_from_tf',
        output='screen',
        parameters=[{
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'publish_rate': 50.0,
            'broadcast_tf': False,
        }]
    )

    # C++ MPCC Controller Node
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
            'use_direct_motor': LaunchConfiguration('use_direct_motor'),
        }],
    )

    # C++ Sign Detector Node (HSV + contour detection)
    sign_detector_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='sign_detector_node',
        name='sign_detector',
        output='screen',
    )

    # C++ Mission Manager Node
    mission_manager_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='mission_manager_node',
        name='mission_manager',
        output='screen',
        parameters=[{
            'mpcc_mode': True,
        }],
    )

    # State Estimator Node (C++ EKF, conditional)
    state_estimator_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='state_estimator_node',
        name='state_estimator',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_state_estimator')),
    )

    # Obstacle Tracker Node (C++ multi-class Kalman + lidar)
    obstacle_tracker_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='obstacle_tracker_node',
        name='obstacle_tracker',
        output='screen',
    )

    # Traffic Light Map Node (C++)
    traffic_light_map_node = Node(
        package='acc_mpcc_controller_cpp',
        executable='traffic_light_map_node',
        name='traffic_light_map',
        output='screen',
    )

    # Dashboard Node (Python, opt-in — no C++ equivalent yet)
    dashboard_node = Node(
        package='acc_stage1_mission',
        executable='dashboard',
        name='dashboard',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_dashboard')),
    )

    # Path Overlay Node (Python, opt-in — bird's-eye track + planned path)
    path_overlay_node = Node(
        package='acc_stage1_mission',
        executable='path_overlay',
        name='path_overlay',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_overlay')),
    )

    return LaunchDescription([
        reference_velocity_arg,
        contour_weight_arg,
        lag_weight_arg,
        horizon_arg,
        boundary_weight_arg,
        use_direct_motor_arg,
        use_state_estimator_arg,
        use_dashboard_arg,
        use_overlay_arg,
        odom_from_tf_node,
        state_estimator_node,
        obstacle_tracker_node,
        traffic_light_map_node,
        mpcc_controller_node,
        sign_detector_node,
        mission_manager_node,
        dashboard_node,
        path_overlay_node,
    ])
