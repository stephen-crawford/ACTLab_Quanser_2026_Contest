#!/usr/bin/env python3
"""
MPCC Controller - Model Predictive Contouring Control for QCar2

Solver backend selection (automatic, configured via config/modules.yaml):
  1. C++ SQP (Eigen, gradient projection) — fastest, preferred
  2. CasADi MPCC (pympc_core, fresh Opti per solve) — proven stable
  3. Pure Pursuit + Stanley — safe fallback when no optimizer available

The auto-selection hierarchy is implemented in SOLVER_BACKEND at module
level: C++ > CasADi > fallback. This matches the 'auto' setting in
config/modules.yaml -> controller.backend.

To force a specific backend:
  config/modules.yaml -> controller.backend: casadi
  (or modify SOLVER_BACKEND directly for development)

Key Features:
- CasADi MPCC with fresh Opti per solve (no constraint accumulation)
- Ackermann vehicle dynamics with RK4 integration
- Cubic spline path representation (smooth tangents)
- Contouring + lag cost for path following
- Linearized obstacle avoidance constraints
- Road boundary soft constraints
- Pure Pursuit + Stanley fallback when CasADi unavailable

Usage:
    ros2 run acc_stage1_mission mpcc_controller

Topics:
    Subscribes:
        /odom - Vehicle odometry
        /plan - Reference path from planner
        /obstacle_detections - Detected obstacles
        /motion_enable - Enable/disable from obstacle detector

    Publishes:
        /cmd_vel_nav - Velocity commands
        /mpcc/predicted_path - Predicted trajectory (visualization)
"""

import datetime
import os
import math
import time as time_module
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf2_ros import TransformException

try:
    from qcar2_interfaces.msg import MotorCommands
    HAS_MOTOR_COMMANDS = True
except ImportError:
    HAS_MOTOR_COMMANDS = False

# Import road boundaries and traffic control state
from acc_stage1_mission.road_boundaries import RoadBoundarySpline
from acc_stage1_mission.traffic_control_state import TrafficControlState

# Import pympc_core - fixed MPCC solver from PyMPC framework
from acc_stage1_mission.pympc_core import MPCCSolver, CubicSplinePath
from acc_stage1_mission.pympc_core import MPCCConfig as CoreMPCCConfig
from acc_stage1_mission.pympc_core import HAS_CPP_SOLVER, CppMPCCSolver

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    print("CasADi not available")

# Determine best available solver backend
if HAS_CPP_SOLVER:
    SOLVER_BACKEND = "cpp"
elif CASADI_AVAILABLE:
    SOLVER_BACKEND = "casadi"
else:
    SOLVER_BACKEND = "fallback"
    print("No high-performance solver available - using Pure Pursuit fallback")


# =============================================================================
# CONFIGURATION - EDIT THESE TO TUNE BEHAVIOR
# =============================================================================
@dataclass
class MPCCConfig:
    """MPCC Controller Configuration

    TUNING NOTES (from root cause analysis):
    - contour_weight MUST be higher than lag_weight to prevent lane violations.
      The previous 8:15 ratio prioritized progress over staying in lane, causing
      the vehicle to cut corners and violate lane boundaries.
    - reference_velocity reduced from 0.45 to 0.35 for safer cornering on the
      tight 1:10 scale track. Competition penalizes lane violations heavily
      (-1 to -5 stars) while there's no explicit speed bonus.
    - boundary_weight increased to enforce lane constraints more strictly.
    """
    # Horizon and timing
    horizon: int = 20              # Prediction horizon steps
    dt: float = 0.1                # Time step (seconds)

    # Vehicle parameters (QCar2)
    wheelbase: float = 0.256       # Wheelbase length (m)
    max_velocity: float = 0.60     # Max forward velocity (m/s) - increased for direct motor mode
    min_velocity: float = -0.15    # Max reverse velocity (m/s)
    max_steering: float = 0.45     # Max steering angle (rad)
    max_acceleration: float = 0.6  # Max acceleration (m/s^2) - smoother transitions
    max_steering_rate: float = 0.6 # Max steering rate (rad/s) - smoother steering

    # Cost weights — CONTOUR > LAG to prevent lane violations.
    # Previous ratio (8:15) caused vehicle to prioritize forward progress over
    # staying centered in lane, resulting in corner cutting and lane departures.
    contour_weight: float = 25.0   # Lateral deviation from path center (PRIMARY)
    lag_weight: float = 5.0        # Progress along path (secondary to lane keeping)
    velocity_weight: float = 2.0   # Track reference velocity
    steering_weight: float = 3.0   # Penalize large steering
    acceleration_weight: float = 1.5  # Smooth acceleration
    steering_rate_weight: float = 4.0 # Smooth steering changes (prevents oscillation)

    # Obstacle avoidance
    obstacle_weight: float = 200.0  # Obstacle avoidance penalty
    safety_margin: float = 0.10     # Extra safety distance (m)
    robot_radius: float = 0.13      # Vehicle collision radius (m) - QCar2 is small

    # Reference velocity - increased for direct motor mode
    reference_velocity: float = 0.50  # Target velocity (m/s)

    # Solver settings
    max_iterations: int = 75        # More iterations for better solutions
    tolerance: float = 1e-5         # Tighter tolerance

    # Road boundary constraints - ENABLED for lane keeping
    boundary_enabled: bool = True   # ENABLED - keeps vehicle within lane
    boundary_weight: float = 30.0   # Penalty for boundary violations (INCREASED)
    boundary_default_width: float = 0.22  # Default road half-width (tighter than 0.25)


class VehicleState:
    """Vehicle state representation"""
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0):
        self.x = x          # Position x
        self.y = y          # Position y
        self.theta = theta  # Heading angle
        self.v = v          # Velocity
        self.delta = delta  # Steering angle

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.delta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'VehicleState':
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


class ReferencePath:
    """Reference path with progress tracking"""
    def __init__(self, waypoints: List[Tuple[float, float]]):
        self.waypoints = np.array(waypoints)
        self.n_points = len(waypoints)
        self._compute_path_properties()

    def _compute_path_properties(self):
        """Compute cumulative distance and tangent angles"""
        self.cumulative_dist = np.zeros(self.n_points)
        self.tangent_angles = np.zeros(self.n_points)

        for i in range(1, self.n_points):
            dx = self.waypoints[i, 0] - self.waypoints[i-1, 0]
            dy = self.waypoints[i, 1] - self.waypoints[i-1, 1]
            self.cumulative_dist[i] = self.cumulative_dist[i-1] + np.sqrt(dx**2 + dy**2)
            self.tangent_angles[i-1] = np.arctan2(dy, dx)

        self.tangent_angles[-1] = self.tangent_angles[-2] if self.n_points > 1 else 0.0
        self.total_length = self.cumulative_dist[-1]

    def get_position_at_progress(self, s: float) -> Tuple[float, float]:
        """Get position at path progress s"""
        s = np.clip(s, 0, self.total_length)
        idx = np.searchsorted(self.cumulative_dist, s) - 1
        idx = np.clip(idx, 0, self.n_points - 2)

        # Linear interpolation
        local_s = s - self.cumulative_dist[idx]
        segment_len = self.cumulative_dist[idx + 1] - self.cumulative_dist[idx]

        if segment_len > 1e-6:
            alpha = local_s / segment_len
        else:
            alpha = 0.0

        x = self.waypoints[idx, 0] + alpha * (self.waypoints[idx + 1, 0] - self.waypoints[idx, 0])
        y = self.waypoints[idx, 1] + alpha * (self.waypoints[idx + 1, 1] - self.waypoints[idx, 1])

        return x, y

    def get_tangent_at_progress(self, s: float) -> float:
        """Get tangent angle at path progress s"""
        s = np.clip(s, 0, self.total_length)
        idx = np.searchsorted(self.cumulative_dist, s) - 1
        idx = np.clip(idx, 0, self.n_points - 2)
        return self.tangent_angles[idx]

    def find_closest_progress(self, x: float, y: float) -> float:
        """Find path progress closest to point (x, y) with segment interpolation"""
        # Find closest waypoint
        distances = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest_idx = np.argmin(distances)

        # Interpolate along adjacent segments for more accurate progress
        best_progress = self.cumulative_dist[closest_idx]
        min_dist = distances[closest_idx]

        point = np.array([x, y])

        # Check segment before closest point
        if closest_idx > 0:
            p1 = self.waypoints[closest_idx - 1]
            p2 = self.waypoints[closest_idx]
            v = p2 - p1
            seg_len_sq = np.dot(v, v)
            if seg_len_sq > 1e-10:
                t = np.clip(np.dot(point - p1, v) / seg_len_sq, 0.0, 1.0)
                proj = p1 + t * v
                dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
                    seg_len = np.sqrt(seg_len_sq)
                    best_progress = self.cumulative_dist[closest_idx - 1] + t * seg_len

        # Check segment after closest point
        if closest_idx < self.n_points - 1:
            p1 = self.waypoints[closest_idx]
            p2 = self.waypoints[closest_idx + 1]
            v = p2 - p1
            seg_len_sq = np.dot(v, v)
            if seg_len_sq > 1e-10:
                t = np.clip(np.dot(point - p1, v) / seg_len_sq, 0.0, 1.0)
                proj = p1 + t * v
                dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
                    seg_len = np.sqrt(seg_len_sq)
                    best_progress = self.cumulative_dist[closest_idx] + t * seg_len

        return best_progress

    def compute_contouring_error(self, x: float, y: float, s: float) -> Tuple[float, float]:
        """
        Compute contouring (lateral) and lag (longitudinal) errors.

        Returns:
            e_c: Contouring error (lateral deviation from path)
            e_l: Lag error (distance behind reference point)
        """
        # Reference point on path
        ref_x, ref_y = self.get_position_at_progress(s)
        theta_ref = self.get_tangent_at_progress(s)

        # Error in path frame
        dx = x - ref_x
        dy = y - ref_y

        # Rotate to path frame
        cos_t = np.cos(theta_ref)
        sin_t = np.sin(theta_ref)

        e_l = cos_t * dx + sin_t * dy      # Lag (along path)
        e_c = -sin_t * dx + cos_t * dy     # Contour (perpendicular)

        return e_c, e_l


class Obstacle:
    """Obstacle representation"""
    def __init__(self, x: float, y: float, radius: float = 0.3,
                 vx: float = 0.0, vy: float = 0.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx  # Velocity for prediction
        self.vy = vy

    def predict_position(self, dt: float) -> Tuple[float, float]:
        """Predict position after dt seconds"""
        return self.x + self.vx * dt, self.y + self.vy * dt


class MPCCBridge:
    """
    Bridge between the ROS2 node and pympc_core.MPCCSolver.

    Handles conversion between ROS2 types (VehicleState, ReferencePath, Obstacle)
    and the pympc_core types (numpy arrays, CubicSplinePath, obstacle tuples).
    """
    def __init__(self, config: MPCCConfig, road_boundaries: Optional[RoadBoundarySpline] = None):
        self.config = config
        self.road_boundaries = road_boundaries
        self._last_obstacle_info = None
        self._backend = SOLVER_BACKEND

        # Create pympc_core config
        core_config = CoreMPCCConfig(
            horizon=config.horizon,
            dt=config.dt,
            wheelbase=config.wheelbase,
            max_velocity=config.max_velocity,
            min_velocity=max(0.0, config.min_velocity),
            max_steering=config.max_steering,
            max_acceleration=config.max_acceleration,
            max_steering_rate=config.max_steering_rate,
            contour_weight=config.contour_weight,
            lag_weight=config.lag_weight,
            velocity_weight=config.velocity_weight,
            steering_weight=config.steering_weight,
            acceleration_weight=config.acceleration_weight,
            steering_rate_weight=config.steering_rate_weight,
            boundary_weight=config.boundary_weight,
            robot_radius=config.robot_radius,
            safety_margin=config.safety_margin,
            max_iter=config.max_iterations,
            tolerance=config.tolerance,
            reference_velocity=config.reference_velocity,
        )

        # Use C++ solver if available, else CasADi, else fallback
        if self._backend == "cpp":
            self._solver = CppMPCCSolver(core_config)
        else:
            self._solver = MPCCSolver(core_config)

        self._spline_path: Optional[CubicSplinePath] = None
        self._prev_progress = 0.0
        self._prev_v_cmd = 0.0

    def set_path(self, path: 'ReferencePath'):
        """Convert ReferencePath to CubicSplinePath."""
        try:
            self._spline_path = CubicSplinePath(path.waypoints)
        except Exception:
            # Fall back to linear if spline fails
            self._spline_path = CubicSplinePath(path.waypoints, smooth=False)
        self._solver.reset()
        self._prev_progress = 0.0
        self._prev_v_cmd = 0.0

    def reset_progress(self):
        """Reset forward-progress tracking."""
        self._prev_progress = 0.0
        self._prev_v_cmd = 0.0
        self._solver.reset()

    def solve(self, state: 'VehicleState', path: 'ReferencePath',
              obstacles: List['Obstacle'], current_progress: float) -> Tuple[float, float, np.ndarray]:
        """
        Solve MPCC using pympc_core.

        Returns:
            v_cmd, delta_cmd, predicted_trajectory
        """
        # Enforce forward-only progress
        if current_progress < self._prev_progress - 0.05:
            current_progress = self._prev_progress
        self._prev_progress = current_progress

        # Build spline path if not already done
        if self._spline_path is None:
            self.set_path(path)

        # Convert state to numpy array
        x0 = state.as_array()  # [x, y, theta, v, delta]

        # Convert obstacles to (x, y, radius, vx, vy) tuples
        obs_list = [(obs.x, obs.y, obs.radius, obs.vx, obs.vy) for obs in obstacles]

        # Compute boundary constraints if available
        boundary_constraints = None
        if self.road_boundaries is not None and self.config.boundary_enabled:
            boundary_constraints = []
            for k in range(self.config.horizon):
                s_k = current_progress + k * self.config.reference_velocity * self.config.dt
                ref_x, ref_y = self._spline_path.get_position(
                    min(s_k, self._spline_path.total_length - 0.01))
                theta_ref = self._spline_path.get_tangent(
                    min(s_k, self._spline_path.total_length - 0.01))
                try:
                    A, b_left, b_right = self.road_boundaries.get_boundary_constraints_from_path(
                        ref_x, ref_y, theta_ref,
                        default_width=self.config.boundary_default_width)
                    boundary_constraints.append((A, b_left, b_right))
                except Exception:
                    boundary_constraints.append((np.array([0.0, 1.0]), 1000.0, 1000.0))

        # Solve
        result = self._solver.solve(
            x0, self._spline_path, current_progress,
            obstacles=obs_list,
            boundary_constraints=boundary_constraints,
        )

        v_cmd = result.v_cmd
        delta_cmd = result.delta_cmd

        # The MPCC solver already enforces acceleration and steering rate
        # constraints as part of its optimization. Post-hoc rate limiting
        # here would cause the actual trajectory to diverge from the solver's
        # predicted trajectory, degrading warm-start quality and overall
        # performance. We only clamp to the feasible range.
        v_cmd = max(v_cmd, 0.0)

        return v_cmd, delta_cmd, result.predicted_trajectory

    def _stanley_steering(self, state: 'VehicleState', path: 'ReferencePath',
                          current_progress: float) -> Tuple[float, float, float]:
        """Stanley controller for diagnostic reporting."""
        if self._spline_path is None:
            return 0.0, 0.0, 0.0
        path_tangent = self._spline_path.get_tangent(current_progress)
        ref_x, ref_y = self._spline_path.get_position(current_progress)
        heading_error = self._normalize_angle(path_tangent - state.theta)
        e_cross = (np.cos(path_tangent) * (state.y - ref_y) -
                   np.sin(path_tangent) * (state.x - ref_x))
        return 0.0, heading_error, e_cross

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class MPCCControllerNode(Node):
    """ROS2 Node for MPCC Controller"""

    def __init__(self):
        super().__init__('mpcc_controller')

        # Configuration
        self.config = MPCCConfig()

        # Declare parameters
        self.declare_parameter('reference_velocity', self.config.reference_velocity)
        self.declare_parameter('contour_weight', self.config.contour_weight)
        self.declare_parameter('lag_weight', self.config.lag_weight)
        self.declare_parameter('horizon', self.config.horizon)
        self.declare_parameter('use_direct_motor', True)

        # Update config from parameters
        self.config.reference_velocity = self.get_parameter('reference_velocity').value
        self.config.contour_weight = self.get_parameter('contour_weight').value
        self.config.lag_weight = self.get_parameter('lag_weight').value
        self.config.horizon = self.get_parameter('horizon').value
        self.use_direct_motor = self.get_parameter('use_direct_motor').value and HAS_MOTOR_COMMANDS

        # Load road boundaries
        self.road_boundaries = None
        try:
            pkg_share = get_package_share_directory('acc_stage1_mission')
            boundary_config_path = os.path.join(pkg_share, 'config', 'road_boundaries.yaml')
            if os.path.exists(boundary_config_path):
                self.road_boundaries = RoadBoundarySpline(boundary_config_path)
                self.get_logger().info(f"Loaded road boundaries from {boundary_config_path}")
            else:
                self.get_logger().warn(f"Road boundary config not found: {boundary_config_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load road boundaries: {e}")

        # Initialize solver using pympc_core bridge
        self.solver = MPCCBridge(self.config, self.road_boundaries)
        solver_types = {
            "cpp": "C++ MPCC (Eigen, SQP+gradient projection)",
            "casadi": "CasADi MPCC (pympc_core, fresh Opti per solve)",
            "fallback": "Fallback (Pure Pursuit + Stanley)",
        }
        solver_type = solver_types.get(self.solver._backend, "Unknown")
        self.get_logger().info(f"Using {solver_type}")

        # State
        self.current_state = VehicleState()
        self.reference_path: Optional[ReferencePath] = None
        self.current_progress = 0.0
        self.obstacles: List[Obstacle] = []
        self.motion_enabled = True
        self._motion_disabled_time = 0.0
        self._motion_disable_timeout = 8.0  # Auto-resume after 8s
        self._motion_resume_cooldown_until = 0.0  # Ignore re-disabling until this time
        self._motion_resume_cooldown_s = 5.0
        self._motion_enable_consecutive = 0  # Hysteresis counter for re-enabling
        self._MOTION_ENABLE_HYSTERESIS = 5   # Require 5 consecutive true msgs (~500ms)
        self.has_odom = False
        self.has_path = False

        # Obstacle age timeout: clear obstacles that persist > 5s (phantom detections)
        self._obstacle_first_seen_time: dict = {}  # class -> first_seen_time
        self._obstacle_timeout_s = 5.0

        # Traffic control state
        self._traffic_control_state: Optional[TrafficControlState] = None
        self._needs_path_recompute = False
        self._was_stopped_for_traffic = False
        self._traffic_stop_start_time = 0.0

        # Mission hold state - when True, stop sending commands (dwell at pickup/dropoff)
        self._mission_hold = False

        # TF for getting pose (Cartographer provides odom via TF, not /odom topic)
        # IMPORTANT: We MUST use map frame for position since paths are in map frame
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._last_velocity = 0.0  # Track velocity from position changes
        self._last_pose_time = None
        self._last_x = None
        self._last_y = None
        self._last_odom_msg_time = None  # Track when we last got /odom topic data
        self._odom_timeout = 1.0  # Fall back to TF if /odom is older than this
        self._tf_odom_source = None  # Track which TF source is working
        self._odom_velocity = 0.0  # Velocity from /odom topic (more accurate than TF)

        # Publishers
        # Use /cmd_vel_nav for Twist (debug/legacy mode)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        # Direct MotorCommands publisher (bypasses nav2_qcar_command_convert)
        self.motor_pub = None
        if self.use_direct_motor:
            self.motor_pub = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 1)
        self.viz_pub = self.create_publisher(MarkerArray, '/mpcc/predicted_path', 10)
        self.status_pub = self.create_publisher(String, '/mpcc/status', 10)

        # Subscribers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos)

        # Use transient_local QoS to receive latched path from mission_manager
        path_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.path_sub = self.create_subscription(
            Path, '/plan', self._path_callback, path_qos)

        self.motion_sub = self.create_subscription(
            Bool, '/motion_enable', self._motion_callback, 10)

        self.obstacle_sub = self.create_subscription(
            String, '/obstacle_info', self._obstacle_callback, 10)

        # Traffic control state subscriber
        self.traffic_control_sub = self.create_subscription(
            String, '/traffic_control_state', self._traffic_control_callback, 10)

        # Mission hold subscriber - mission manager signals dwell/stop
        self.hold_sub = self.create_subscription(
            Bool, '/mission/hold', self._hold_callback, 10)

        # JointState subscriber for encoder-based velocity (more accurate)
        self.joint_sub = self.create_subscription(
            JointState, '/qcar2_joint', self._joint_state_callback, qos)
        self._joint_velocity = 0.0
        self._has_joint_velocity = False

        # Stall detection: detect when commands are sent but vehicle doesn't move
        self._stall_cmd_start_time = None  # When we first started sending non-zero cmds
        self._stall_warned = False
        self._stall_timeout = 5.0  # Warn after 5s of non-zero cmds with v=0

        # Control timer (20 Hz)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        # Diagnostic timer (every 2 seconds)
        self._diag_timer = self.create_timer(2.0, self._diagnostic_log)
        self._last_cmd_time = 0
        self._cmd_count = 0

        # Initialize log file
        self._init_log_file()

        motor_mode = "direct MotorCommands" if self.use_direct_motor else "Twist via nav2_qcar_command_convert"
        self.get_logger().info("MPCC Controller initialized")
        self.get_logger().info(f"  Horizon: {self.config.horizon}")
        self.get_logger().info(f"  Reference velocity: {self.config.reference_velocity} m/s")
        self.get_logger().info(f"  Max velocity: {self.config.max_velocity} m/s")
        self.get_logger().info(f"  Motor mode: {motor_mode}")
        self.get_logger().info(f"  Solver: pympc_core ({self.solver._backend})")
        self.get_logger().info(f"  Log file: {self._log_path}")
        self._log(f"MPCC initialized: horizon={self.config.horizon} v_ref={self.config.reference_velocity} "
                  f"solver=pympc_core({self.solver._backend})")

        # Delayed pipeline check (3s after init to allow other nodes to start)
        self._pipeline_check_timer = self.create_timer(3.0, self._check_cmd_pipeline)
        self._pipeline_check_done = False

    def _check_cmd_pipeline(self):
        """Check that command pipeline has downstream subscribers."""
        if self._pipeline_check_done:
            return
        self._pipeline_check_done = True
        self._pipeline_check_timer.cancel()

        if self.use_direct_motor and self.motor_pub is not None:
            subs = self.motor_pub.get_subscription_count()
            if subs == 0:
                msg = (
                    "WARNING: /qcar2_motor_speed_cmd has 0 subscribers! "
                    "qcar2_hardware is not running. "
                    "Vehicle will not move. Ensure Terminal 1 (qcar2_virtual_launch.py) is running."
                )
                self.get_logger().error(msg)
                self._log(msg)
            else:
                self.get_logger().info(f"Command pipeline OK: /qcar2_motor_speed_cmd has {subs} subscriber(s) (direct motor mode)")
                self._log(f"Pipeline check: /qcar2_motor_speed_cmd has {subs} subscriber(s) (direct)")
        else:
            subs = self.cmd_pub.get_subscription_count()
            if subs == 0:
                msg = (
                    "WARNING: /cmd_vel_nav has 0 subscribers! "
                    "nav2_qcar_command_convert is not running. "
                    "Vehicle will not move. Ensure Terminal 1 (qcar2_virtual_launch.py) is running."
                )
                self.get_logger().error(msg)
                self._log(msg)
            else:
                self.get_logger().info(f"Command pipeline OK: /cmd_vel_nav has {subs} subscriber(s)")
                self._log(f"Pipeline check: /cmd_vel_nav has {subs} subscriber(s)")

    def _init_log_file(self):
        """Create timestamped MPCC log file."""
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for candidate in [
            '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/logs',
            os.path.expanduser('~/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission/logs'),
            '/tmp/mission_logs',
        ]:
            try:
                os.makedirs(candidate, exist_ok=True)
                test_path = os.path.join(candidate, '.write_test')
                with open(test_path, 'w') as f:
                    f.write('ok')
                os.remove(test_path)
                log_dir = candidate
                break
            except Exception:
                continue
        else:
            log_dir = '/tmp/mission_logs'
            os.makedirs(log_dir, exist_ok=True)

        self._log_path = os.path.join(log_dir, f'mpcc_{ts}.log')
        self._log_start = time_module.time()
        with open(self._log_path, 'w') as f:
            f.write(f"# MPCC Controller Log - {ts}\n")
            f.write(f"# Format: [time] +elapsed EVENT | details\n\n")

    def _log(self, msg: str):
        """Append a line to the MPCC log file."""
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        elapsed = time_module.time() - self._log_start
        try:
            with open(self._log_path, 'a') as f:
                f.write(f"[{ts}] +{elapsed:7.1f}s  {msg}\n")
        except Exception:
            pass

    def _diagnostic_log(self):
        """Periodic diagnostic logging"""
        status_parts = []

        if not self.has_odom:
            # Provide more detail about odom status
            odom_topic_status = "topic=never" if self._last_odom_msg_time is None else f"topic={self.get_clock().now().nanoseconds / 1e9 - self._last_odom_msg_time:.1f}s ago"
            tf_status = self._tf_odom_source if self._tf_odom_source else "TF=unavailable"
            status_parts.append(f"NO ODOM ({odom_topic_status}, {tf_status})")
        if not self.has_path:
            status_parts.append("NO PATH (waiting for /plan)")
        if not self.motion_enabled:
            status_parts.append("MOTION DISABLED")

        if self._mission_hold:
            status_parts.append("MISSION HOLD (dwell)")

        if status_parts:
            blocked_msg = f"MPCC blocked: {', '.join(status_parts)}"
            self.get_logger().warn(blocked_msg)
            self._log(blocked_msg)
        else:
            # All good - show current state
            progress_pct = 0
            if self.reference_path is not None and self.reference_path.total_length > 0:
                progress_pct = 100 * self.current_progress / self.reference_path.total_length
            odom_src = "topic" if self._last_odom_msg_time and (self.get_clock().now().nanoseconds / 1e9 - self._last_odom_msg_time) < self._odom_timeout else (self._tf_odom_source or "TF")

            # Check boundary status if available
            boundary_status = ""
            if self.road_boundaries is not None and self.reference_path is not None:
                is_violated, violation = self.road_boundaries.check_boundary_violation(
                    self.current_state.x, self.current_state.y, self.current_state.theta)
                if is_violated:
                    boundary_status = f" BOUNDARY_VIOLATION({violation:.2f}m)"
                else:
                    boundary_status = " bounds=OK"

            # Get zone velocity limit
            zone_info = ""
            if self.road_boundaries is not None:
                zone_v = self.road_boundaries.get_velocity_limit(
                    self.current_state.x, self.current_state.y)
                if zone_v < self.config.max_velocity:
                    zone_info = f" zone_vlimit={zone_v:.2f}"

            # Get path target for diagnostics
            path_info = ""
            if self.reference_path is not None and self.current_progress < self.reference_path.total_length:
                target_x, target_y = self.reference_path.get_position_at_progress(
                    min(self.current_progress + 0.3, self.reference_path.total_length - 0.01))
                dx = target_x - self.current_state.x
                dy = target_y - self.current_state.y
                dist_to_path = np.sqrt(dx**2 + dy**2)
                path_info = f" path_dist={dist_to_path:.2f}m"

            active_msg = (
                f"MPCC active: pos=({self.current_state.x:.2f}, {self.current_state.y:.2f}) "
                f"theta={np.degrees(self.current_state.theta):.1f}deg "
                f"v={self.current_state.v:.2f} progress={progress_pct:.0f}% cmds={self._cmd_count} odom={odom_src}{boundary_status}{zone_info}{path_info}"
            )
            self.get_logger().info(active_msg)
            self._log(active_msg)
            self._cmd_count = 0

    def _odom_callback(self, msg: Odometry):
        """
        Update current vehicle state from odometry topic.

        IMPORTANT: We need coordinates in 'map' frame since paths are in 'map' frame.
        The /odom topic may be in 'odom' frame, so we need to transform if necessary.
        For simplicity, we prefer the TF-based update which always uses map frame.
        This callback is kept as a backup and for velocity data.
        """
        # Store velocity from odom message (useful even if position comes from TF)
        self._odom_velocity = msg.twist.twist.linear.x

        # Check if message is already in map frame (odom_from_tf might use map->base_link)
        if msg.header.frame_id == 'map':
            # Direct use - already in map frame
            self.current_state.x = msg.pose.pose.position.x
            self.current_state.y = msg.pose.pose.position.y

            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_state.theta = np.arctan2(siny_cosp, cosy_cosp)
            self.current_state.v = msg.twist.twist.linear.x
            self.has_odom = True
            self._last_odom_msg_time = self.get_clock().now().nanoseconds / 1e9
        else:
            # Odom is in 'odom' frame - we need map->base_link for path following
            # Try to get pose from TF instead
            if self._update_state_from_tf():
                # Use velocity from odom message since TF velocity estimation is noisy
                self.current_state.v = msg.twist.twist.linear.x
                self._last_odom_msg_time = self.get_clock().now().nanoseconds / 1e9

    def _joint_state_callback(self, msg: JointState):
        """Compute actual velocity from encoder ticks (matches reference MPC_node.py)."""
        if msg.velocity:
            encoder_vel = msg.velocity[0]
            v = (encoder_vel / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * math.pi) * 0.033
            self._joint_velocity = v
            self._has_joint_velocity = True

    def _update_state_from_tf(self) -> bool:
        """
        Update vehicle state from TF. Tries multiple frame combinations.
        Returns True if successful.

        Frame priority:
        1. map -> base_link (full localized pose from Cartographer)
        2. odom -> base_link (odometry frame)
        """
        # List of (source_frame, target_frame) to try
        tf_sources = [
            ('map', 'base_link'),
            ('odom', 'base_link'),
        ]

        for source_frame, target_frame in tf_sources:
            try:
                t = self._tf_buffer.lookup_transform(
                    source_frame, target_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.05)
                )

                x = t.transform.translation.x
                y = t.transform.translation.y

                # Extract yaw from quaternion
                q = t.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                theta = np.arctan2(siny_cosp, cosy_cosp)

                # Estimate velocity from position change
                current_time = self.get_clock().now().nanoseconds / 1e9
                if self._last_pose_time is not None and self._last_x is not None:
                    dt = current_time - self._last_pose_time
                    if dt > 0.01:  # Avoid division by very small dt
                        dx = x - self._last_x
                        dy = y - self._last_y
                        self._last_velocity = np.sqrt(dx**2 + dy**2) / dt

                self._last_pose_time = current_time
                self._last_x = x
                self._last_y = y

                # Update state
                self.current_state.x = x
                self.current_state.y = y
                self.current_state.theta = theta
                self.current_state.v = self._last_velocity
                self.has_odom = True

                # Log when TF source changes
                tf_key = f"{source_frame}->{target_frame}"
                if self._tf_odom_source != tf_key:
                    self.get_logger().info(f"MPCC using TF: {tf_key}")
                    self._log(f"ODOM SOURCE: TF {tf_key} pos=({x:.3f}, {y:.3f}) theta={np.degrees(theta):.1f}deg")
                    self._tf_odom_source = tf_key

                return True

            except TransformException:
                continue  # Try next source

        # All sources failed
        if not hasattr(self, '_last_tf_warn') or (self.get_clock().now().nanoseconds / 1e9 - self._last_tf_warn) > 5.0:
            self.get_logger().warn(f"TF lookup failed for all sources: {[f'{s[0]}->{s[1]}' for s in tf_sources]}")
            self._last_tf_warn = self.get_clock().now().nanoseconds / 1e9
        return False

    def _path_callback(self, msg: Path):
        """Update reference path, skipping reset if path is unchanged."""
        if len(msg.poses) < 2:
            self.get_logger().warn(f"Path too short ({len(msg.poses)} poses), need at least 2")
            return

        waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

        # Check if this is the same path we already have (avoid resetting
        # spline/progress/solver on repeated publishes of the same path)
        if self.reference_path is not None and self.has_path:
            same_count = len(waypoints) == len(self.reference_path.waypoints)
            if same_count:
                old_start = self.reference_path.waypoints[0]
                old_end = self.reference_path.waypoints[-1]
                new_start = waypoints[0]
                new_end = waypoints[-1]
                if (abs(new_start[0] - old_start[0]) < 0.01 and
                    abs(new_start[1] - old_start[1]) < 0.01 and
                    abs(new_end[0] - old_end[0]) < 0.01 and
                    abs(new_end[1] - old_end[1]) < 0.01):
                    # Same path — skip full reset
                    return

        self.reference_path = ReferencePath(waypoints)
        self.has_path = True

        # Reset progress to closest point
        self.current_progress = self.reference_path.find_closest_progress(
            self.current_state.x, self.current_state.y)

        # Build spline path and reset solver for new path
        self.solver.set_path(self.reference_path)
        self.solver.reset_progress()

        path_msg = (
            f"PATH RECEIVED: {len(waypoints)} waypoints, length={self.reference_path.total_length:.2f}m, "
            f"from ({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f}) to ({waypoints[-1][0]:.2f}, {waypoints[-1][1]:.2f})"
        )
        self.get_logger().info(path_msg)
        self._log(path_msg)

    def _motion_callback(self, msg: Bool):
        """Handle motion enable/disable with auto-resume timeout and hysteresis.

        Intermittent detections can cause motion_enable to flicker between
        true/false, resetting the auto-resume timer. We require N consecutive
        true messages before re-enabling to prevent this.
        """
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.data:
            self._motion_enable_consecutive += 1
            if self._motion_enable_consecutive >= self._MOTION_ENABLE_HYSTERESIS:
                self.motion_enabled = True
                self._motion_disabled_time = 0.0
                self._motion_enable_consecutive = 0
            # Don't reset motion_disabled_time on single true messages
        else:
            self._motion_enable_consecutive = 0
            # Ignore re-disabling during post-resume cooldown
            if self._motion_resume_cooldown_until > now:
                return
            if self.motion_enabled:
                self._motion_disabled_time = now
            self.motion_enabled = False

    # Classes that are PHYSICAL obstacles requiring avoidance
    # Signs and lights are NOT physical obstacles — they're traffic control
    _OBSTACLE_CLASSES = {'person', 'car', 'motorcycle', 'bus', 'truck',
                         'traffic_cone', 'sports ball'}

    def _obstacle_callback(self, msg: String):
        """Update obstacle list from detection — only physical obstacles."""
        import json
        try:
            data = json.loads(msg.data)
            self.obstacles = []
            current_time = time_module.time()
            seen_classes = set()

            for det in data.get('detections', []):
                obj_class = det.get('class', '')
                # Only add physical obstacles, NOT signs/lights
                if obj_class not in self._OBSTACLE_CLASSES:
                    continue

                dist = det.get('distance')
                if dist and 0.4 < dist < 2.0:  # Ignore <0.4m (noise/self/camera housing/road markings)
                    # Obstacle age timeout: if same class detected continuously for > 5s, ignore it
                    seen_classes.add(obj_class)
                    if obj_class not in self._obstacle_first_seen_time:
                        self._obstacle_first_seen_time[obj_class] = current_time
                    elif (current_time - self._obstacle_first_seen_time[obj_class]) > self._obstacle_timeout_s:
                        # Phantom obstacle timeout — but NEVER ignore pedestrians
                        if obj_class == 'person':
                            pass  # Always respect pedestrian detections
                        else:
                            self._log(f"OBSTACLE TIMEOUT: {obj_class} detected for "
                                      f"{current_time - self._obstacle_first_seen_time[obj_class]:.1f}s, ignoring")
                            continue

                    obs_x = self.current_state.x + dist * np.cos(self.current_state.theta)
                    obs_y = self.current_state.y + dist * np.sin(self.current_state.theta)

                    # Set radius based on object type
                    if obj_class == 'person':
                        radius = 0.25
                    elif obj_class in ('car', 'bus', 'truck'):
                        radius = 0.4
                    else:
                        radius = 0.10  # cones, sports ball — small objects
                    self.obstacles.append(Obstacle(obs_x, obs_y, radius))

            # Clear first-seen time for classes no longer detected
            for cls in list(self._obstacle_first_seen_time.keys()):
                if cls not in seen_classes:
                    del self._obstacle_first_seen_time[cls]

        except json.JSONDecodeError:
            pass

    def _traffic_control_callback(self, msg: String):
        """Handle traffic control state updates."""
        try:
            new_state = TrafficControlState.from_json(msg.data)

            # Log state transitions
            old_state = self._traffic_control_state
            if old_state is not None:
                if old_state.should_stop and not new_state.should_stop:
                    self.get_logger().info(
                        f"Traffic control cleared: {old_state.control_type} -> {new_state.control_type}, can proceed")
                elif not old_state.should_stop and new_state.should_stop:
                    self.get_logger().info(
                        f"Traffic control activated: {new_state.control_type}, should_stop=True")

            self._traffic_control_state = new_state
        except Exception as e:
            self.get_logger().warn(f"Failed to parse traffic control state: {e}")

    def _hold_callback(self, msg: Bool):
        """Handle mission hold signal from mission manager."""
        was_held = self._mission_hold
        self._mission_hold = msg.data
        if msg.data and not was_held:
            self.get_logger().info("Mission HOLD received - stopping")
            self._log("HOLD: mission hold received - stopping")
            self._publish_stop()
        elif not msg.data and was_held:
            self.get_logger().info("Mission HOLD released - resuming")
            self._log("HOLD: released - resuming")

    def _handle_traffic_control(self) -> bool:
        """
        Handle traffic control state (stop signs, traffic lights).

        Returns:
            True if vehicle should stop for traffic control, False otherwise
        """
        if self._traffic_control_state is None:
            return False

        tc = self._traffic_control_state
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Skip traffic control during post-resume cooldown
        if self._motion_resume_cooldown_until > current_time:
            return False

        if tc.should_stop:
            # Track when traffic stop started
            if not self._was_stopped_for_traffic:
                self._traffic_stop_start_time = current_time

            # Timeout: if stopped for traffic for > 10s, force resume
            if (hasattr(self, '_traffic_stop_start_time') and
                    self._traffic_stop_start_time > 0 and
                    (current_time - self._traffic_stop_start_time) > 10.0):
                # Yield signs get longer cooldown to drive past the sign
                cooldown = 8.0 if tc.control_type == "yield_sign" else self._motion_resume_cooldown_s
                self.get_logger().warn(
                    f"Traffic control stop timeout (10s) for {tc.control_type}, "
                    f"force resuming (cooldown={cooldown:.0f}s)")
                self._log(f"TRAFFIC TIMEOUT: force resume after 10s, was={tc.control_type}, cooldown={cooldown:.0f}s")
                self._was_stopped_for_traffic = False
                self._traffic_stop_start_time = 0.0
                self._motion_resume_cooldown_until = current_time + cooldown
                return False

            # Stop the vehicle
            self._publish_stop()

            if tc.control_type == "traffic_light":
                self._publish_status(f"Stopped at RED light (dist={tc.distance:.2f}m)")
                self._log(f"TRAFFIC STOP: red light dist={tc.distance:.2f}m")
            elif tc.control_type == "stop_sign":
                self._publish_status(f"Stop sign: waiting {tc.stop_duration:.1f}s")
                self._log(f"TRAFFIC STOP: stop sign wait={tc.stop_duration:.1f}s")
            elif tc.control_type == "yield_sign":
                self._publish_status(f"Yield sign: checking (dist={tc.distance:.2f}m)")
                self._log(f"TRAFFIC STOP: yield sign dist={tc.distance:.2f}m")

            # Mark that we need path recompute when we resume
            self._needs_path_recompute = True
            self._was_stopped_for_traffic = True
            return True

        elif self._was_stopped_for_traffic and not tc.should_stop:
            # Traffic control cleared - update progress to current position
            # instead of requesting a full path recompute (which causes discontinuities)
            self._was_stopped_for_traffic = False
            if self.reference_path is not None:
                self.current_progress = self.reference_path.find_closest_progress(
                    self.current_state.x, self.current_state.y)
                self.get_logger().info(
                    f"Traffic cleared - resuming from progress={self.current_progress:.2f}/{self.reference_path.total_length:.2f}")
                # Only request recompute if we're very far from the path
                ref_x, ref_y = self.reference_path.get_position_at_progress(self.current_progress)
                dist_to_path = np.sqrt(
                    (self.current_state.x - ref_x)**2 + (self.current_state.y - ref_y)**2)
                if dist_to_path > 0.5:
                    self.get_logger().info(
                        f"Far from path ({dist_to_path:.2f}m) - requesting recompute")
                    self._request_path_recompute()

        return False

    def _request_path_recompute(self):
        """Request mission manager to recompute path from current position."""
        # Signal that MPCC needs a new path by publishing status
        # Mission manager will detect this and resend the goal
        self._publish_status("Requesting path recompute after traffic control")
        self.get_logger().info("Traffic control cleared - requesting path recompute")

    def _control_loop(self):
        """Main control loop"""
        # IMPORTANT: We need position in 'map' frame since paths are in 'map' frame.
        # Always prefer TF (map->base_link) over /odom topic for position.
        # Use /odom topic only for velocity (which is in body frame, so frame doesn't matter).

        # Always try to update from TF first (gives us map frame coordinates)
        tf_success = self._update_state_from_tf()

        if not tf_success:
            # TF failed - check if we have recent /odom data as backup
            current_time = self.get_clock().now().nanoseconds / 1e9
            odom_stale = (
                self._last_odom_msg_time is None or
                (current_time - self._last_odom_msg_time) > self._odom_timeout
            )
            if not self.has_odom or odom_stale:
                # No odometry from any source - can't control
                return

        if not self.has_odom:
            return

        # Use best available velocity source:
        # 1. JointState encoder (most accurate, like reference MPC_node.py)
        # 2. Odom topic velocity
        # 3. TF-derived velocity (least accurate)
        if self._has_joint_velocity:
            self.current_state.v = self._joint_velocity
        elif self._odom_velocity != 0.0 or self._last_odom_msg_time is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self._last_odom_msg_time and (current_time - self._last_odom_msg_time) < 0.5:
                self.current_state.v = self._odom_velocity

        if not self.has_path or self.reference_path is None:
            return

        # Check mission hold (dwell at pickup/dropoff)
        if self._mission_hold:
            self._publish_stop()
            return

        if not self.motion_enabled:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if (self._motion_disabled_time > 0 and
                    (current_time - self._motion_disabled_time) > self._motion_disable_timeout):
                self.get_logger().warn(
                    f"motion_enable false for {current_time - self._motion_disabled_time:.1f}s, "
                    f"auto-resuming (timeout={self._motion_disable_timeout:.0f}s)")
                self._log(f"MOTION AUTO-RESUME: after {current_time - self._motion_disabled_time:.1f}s")
                self.motion_enabled = True
                self._motion_disabled_time = 0.0
                self._motion_resume_cooldown_until = current_time + self._motion_resume_cooldown_s
            else:
                self._publish_stop()
                return

        # Check traffic control state (stop signs, traffic lights)
        if self._handle_traffic_control():
            return  # Stopped for traffic control

        # Update progress (enforce forward-only to prevent oscillation)
        new_progress = self.reference_path.find_closest_progress(
            self.current_state.x, self.current_state.y)
        # Allow small backward adjustment (0.05m) for noise, but prevent large jumps
        if new_progress >= self.current_progress - 0.05:
            self.current_progress = new_progress

        self._cmd_count += 1

        # Check if reached end of path
        remaining_distance = self.reference_path.total_length - self.current_progress
        if remaining_distance < 0.15:
            self._publish_stop()
            self._publish_status("Goal reached")
            self._log(f"GOAL REACHED: remaining={remaining_distance:.3f}m progress={self.current_progress:.2f}/{self.reference_path.total_length:.2f}")
            return

        # Apply velocity zone limits from road boundaries
        zone_velocity_limit = self.config.max_velocity
        if self.road_boundaries is not None:
            zone_velocity_limit = self.road_boundaries.get_velocity_limit(
                self.current_state.x, self.current_state.y)

        # Solve MPCC
        v_cmd, delta_cmd, predicted = self.solver.solve(
            self.current_state,
            self.reference_path,
            self.obstacles,
            self.current_progress
        )

        # Log tracking diagnostics at ~1Hz (every 20th cycle)
        if self._cmd_count % 20 == 0:
            _, heading_err, cross_track = self.solver._stanley_steering(
                self.current_state, self.reference_path, self.current_progress)
            if abs(heading_err) > np.pi / 6 or abs(cross_track) > 0.15:
                self._log(
                    f"TRACKING: h_err={np.degrees(heading_err):.1f}deg "
                    f"xtrack={cross_track:.3f}m"
                )

        # Apply safety limits
        # Decelerate near the goal for a clean full stop
        if remaining_distance < 0.5:
            decel_factor = remaining_distance / 0.5
            v_cmd = v_cmd * decel_factor
            if remaining_distance > 0.2:
                v_cmd = max(v_cmd, 0.08)

        # Clamp velocity to safe range, respecting zone limits
        effective_max_v = min(self.config.max_velocity, zone_velocity_limit)
        v_cmd = np.clip(v_cmd, 0.0, effective_max_v)

        # Clamp steering angle to prevent extreme values
        delta_cmd = np.clip(delta_cmd, -self.config.max_steering, self.config.max_steering)

        # Publish MotorCommands directly (bypass nav2_qcar_command_convert)
        if self.use_direct_motor and self.motor_pub is not None:
            motor_cmd = MotorCommands()
            motor_cmd.motor_names = ['steering_angle', 'motor_throttle']
            motor_cmd.values = [float(delta_cmd), float(v_cmd)]
            self.motor_pub.publish(motor_cmd)

        # Also publish Twist for debug/visualization (and legacy mode)
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        omega = v_cmd * np.tan(delta_cmd) / self.config.wheelbase
        omega = np.clip(omega, -1.5, 1.5)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # Stall detection: warn if sending commands but vehicle isn't moving
        now = self.get_clock().now().nanoseconds / 1e9
        if v_cmd > 0.05:
            if self._stall_cmd_start_time is None:
                self._stall_cmd_start_time = now
            elif abs(self.current_state.v) < 0.01 and (now - self._stall_cmd_start_time) > self._stall_timeout:
                if not self._stall_warned:
                    self._stall_warned = True
                    if self.use_direct_motor and self.motor_pub is not None:
                        subs = self.motor_pub.get_subscription_count()
                        topic_name = "/qcar2_motor_speed_cmd"
                    else:
                        subs = self.cmd_pub.get_subscription_count()
                        topic_name = "/cmd_vel_nav"
                    warn_msg = (
                        f"STALL DETECTED: Publishing v={v_cmd:.2f} to {topic_name} but vehicle v=0.00 "
                        f"for {now - self._stall_cmd_start_time:.0f}s. "
                        f"{topic_name} has {subs} subscriber(s). "
                    )
                    if subs == 0:
                        warn_msg += f"No subscribers! Check Terminal 1 (qcar2_virtual_launch.py)."
                    else:
                        warn_msg += "Subscribers present - check qcar2_hardware pipeline, or QLabs connection."
                    self.get_logger().error(warn_msg)
                    self._log(warn_msg)
        else:
            self._stall_cmd_start_time = None
            self._stall_warned = False

        # Log commands at ~1Hz (every 20th cycle)
        if self._cmd_count % 20 == 1:
            progress_pct = 100 * self.current_progress / self.reference_path.total_length if self.reference_path.total_length > 0 else 0
            # Include heading error and cross-track for diagnostics
            xtrack_str = ""
            herr_str = ""
            _, he, xt = self.solver._stanley_steering(
                self.current_state, self.reference_path, self.current_progress)
            herr_str = f" h_err={np.degrees(he):.1f}deg"
            xtrack_str = f" xtrack={xt:.3f}m"
            self._log(
                f"CMD: v={cmd.linear.x:.3f} omega={cmd.angular.z:.3f} delta={np.degrees(delta_cmd):.1f}deg "
                f"pos=({self.current_state.x:.3f},{self.current_state.y:.3f}) "
                f"remaining={remaining_distance:.2f}m progress={progress_pct:.0f}%{herr_str}{xtrack_str}"
            )

        # Publish visualization
        self._publish_predicted_path(predicted)

        # Include more diagnostic info in status
        progress_pct = 100 * self.current_progress / self.reference_path.total_length if self.reference_path.total_length > 0 else 0
        motor_str = "direct" if self.use_direct_motor else "twist"
        self._publish_status(f"v={v_cmd:.2f}, delta={np.degrees(delta_cmd):.1f}deg, progress={progress_pct:.0f}%, motor={motor_str}")

    def _publish_stop(self):
        """Publish zero velocity on all command channels."""
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        if self.use_direct_motor and self.motor_pub is not None:
            motor_cmd = MotorCommands()
            motor_cmd.motor_names = ['steering_angle', 'motor_throttle']
            motor_cmd.values = [0.0, 0.0]
            self.motor_pub.publish(motor_cmd)

    def _publish_predicted_path(self, predicted: np.ndarray):
        """Publish predicted trajectory for visualization"""
        markers = MarkerArray()

        # Path line
        marker = Marker()
        marker.header.frame_id = 'map'  # Must match path frame for proper visualization
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'mpcc_predicted'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        for i in range(len(predicted)):
            p = Point()
            p.x = float(predicted[i, 0])
            p.y = float(predicted[i, 1])
            p.z = 0.1
            marker.points.append(p)

        markers.markers.append(marker)
        self.viz_pub.publish(markers)

    def _publish_status(self, status: str):
        """Publish controller status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = MPCCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
