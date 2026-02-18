#!/usr/bin/env python3
"""
Stage I mission manager for Quanser ACC contest (Virtual ROS / QLabs).

Features:
- Nav2 NavigateToPose: pickup -> [optional waypoints] -> dropoff -> return to hub
- Obstacle/sign detection integration via /motion_enable topic
- Sophisticated retry logic with multiple recovery strategies
- LED status via qcar2_hardware
- Mission status publisher on /mission/status

Config from mission.yaml; hub can be captured from TF (map->base_link) at startup.
"""
import datetime
import math
import os
import time
import yaml
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav2_msgs.action import NavigateToPose, ComputePathToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String
import tf2_ros
from tf2_ros import TransformException
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from action_msgs.msg import GoalStatus
from rclpy.qos import DurabilityPolicy

from acc_stage1_mission.road_graph import RoadGraph, qlabs_path_to_map_path


def yaw_to_quat(yaw: float):
    """Convert yaw (rad) to quaternion (x, y, z, w); only z,w used for 2D."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rad) from quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _load_mission_config(config_path: str) -> dict:
    """Load mission.yaml; return dict with pickup, dropoff, hub, waypoints, dwell_s, goal_tol_m."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def _tuple_from_xy_yaw(d: dict, key: str) -> tuple:
    """Return (x, y, yaw) from config section {x, y, yaw}."""
    return (float(d[key]["x"]), float(d[key]["y"]), float(d[key]["yaw"]))


def _parse_waypoints(cfg: dict, key: str = "waypoints") -> list:
    """Return list of (x, y, yaw) from config waypoints (optional)."""
    out = []
    for w in cfg.get(key, []):
        out.append((float(w["x"]), float(w["y"]), float(w.get("yaw", 0.0))))
    return out


# LED color IDs for qcar2_hardware (see qcar2_hardware.cpp LED_Set())
# 0=Red, 1=Green, 2=Blue, 3=Yellow, 4=Cyan, 5=Magenta, 6=Orange, 7=White
# Per scenario requirements:
# - Magenta at hub/start/return
# - Green while driving
# - Blue at pickup (stopped)
# - Orange at dropoff (stopped)
LED_RED = 0       # Red
LED_GREEN = 1     # Green - driving to destination
LED_BLUE = 2      # Blue - at pickup, stopped
LED_YELLOW = 3    # Yellow
LED_CYAN = 4      # Cyan - recovering
LED_MAGENTA = 5   # Magenta - at hub (start/end)
LED_ORANGE = 6    # Orange - at dropoff, stopped

# Scenario-specific LED assignments
LED_HUB = LED_MAGENTA      # At taxi hub
LED_DRIVING = LED_GREEN    # While driving
LED_PICKUP = LED_BLUE      # At pickup, stopped
LED_DROPOFF = LED_ORANGE   # At dropoff, stopped (scenario requires Orange)
LED_OBSTACLE = LED_RED     # Obstacle detected
LED_RECOVERY = LED_CYAN    # Recovering from error


def qlabs_to_map_frame(qlabs_x: float, qlabs_y: float, qlabs_yaw: float,
                        origin_x: float = -1.205, origin_y: float = -0.83,
                        origin_heading_deg: float = -44.7,
                        origin_heading_rad: float = None) -> tuple:
    """
    Transform QLabs world coordinates to Cartographer map frame.

    Cartographer builds its map starting from where the car first initialized.
    The map frame origin is at the car's starting position, with X-axis aligned
    to the car's initial heading.

    Args:
        qlabs_x, qlabs_y: Position in QLabs world coordinates
        qlabs_yaw: Heading in QLabs world (radians)
        origin_x, origin_y: Car's starting position in QLabs
        origin_heading_deg: Car's initial heading in QLabs (degrees)
        origin_heading_rad: Calibrated transform angle (radians). When provided,
            used directly instead of deriving from origin_heading_deg.

    Returns:
        (map_x, map_y, map_yaw): Position and heading in Cartographer map frame
    """
    # Step 1: Translate to car's starting position
    translated_x = qlabs_x - origin_x
    translated_y = qlabs_y - origin_y

    # Step 2: Rotate by the calibrated transform angle
    # Use origin_heading_rad directly when available (empirically calibrated),
    # fall back to degrees conversion for backward compatibility.
    if origin_heading_rad is not None:
        theta = origin_heading_rad
    else:
        theta = math.radians(-origin_heading_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    map_x = translated_x * cos_t + translated_y * sin_t
    map_y = -translated_x * sin_t + translated_y * cos_t

    # Step 3: Transform heading
    map_yaw = qlabs_yaw - (-theta)
    # Normalize to [-pi, pi]
    while map_yaw > math.pi:
        map_yaw -= 2 * math.pi
    while map_yaw < -math.pi:
        map_yaw += 2 * math.pi

    return (map_x, map_y, map_yaw)


def map_to_qlabs_frame(map_x: float, map_y: float, map_yaw: float,
                        origin_x: float = -1.205, origin_y: float = -0.83,
                        origin_heading_deg: float = -44.7,
                        origin_heading_rad: float = None) -> tuple:
    """
    Transform Cartographer map frame coordinates to QLabs world coordinates.

    Inverse of qlabs_to_map_frame.

    Returns:
        (qlabs_x, qlabs_y, qlabs_yaw): Position and heading in QLabs world frame
    """
    # Step 1: Rotate back by R(θ) — the inverse of R(θ)^T used in qlabs_to_map_frame
    if origin_heading_rad is not None:
        theta = origin_heading_rad
    else:
        theta = math.radians(-origin_heading_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rotated_x = map_x * cos_t - map_y * sin_t
    rotated_y = map_x * sin_t + map_y * cos_t

    # Step 2: Translate back to QLabs origin
    qlabs_x = rotated_x + origin_x
    qlabs_y = rotated_y + origin_y

    # Step 3: Transform heading
    qlabs_yaw = map_yaw + (-theta)
    # Normalize to [-pi, pi]
    while qlabs_yaw > math.pi:
        qlabs_yaw -= 2 * math.pi
    while qlabs_yaw < -math.pi:
        qlabs_yaw += 2 * math.pi

    return (qlabs_x, qlabs_y, qlabs_yaw)


class RecoveryStrategy(Enum):
    """Recovery strategies when navigation fails."""
    RETRY_SAME = auto()       # Retry same goal
    BACKUP_AND_RETRY = auto() # Back up slightly, then retry
    CLEAR_COSTMAP = auto()    # Clear costmaps and retry
    SKIP_WAYPOINT = auto()    # Skip current waypoint (if applicable)
    RESTART_MISSION = auto()  # Restart from pickup


class MissionState(Enum):
    """Mission state machine states."""
    WAIT_FOR_NAV = auto()
    CAPTURING_HUB = auto()
    GO_LEG = auto()
    PAUSED_OBSTACLE = auto()
    DWELL = auto()
    RECOVERING = auto()
    DONE = auto()
    ABORTED = auto()


class MissionManager(Node):
    """
    Stage I mission manager using Nav2:
      pickup -> [waypoints] -> dropoff -> hub.

    Features:
    - Obstacle detection integration (subscribes to /motion_enable)
    - Sophisticated retry logic with multiple recovery strategies
    - LED status feedback
    """

    def __init__(self):
        super().__init__("mission_manager")

        # Declare parameters
        self.declare_parameter("config_file", "")
        self.declare_parameter("use_tf_hub", True)
        self.declare_parameter("hub_tf_timeout_s", 15.0)
        self.declare_parameter("goal_timeout_s", 120.0)
        self.declare_parameter("max_retries_per_leg", 2)
        self.declare_parameter("enable_led", True)
        self.declare_parameter("goal_tol_m", 0.35)
        self.declare_parameter("enable_obstacle_detection", True)
        self.declare_parameter("obstacle_pause_timeout_s", 30.0)
        self.declare_parameter("backup_distance_m", 0.15)
        self.declare_parameter("backup_speed", 0.1)
        # MPCC mode: use ComputePathToPose instead of NavigateToPose
        # This lets MPCC controller handle path following instead of Nav2's controller
        self.declare_parameter("mpcc_mode", False)

        # Get parameters
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        self._use_tf_hub = self.get_parameter("use_tf_hub").get_parameter_value().bool_value
        self._hub_tf_timeout_s = self.get_parameter("hub_tf_timeout_s").get_parameter_value().double_value
        self._goal_timeout_s = self.get_parameter("goal_timeout_s").get_parameter_value().double_value
        self._max_retries = int(self.get_parameter("max_retries_per_leg").get_parameter_value().integer_value)
        self._enable_led = self.get_parameter("enable_led").get_parameter_value().bool_value
        self._goal_tol_m = self.get_parameter("goal_tol_m").get_parameter_value().double_value
        self._enable_obstacle_detection = self.get_parameter("enable_obstacle_detection").get_parameter_value().bool_value
        self._obstacle_pause_timeout_s = self.get_parameter("obstacle_pause_timeout_s").get_parameter_value().double_value
        self._backup_distance_m = self.get_parameter("backup_distance_m").get_parameter_value().double_value
        self._backup_speed = self.get_parameter("backup_speed").get_parameter_value().double_value
        self._mpcc_mode = self.get_parameter("mpcc_mode").get_parameter_value().bool_value

        # Load config
        if not config_file:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory("acc_stage1_mission")
            config_file = os.path.join(pkg_share, "config", "mission.yaml")

        if not os.path.isfile(config_file):
            self.get_logger().error("Mission config not found: %s" % config_file)
            # Default to QLabs scenario coordinates (will be transformed below)
            self.pickup = (0.125, 4.395, 1.57)
            self.dropoff = (-0.905, 0.800, 0.0)
            self.hub = (-1.205, -0.83, -0.78)
            self.dwell_s = 3.0
            waypoint_dwell_s = 0.0
            pickup_waypoints = []
            dropoff_waypoints = []
            return_waypoints = []
            use_qlabs_coords = True
            transform_params = {"origin_x": -1.205, "origin_y": -0.83,
                                "origin_heading_deg": -44.7, "origin_heading_rad": 0.7177}
        else:
            cfg = _load_mission_config(config_file)
            self.pickup = _tuple_from_xy_yaw(cfg, "pickup")
            self.dropoff = _tuple_from_xy_yaw(cfg, "dropoff")
            self.hub = _tuple_from_xy_yaw(cfg, "hub")
            self.dwell_s = float(cfg.get("dwell_s", 3.0))
            waypoint_dwell_s = float(cfg.get("waypoint_dwell_s", 0.0))
            # Parse waypoint lists (new format with separate lists, or legacy single list)
            pickup_waypoints = _parse_waypoints(cfg, "pickup_waypoints")
            dropoff_waypoints = _parse_waypoints(cfg, "dropoff_waypoints")
            return_waypoints = _parse_waypoints(cfg, "return_waypoints")
            # Fallback to legacy 'waypoints' key if new keys not present
            if not pickup_waypoints and not dropoff_waypoints and not return_waypoints:
                pickup_waypoints = _parse_waypoints(cfg, "waypoints")
            if "goal_tol_m" in cfg:
                self._goal_tol_m = float(cfg["goal_tol_m"])

            # Check if using QLabs coordinates that need transformation
            use_qlabs_coords = cfg.get("use_qlabs_coords", False)
            transform_cfg = cfg.get("transform", {})
            transform_params = {
                "origin_x": float(transform_cfg.get("origin_x", -1.205)),
                "origin_y": float(transform_cfg.get("origin_y", -0.83)),
                "origin_heading_deg": float(transform_cfg.get("origin_heading_deg", -44.7)),
            }
            # Use calibrated radian value when available
            if "origin_heading_rad" in transform_cfg:
                transform_params["origin_heading_rad"] = float(transform_cfg["origin_heading_rad"])

        # Transform coordinates from QLabs to map frame if needed
        if use_qlabs_coords:
            self.get_logger().info("Transforming QLabs coordinates to map frame...")
            self.get_logger().info(f"  Transform params: origin=({transform_params['origin_x']:.3f}, {transform_params['origin_y']:.3f}), heading={transform_params['origin_heading_deg']:.1f}deg")

            # Transform main locations
            self.pickup = qlabs_to_map_frame(*self.pickup, **transform_params)
            self.dropoff = qlabs_to_map_frame(*self.dropoff, **transform_params)
            self.hub = qlabs_to_map_frame(*self.hub, **transform_params)

            self.get_logger().info(f"  Pickup (map): ({self.pickup[0]:.3f}, {self.pickup[1]:.3f})")
            self.get_logger().info(f"  Dropoff (map): ({self.dropoff[0]:.3f}, {self.dropoff[1]:.3f})")
            self.get_logger().info(f"  Hub (map): ({self.hub[0]:.3f}, {self.hub[1]:.3f})")

            # Transform waypoints
            pickup_waypoints = [qlabs_to_map_frame(*w, **transform_params) for w in pickup_waypoints]
            dropoff_waypoints = [qlabs_to_map_frame(*w, **transform_params) for w in dropoff_waypoints]
            return_waypoints = [qlabs_to_map_frame(*w, **transform_params) for w in return_waypoints]

        # Store transform params for later use
        self._transform_params = transform_params
        self._use_qlabs_coords = use_qlabs_coords

        # Build legs: (target (x,y,yaw), label, dwell_after_s, is_skippable)
        self.legs = []

        if self._mpcc_mode:
            # MPCC mode with road graph: use only 3 direct legs.
            # The road graph generates the complete lane-following path for
            # each leg, so intermediate waypoints are not needed and actually
            # break route selection.
            self.legs.append((self.pickup, "pickup", self.dwell_s, False))
            self.legs.append((self.dropoff, "dropoff", self.dwell_s, False))
            self.legs.append((self.hub, "hub", 0.0, False))

            # Map each leg to a road graph route name
            self._leg_route_names = [
                'hub_to_pickup',
                'pickup_to_dropoff',
                'dropoff_to_hub',
            ]
            # Track previous leg label for route selection
            self._prev_leg_label = 'hub'
        else:
            # Nav2 mode: use intermediate waypoints for turn-by-turn guidance
            for i, w in enumerate(pickup_waypoints):
                self.legs.append((w, "to_pickup_wp%d" % (i + 1), waypoint_dwell_s, True))
            self.legs.append((self.pickup, "pickup", self.dwell_s, False))

            for i, w in enumerate(dropoff_waypoints):
                self.legs.append((w, "to_dropoff_wp%d" % (i + 1), waypoint_dwell_s, True))
            self.legs.append((self.dropoff, "dropoff", self.dwell_s, False))

            for i, w in enumerate(return_waypoints):
                self.legs.append((w, "to_hub_wp%d" % (i + 1), waypoint_dwell_s, True))
            self.legs.append((self.hub, "hub", 0.0, False))

        # Callback group for concurrent callbacks
        self._cb_group = ReentrantCallbackGroup()

        # Nav2 action clients
        if self._mpcc_mode:
            # MPCC mode: use road-graph-based path planner (bypasses Nav2 NavFn)
            # Initialize road graph for lane-following path generation
            self._road_graph = RoadGraph(ds=0.01)  # 1cm waypoint spacing
            self.get_logger().info("Road graph initialized with routes: %s" %
                                   self._road_graph.get_route_names())

            # Still create Nav2 client as fallback
            self.nav_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose",
                                            callback_group=self._cb_group)
            # Use transient_local durability so late subscribers (MPCC controller) still receive the path
            path_qos = QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE
            )
            self._path_pub = self.create_publisher(Path, "/plan", path_qos)
            self._current_path = None  # Store path for periodic republishing
            self.get_logger().info("MPCC MODE: Road-graph path planner (lane-following)")
        else:
            # Normal mode: use Nav2's full navigation (planning + controlling)
            self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose",
                                            callback_group=self._cb_group)
            self._path_pub = None
            self._road_graph = None

        self.goal_handle = None
        self.goal_sent_time = None
        self._navigation_paused = False
        self._goal_result_received = False
        self._last_goal_status = None
        self._current_target = None  # For MPCC mode goal checking

        # State machine
        self.state = MissionState.WAIT_FOR_NAV
        self.leg_index = 0
        self.retry_count = 0
        self._recovery_strategy_index = 0

        # Timing
        self.stop_until = 0.0
        self._obstacle_pause_start = None
        self._backup_start_time = None

        # Resume cooldown: prevent rapid obstacle pause/resume cycling
        self._last_resume_time = 0.0
        self._resume_cooldown_s = 2.0

        # Track time of last goal result for fast-fail detection
        self._last_goal_result_time = None

        # TF for hub capture and position tracking
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publishers
        self._status_pub = self.create_publisher(String, "mission/status", 10)
        self._cmd_vel_pub = self.create_publisher(Twist, "cmd_vel_nav", 10)

        # LED client
        self._led_client = None
        if self._enable_led:
            self._led_client = self.create_client(SetParameters, "/qcar2_hardware/set_parameters")

        # Obstacle detection subscriber
        self._motion_enabled = True
        if self._enable_obstacle_detection:
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._motion_sub = self.create_subscription(
                Bool, "/motion_enable", self._motion_enable_cb, qos,
                callback_group=self._cb_group
            )
            self.get_logger().info("Obstacle detection enabled - subscribing to /motion_enable")

        # MPCC hold publisher - tells MPCC controller to stop sending commands
        if self._mpcc_mode:
            self._hold_pub = self.create_publisher(Bool, "/mission/hold", 10)

        # MPCC status subscriber (for path recomputation requests and goal reached)
        if self._mpcc_mode:
            self._mpcc_status_sub = self.create_subscription(
                String, "/mpcc/status", self._mpcc_status_callback, 10,
                callback_group=self._cb_group
            )
            self.get_logger().info("Subscribed to /mpcc/status for path recompute requests")

        # Traffic control state subscriber — log stop-sign / traffic-light
        # events into the unified behavior log
        self._traffic_sub = self.create_subscription(
            String, "/traffic_control_state", self._traffic_control_behavior_cb, 10,
            callback_group=self._cb_group
        )
        self._last_traffic_should_stop = False
        self._last_traffic_type = "none"

        # Main timer
        self.timer = self.create_timer(0.1, self.tick, callback_group=self._cb_group)

        self.get_logger().info("=" * 60)
        self.get_logger().info("MissionManager Configuration - ACC Competition")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Mode: {'MPCC (path only)' if self._mpcc_mode else 'Nav2 (full navigation)'}")
        self.get_logger().info(f"  Legs: {len(self.legs)}")
        self.get_logger().info(f"  Goal tolerance: {self._goal_tol_m}m")
        self.get_logger().info(f"  Obstacle detection: {self._enable_obstacle_detection}")
        self.get_logger().info("")
        self.get_logger().info("Mission Locations (Map Frame):")
        self.get_logger().info(f"  Pickup:  ({self.pickup[0]:.3f}, {self.pickup[1]:.3f}) yaw={math.degrees(self.pickup[2]):.1f}deg")
        self.get_logger().info(f"  Dropoff: ({self.dropoff[0]:.3f}, {self.dropoff[1]:.3f}) yaw={math.degrees(self.dropoff[2]):.1f}deg")
        self.get_logger().info(f"  Hub:     ({self.hub[0]:.3f}, {self.hub[1]:.3f}) yaw={math.degrees(self.hub[2]):.1f}deg")
        self.get_logger().info("")
        self.get_logger().info("Mission Legs:")
        for i, (target, label, dwell, skip) in enumerate(self.legs):
            skip_str = "(skippable)" if skip else "(required)"
            dwell_str = f"dwell={dwell:.1f}s" if dwell > 0 else ""
            self.get_logger().info(f"  {i}: {label:20s} -> ({target[0]:6.2f}, {target[1]:6.2f}) {dwell_str} {skip_str}")
        self.get_logger().info("=" * 60)

        # Initialize log files for behavior events and coordinate trace
        self._init_log_files()

        # Coordinate logging timer (1 Hz) - logs position for trajectory analysis
        self._coord_log_timer = self.create_timer(1.0, self._log_coordinates)

        self._publish_status("WAIT_FOR_NAV")

    # -------------------------------------------------------------------------
    # Logging - Behavior Events & Coordinate Trace
    # -------------------------------------------------------------------------

    def _init_log_files(self):
        """Create timestamped log files for behavior events and coordinate trace."""
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Write logs to the package source tree so they're visible on the host
        # (the Development/ dir is bind-mounted into the container)
        # Try multiple paths: container mount, then host path, then /tmp fallback
        for candidate in [
            '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/logs',
            os.path.expanduser('~/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission/logs'),
            '/tmp/mission_logs',
        ]:
            try:
                os.makedirs(candidate, exist_ok=True)
                # Test that we can actually write
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

        self._behavior_log_path = os.path.join(log_dir, f'behavior_{ts}.log')
        self._coord_log_path = os.path.join(log_dir, f'coordinates_{ts}.csv')

        with open(self._behavior_log_path, 'w') as f:
            f.write(f"# Mission Behavior Log - {ts}\n")
            f.write(f"# Format: [time] EVENT | details\n\n")

        with open(self._coord_log_path, 'w') as f:
            f.write("time,elapsed_s,state,map_x,map_y,map_yaw_deg,"
                    "qlabs_x,qlabs_y,qlabs_yaw_deg,"
                    "target_label,dist_to_target\n")

        self._log_start_time = time.time()

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Behavior log : {self._behavior_log_path}")
        self.get_logger().info(f"  Coordinate log: {self._coord_log_path}")
        self.get_logger().info("=" * 60)

    def _log_behavior(self, event: str, details: str = ""):
        """Append a timestamped behavior event to the log file."""
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        elapsed = time.time() - self._log_start_time
        line = f"[{ts}] +{elapsed:7.1f}s  {event:25s} | {details}\n"
        try:
            with open(self._behavior_log_path, 'a') as f:
                f.write(line)
        except Exception:
            pass
        self.get_logger().info(f"EVENT: {event} | {details}")

    def _log_coordinates(self):
        """Log current position at 1 Hz for trajectory analysis."""
        pose = self._get_current_pose()
        if pose is None:
            return

        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        elapsed = time.time() - self._log_start_time
        map_x, map_y, map_yaw = pose

        # Convert to QLabs frame for easier interpretation
        if self._use_qlabs_coords:
            qlabs_x, qlabs_y, qlabs_yaw = map_to_qlabs_frame(
                map_x, map_y, map_yaw, **self._transform_params)
        else:
            qlabs_x, qlabs_y, qlabs_yaw = map_x, map_y, map_yaw

        # Current target info
        target_label = ""
        dist_to_target = -1.0
        if self.leg_index < len(self.legs):
            target, label, _, _ = self.legs[self.leg_index]
            target_label = label
            dist_to_target = math.sqrt(
                (target[0] - map_x)**2 + (target[1] - map_y)**2)

        state_name = self.state.name

        line = (f"{ts},{elapsed:.1f},{state_name},"
                f"{map_x:.4f},{map_y:.4f},{math.degrees(map_yaw):.1f},"
                f"{qlabs_x:.4f},{qlabs_y:.4f},{math.degrees(qlabs_yaw):.1f},"
                f"{target_label},{dist_to_target:.3f}\n")
        try:
            with open(self._coord_log_path, 'a') as f:
                f.write(line)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Obstacle Detection
    # -------------------------------------------------------------------------

    def _motion_enable_cb(self, msg: Bool):
        """Callback for /motion_enable topic from YOLO/traffic detector."""
        was_enabled = self._motion_enabled
        self._motion_enabled = msg.data

        if was_enabled and not self._motion_enabled:
            self.get_logger().info("Obstacle/sign detected - motion DISABLED")
            self._log_behavior("OBSTACLE_DETECTED", "motion disabled")
            self._set_led(LED_OBSTACLE)  # Red for obstacle
        elif not was_enabled and self._motion_enabled:
            self.get_logger().info("Obstacle cleared - motion ENABLED")
            self._log_behavior("OBSTACLE_CLEARED", "motion re-enabled")
            if self.state == MissionState.PAUSED_OBSTACLE:
                self._resume_from_pause()

    def _mpcc_status_callback(self, msg: String):
        """
        Callback for MPCC status messages.

        Handles:
        - Path recomputation requests when traffic controls clear
        - Goal reached notification from MPCC controller
        """
        if "Goal reached" in msg.data:
            if self.state == MissionState.GO_LEG and self._mpcc_mode:
                self.get_logger().info("MPCC reports goal reached via status")
                # Verify with distance check before accepting
                current_pose = self._get_current_pose()
                if current_pose is not None and self._current_target is not None:
                    target_x, target_y, _ = self._current_target
                    cur_x, cur_y, _ = current_pose
                    dist = math.sqrt((target_x - cur_x)**2 + (target_y - cur_y)**2)
                    # Use a more generous tolerance for MPCC-reported goal reached
                    if dist < self._goal_tol_m * 2.0:
                        self.get_logger().info("MPCC goal confirmed (dist=%.3f)" % dist)
                        self._log_behavior("MPCC_GOAL_CONFIRMED", f"dist={dist:.3f}")
                        self.goal_sent_time = None
                        self._on_goal_success()
                        return
                    else:
                        self.get_logger().info("MPCC goal report but dist=%.3f > tol, waiting..." % dist)

        elif "Requesting path recompute" in msg.data:
            self.get_logger().info("MPCC requested path recompute - resending goal")
            self._log_behavior("PATH_RECOMPUTE", "MPCC requested path recompute")
            if self.state == MissionState.GO_LEG and self._mpcc_mode:
                # Recompute path from current position to current target
                target, label, _, _ = self.legs[self.leg_index]
                self.send_goal(target, label + "_recompute")

    def _traffic_control_behavior_cb(self, msg: String):
        """Log traffic control state changes to the unified behavior log."""
        import json
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        should_stop = data.get("should_stop", False)
        ctrl_type = data.get("control_type", "none")
        distance = data.get("distance", 0.0)
        light_state = data.get("light_state", "unknown")
        stop_dur = data.get("stop_duration", 0.0)

        # Log transitions: started stopping or resumed driving
        if should_stop and not self._last_traffic_should_stop:
            if ctrl_type == "stop_sign":
                self._log_behavior("STOP_SIGN_STOP",
                                   f"dist={distance:.2f}m wait={stop_dur:.1f}s")
            elif ctrl_type == "traffic_light":
                self._log_behavior("TRAFFIC_LIGHT_RED",
                                   f"dist={distance:.2f}m light={light_state}")
            elif ctrl_type == "yield_sign":
                self._log_behavior("YIELD_SIGN_STOP",
                                   f"dist={distance:.2f}m")
            else:
                self._log_behavior("TRAFFIC_STOP",
                                   f"type={ctrl_type} dist={distance:.2f}m")
        elif not should_stop and self._last_traffic_should_stop:
            self._log_behavior("TRAFFIC_CLEARED",
                               f"was={self._last_traffic_type}")
        self._last_traffic_should_stop = should_stop
        self._last_traffic_type = ctrl_type

    def _pause_for_obstacle(self):
        """Pause navigation due to obstacle detection."""
        if self.state != MissionState.GO_LEG:
            return

        # Cooldown: skip pause if we just resumed (prevents rapid 0.1-0.5s cycling)
        if time.time() - self._last_resume_time < self._resume_cooldown_s:
            return

        self.get_logger().info("Pausing navigation for obstacle/sign")
        self._log_behavior("PAUSED_OBSTACLE", "navigation paused for obstacle/sign")
        self._navigation_paused = True
        self._obstacle_pause_start = time.time()
        self.state = MissionState.PAUSED_OBSTACLE
        self._publish_status("PAUSED_OBSTACLE")
        self._set_led(LED_OBSTACLE)  # Red for obstacle

        # In MPCC mode, don't cancel the goal — the path is already computed
        # and MPCC handles stopping via motion_enable. Cancelling would force
        # a full path recompute on resume, resetting MPCC progress.
        if not self._mpcc_mode and self.goal_handle is not None:
            self.get_logger().info("Cancelling current navigation goal")
            self.goal_handle.cancel_goal_async()

    def _resume_from_pause(self):
        """Resume navigation after obstacle clears."""
        self.get_logger().info("Resuming navigation after obstacle cleared")
        self._log_behavior("RESUMED_AFTER_OBSTACLE", "navigation resumed")
        self._navigation_paused = False
        self._obstacle_pause_start = None
        self._last_resume_time = time.time()
        self._set_led(LED_DRIVING)  # Green while driving

        if self._mpcc_mode and self._current_path is not None:
            # In MPCC mode with an existing path: republish the same path
            # instead of re-requesting from Nav2. This avoids resetting MPCC's
            # progress tracking and prevents the cancel/recompute/reset loop.
            self.get_logger().info("MPCC mode: republishing existing path (no recompute)")
            self._log_behavior("PATH_REPUBLISHED", "reusing existing path after obstacle")
            self.state = MissionState.GO_LEG
            self._path_pub.publish(self._current_path)
        elif self._mpcc_mode and self._current_path is None and self._goal_result_received and self._last_goal_status != GoalStatus.STATUS_SUCCEEDED:
            # MPCC mode, no valid path, and last goal failed — don't resend (it will just fail again)
            # Transition to recovery instead
            self.get_logger().warn("MPCC mode: no valid path and last goal failed, entering recovery")
            self._log_behavior("RESUME_TO_RECOVERY", "no valid path after obstacle clear")
            self.state = MissionState.RECOVERING
            self._set_led(LED_RECOVERY)
        else:
            # Normal Nav2 mode or no existing path: resend goal
            self.state = MissionState.GO_LEG
            target, label, _, _ = self.legs[self.leg_index]
            self.send_goal(target, label)

    # -------------------------------------------------------------------------
    # Status & LED
    # -------------------------------------------------------------------------

    def _publish_status(self, state: str):
        msg = String()
        msg.data = state
        self._status_pub.publish(msg)

    def _publish_hold(self, hold: bool):
        """Publish hold signal to MPCC controller. When True, MPCC stops sending commands."""
        if self._mpcc_mode and hasattr(self, '_hold_pub'):
            msg = Bool()
            msg.data = hold
            self._hold_pub.publish(msg)
            self.get_logger().debug("Hold signal: %s" % hold)

    def _set_led(self, color_id: int):
        """Set LED color via qcar2_hardware parameter service."""
        if not self._enable_led or self._led_client is None:
            return
        if not self._led_client.service_is_ready():
            self.get_logger().debug("LED service not ready, skipping")
            return
        param = Parameter()
        param.name = "led_color_id"
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_INTEGER
        param.value.integer_value = color_id
        req = SetParameters.Request()
        req.parameters = [param]
        self._led_client.call_async(req)
        self.get_logger().debug("LED set to color_id=%d" % color_id)

    # -------------------------------------------------------------------------
    # TF Utilities
    # -------------------------------------------------------------------------

    def _capture_hub_from_tf(self) -> bool:
        """Set self.hub from map->base_link if available. Returns True if set."""
        try:
            when = rclpy.time.Time()
            t = self._tf_buffer.lookup_transform(
                "map", "base_link", when, rclpy.duration.Duration(seconds=0.5)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            self.hub = (x, y, yaw)
            self.get_logger().info("Hub captured from TF (map->base_link): (%.3f, %.3f, %.3f)" % self.hub)
            self._log_behavior("HUB_CAPTURED_TF", f"map=({x:.3f}, {y:.3f}, {yaw:.3f})")
            self.legs[-1] = (self.hub, "hub", 0.0, False)
            return True
        except TransformException as e:
            self.get_logger().warn("TF map->base_link not available: %s" % e)
            return False

    def _get_current_pose(self):
        """Get current robot pose from TF. Returns (x, y, yaw) or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            return (x, y, yaw)
        except TransformException:
            return None

    # -------------------------------------------------------------------------
    # Navigation Goal Management
    # -------------------------------------------------------------------------

    def make_goal(self, target):
        x, y, yaw = target
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        _, _, qz, qw = yaw_to_quat(float(yaw))
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        if self._mpcc_mode:
            goal = ComputePathToPose.Goal()
            goal.goal = ps
            goal.use_start = False  # Use robot's current pose as start
        else:
            goal = NavigateToPose.Goal()
            goal.pose = ps
        return goal

    def send_goal(self, target, label):
        self.goal_sent_time = time.time()
        self._goal_result_received = False
        self._last_goal_status = None
        self._current_target = target
        self.get_logger().info("Sending goal (%s): (%.3f, %.3f, %.3f)" % (label, target[0], target[1], target[2]))
        self._log_behavior("GOAL_SENT", f"{label} -> map=({target[0]:.3f}, {target[1]:.3f})")
        self._publish_status("GO_LEG_%s" % label)
        self._set_led(LED_DRIVING)  # Green while driving

        if self._mpcc_mode and self._road_graph is not None:
            # --- Road-graph path generation (bypasses Nav2 NavFn) ---
            return self._send_road_graph_path(target, label)
        else:
            # --- Nav2 path planning (original) ---
            return self._send_nav2_goal(target, label)

    def _send_road_graph_path(self, target, label):
        """Generate path from road graph and publish to /plan."""
        # Get current position in QLabs frame for path stitching
        current_pose_map = self._get_current_pose()
        if current_pose_map is not None:
            cur_qlabs = map_to_qlabs_frame(*current_pose_map, **self._transform_params)
            current_pos_qlabs = cur_qlabs[:2]
        else:
            # No TF yet, use hub as default start
            current_pos_qlabs = (self._transform_params['origin_x'],
                                 self._transform_params['origin_y'])

        # Use explicit route name from the leg index (deterministic selection)
        route_name = None
        if hasattr(self, '_leg_route_names') and self.leg_index < len(self._leg_route_names):
            route_name = self._leg_route_names[self.leg_index]

        if route_name is not None:
            qlabs_waypoints = self._road_graph.plan_path_for_mission_leg(
                route_name, current_pos_qlabs)
            self.get_logger().info(
                "Road graph using route '%s' for leg %d (%s)" %
                (route_name, self.leg_index, label))
        else:
            # Fallback: use goal-based selection
            target_qlabs = map_to_qlabs_frame(*target, **self._transform_params)
            qlabs_waypoints = self._road_graph.plan_path_from_pose(
                current_pos_qlabs, target_qlabs[:2])

        if qlabs_waypoints is None or len(qlabs_waypoints) < 2:
            self.get_logger().warn("Road graph returned no path for %s, falling back to Nav2" % label)
            self._log_behavior("ROAD_GRAPH_FALLBACK", f"{label}: no road graph path, using Nav2")
            return self._send_nav2_goal(target, label)

        # Transform to map frame
        map_waypoints = qlabs_path_to_map_path(
            qlabs_waypoints, **self._transform_params)

        # Build nav_msgs/Path message
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(map_waypoints)):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(map_waypoints[i, 0])
            ps.pose.position.y = float(map_waypoints[i, 1])
            if i < len(map_waypoints) - 1:
                dx = map_waypoints[i + 1, 0] - map_waypoints[i, 0]
                dy = map_waypoints[i + 1, 1] - map_waypoints[i, 1]
                yaw = math.atan2(dy, dx)
            else:
                yaw = target[2]
            _, _, qz, qw = yaw_to_quat(yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)

        # Publish path
        self._current_path = path_msg
        self._path_pub.publish(path_msg)
        self._goal_result_received = True
        self._last_goal_status = GoalStatus.STATUS_SUCCEEDED

        path_length = 0.0
        if len(map_waypoints) > 1:
            diffs = np.diff(map_waypoints, axis=0)
            path_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

        self.get_logger().info(
            "Road graph path: %d poses, %.2fm, route=%s, qlabs=(%.2f,%.2f)"
            % (len(path_msg.poses), path_length,
               route_name or 'auto',
               current_pos_qlabs[0], current_pos_qlabs[1]))
        self._log_behavior("ROAD_GRAPH_PATH",
                           f"{label}: {len(path_msg.poses)} poses, {path_length:.2f}m, route={route_name}")
        return True

    def _send_nav2_goal(self, target, label):
        """Original Nav2-based path planning (fallback)."""
        action_name = "compute_path_to_pose" if self._mpcc_mode else "navigate_to_pose"

        if not self.nav_client.server_is_ready():
            self.get_logger().info("Waiting for %s action server..." % action_name)
            if not self.nav_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("%s action server not available after 10s!" % action_name)
                return False
            self.get_logger().info("%s action server is ready" % action_name)

        goal_msg = self.make_goal(target)
        fut = self.nav_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._goal_response_cb)
        return True

    def _goal_response_cb(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error("Goal rejected by Nav2")
            self._log_behavior("GOAL_REJECTED", "Nav2 rejected goal")
            self.goal_handle = None
            self._goal_result_received = True
            self._last_goal_status = GoalStatus.STATUS_ABORTED
            return

        self.get_logger().info("Goal accepted by Nav2")
        self._log_behavior("GOAL_ACCEPTED", "Nav2 accepted goal")
        rfut = self.goal_handle.get_result_async()
        rfut.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result()
        self._last_goal_status = result.status
        self._goal_result_received = True
        self._last_goal_result_time = time.time()
        self.get_logger().info("Goal finished with status=%d" % result.status)
        self._log_behavior("GOAL_RESULT", f"status={result.status}")

        self.goal_handle = None

        if self._mpcc_mode:
            # MPCC mode: we got a path, publish it for MPCC controller
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                path = result.result.path
                if len(path.poses) > 0:
                    self.get_logger().info("Path computed with %d poses, publishing to /plan" % len(path.poses))
                    self._current_path = path  # Store for republishing
                    self._path_pub.publish(path)
                    # Don't call _on_goal_success yet - wait for MPCC to reach the goal
                    # The tick() function will check progress via TF
                else:
                    self.get_logger().warn("Empty path received!")
                    self._current_path = None
                    self._on_goal_failure()
            else:
                self._current_path = None
                self._on_goal_failure()
        else:
            # Normal mode: Nav2 handled the full navigation
            self.goal_sent_time = None
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self._on_goal_success()
            elif result.status == GoalStatus.STATUS_CANCELED:
                # Cancelled (e.g., due to obstacle pause) - don't treat as failure
                self.get_logger().info("Goal was cancelled")
            else:
                self._on_goal_failure()

    def _on_goal_success(self):
        """Handle successful goal completion."""
        target, label, dwell_s, _ = self.legs[self.leg_index]

        # In MPCC mode, tell controller to hold (stop sending commands)
        if self._mpcc_mode:
            self._publish_hold(True)
            # Also publish zero velocity to ensure the car stops
            stop_cmd = Twist()
            self._cmd_vel_pub.publish(stop_cmd)

        if dwell_s > 0:
            if "pickup" in label:
                self._set_led(LED_PICKUP)  # Blue at pickup (per scenario)
                self.get_logger().info("Arrived at PICKUP - dwelling for %.1fs" % dwell_s)
                self._log_behavior("PICKUP_ARRIVED", f"dwelling {dwell_s:.1f}s at ({target[0]:.3f}, {target[1]:.3f})")
            elif "dropoff" in label:
                self._set_led(LED_DROPOFF)  # Yellow/Orange at dropoff (per scenario)
                self.get_logger().info("Arrived at DROPOFF - dwelling for %.1fs" % dwell_s)
                self._log_behavior("DROPOFF_ARRIVED", f"dwelling {dwell_s:.1f}s at ({target[0]:.3f}, {target[1]:.3f})")
            elif "hub" in label:
                self._set_led(LED_HUB)  # Magenta at hub (per scenario)
                self._log_behavior("HUB_ARRIVED", f"at ({target[0]:.3f}, {target[1]:.3f})")
            else:
                self._set_led(LED_DRIVING)  # Green for waypoints
                self._log_behavior("WAYPOINT_ARRIVED", f"{label} dwelling {dwell_s:.1f}s")

            self.state = MissionState.DWELL
            self.stop_until = time.time() + dwell_s
            self._publish_status("DWELL_%s" % label)
            self.retry_count = 0
            self._recovery_strategy_index = 0
            return

        # In MPCC mode, add a brief settling dwell (0.5s) even for drive-through
        # waypoints. This lets the car decelerate before starting the next path,
        # preventing the momentum-induced oscillation seen in logs.
        if self._mpcc_mode:
            self._log_behavior("WAYPOINT_SETTLING", f"{label} 0.5s settle before next leg")
            self.state = MissionState.DWELL
            self.stop_until = time.time() + 0.5
            self.retry_count = 0
            self._recovery_strategy_index = 0
            return

        self._advance_to_next_leg()

    def _on_goal_failure(self):
        """Handle goal failure - enter recovery."""
        self.get_logger().warn("Goal failed - entering recovery mode")
        self._log_behavior("GOAL_FAILED", "entering recovery")
        self.state = MissionState.RECOVERING
        self._set_led(LED_RECOVERY)  # Cyan for recovery
        self._publish_status("RECOVERING")

    def _advance_to_next_leg(self):
        """Move to the next leg of the mission."""
        self.leg_index += 1
        self.retry_count = 0
        self._recovery_strategy_index = 0

        # Release MPCC hold before sending next goal
        if self._mpcc_mode:
            self._publish_hold(False)

        if self.leg_index >= len(self.legs):
            self.state = MissionState.DONE
            self._set_led(LED_HUB)  # Magenta at hub when done (per scenario)
            self._publish_status("DONE")
            self.get_logger().info("Mission COMPLETE!")
            self._log_behavior("MISSION_COMPLETE", "all legs finished")
            return

        target, label, _, _ = self.legs[self.leg_index]
        self.state = MissionState.GO_LEG
        self.send_goal(target, label)

    # -------------------------------------------------------------------------
    # Recovery Logic
    # -------------------------------------------------------------------------

    def _get_recovery_strategies(self) -> list:
        """Get ordered list of recovery strategies for current leg."""
        _, label, _, is_skippable = self.legs[self.leg_index]
        strategies = [
            RecoveryStrategy.RETRY_SAME,
            RecoveryStrategy.CLEAR_COSTMAP,
            RecoveryStrategy.BACKUP_AND_RETRY,
        ]
        if is_skippable:
            strategies.append(RecoveryStrategy.SKIP_WAYPOINT)
        strategies.append(RecoveryStrategy.RESTART_MISSION)
        return strategies

    def _execute_recovery(self):
        """Execute the current recovery strategy."""
        strategies = self._get_recovery_strategies()

        if self._recovery_strategy_index >= len(strategies):
            # All strategies exhausted
            self.get_logger().error("All recovery strategies exhausted - ABORTING mission")
            self._log_behavior("MISSION_ABORTED", "all recovery strategies exhausted")
            self.state = MissionState.ABORTED
            self._set_led(LED_RED)  # Red for abort/error
            self._publish_status("ABORTED")
            return

        strategy = strategies[self._recovery_strategy_index]
        self.retry_count += 1
        target, label, _, _ = self.legs[self.leg_index]

        self.get_logger().info(
            "Recovery attempt %d using strategy: %s" % (self.retry_count, strategy.name)
        )
        self._log_behavior("RECOVERY_ATTEMPT", f"#{self.retry_count} strategy={strategy.name} leg={label}")

        if strategy == RecoveryStrategy.RETRY_SAME:
            self._recovery_retry_same(target, label)

        elif strategy == RecoveryStrategy.BACKUP_AND_RETRY:
            self._recovery_backup_and_retry(target, label)

        elif strategy == RecoveryStrategy.CLEAR_COSTMAP:
            self._recovery_clear_costmap(target, label)

        elif strategy == RecoveryStrategy.SKIP_WAYPOINT:
            self._recovery_skip_waypoint()

        elif strategy == RecoveryStrategy.RESTART_MISSION:
            self._recovery_restart_mission()

        # Move to next strategy for next attempt
        if self.retry_count >= self._max_retries:
            self._recovery_strategy_index += 1
            self.retry_count = 0

    def _recovery_retry_same(self, target, label):
        """Simply retry the same goal. If last goal failed very fast, skip to SKIP_WAYPOINT for skippable waypoints."""
        _, _, _, is_skippable = self.legs[self.leg_index]

        # Fast-fail detection: if last goal failed within 2s, skip directly for skippable waypoints
        if (is_skippable and self._last_goal_result_time is not None
                and (time.time() - self._last_goal_result_time) < 2.0
                and self._last_goal_status == GoalStatus.STATUS_ABORTED):
            self.get_logger().warn("Fast-fail detected for skippable waypoint %s — skipping" % label)
            self._log_behavior("FAST_FAIL_SKIP", f"skipping {label} (failed within 2s)")
            self._recovery_skip_waypoint()
            return

        self.get_logger().info("Retrying same goal: %s" % label)
        self.state = MissionState.GO_LEG
        self.send_goal(target, label + "_retry")

    def _recovery_backup_and_retry(self, target, label):
        """Back up slightly, then retry."""
        self.get_logger().info("Backing up %.2fm before retry" % self._backup_distance_m)

        # Publish backup velocity
        twist = Twist()
        twist.linear.x = -self._backup_speed
        self._cmd_vel_pub.publish(twist)
        self._backup_start_time = time.time()

        # Will complete in tick() and then retry
        self.state = MissionState.RECOVERING

    def _recovery_clear_costmap(self, target, label):
        """Clear costmaps and retry (if service available)."""
        self.get_logger().info("Attempting to clear costmaps")

        # Try to call clear costmap services
        try:
            from nav2_msgs.srv import ClearEntireCostmap
            clear_global = self.create_client(ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap")
            clear_local = self.create_client(ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap")

            if clear_global.service_is_ready():
                clear_global.call_async(ClearEntireCostmap.Request())
                self.get_logger().info("Cleared global costmap")

            if clear_local.service_is_ready():
                clear_local.call_async(ClearEntireCostmap.Request())
                self.get_logger().info("Cleared local costmap")
        except Exception as e:
            self.get_logger().warn("Could not clear costmaps: %s" % e)

        # Wait a moment then retry
        time.sleep(0.5)
        self.state = MissionState.GO_LEG
        self.send_goal(target, label + "_cleared")

    def _recovery_skip_waypoint(self):
        """Skip current waypoint and move to next leg."""
        _, label, _, _ = self.legs[self.leg_index]
        self.get_logger().warn("Skipping waypoint: %s" % label)
        self._advance_to_next_leg()

    def _recovery_restart_mission(self):
        """Restart the entire mission from pickup."""
        self.get_logger().warn("Restarting mission from leg 0 (pickup)")
        self.leg_index = 0
        self.retry_count = 0
        self._recovery_strategy_index = 0

        target, label, _, _ = self.legs[0]
        self.state = MissionState.GO_LEG
        self.send_goal(target, label + "_restart")

    # -------------------------------------------------------------------------
    # Main Tick
    # -------------------------------------------------------------------------

    def tick(self):
        """Main state machine tick (runs at 10Hz)."""

        # ---- WAIT_FOR_NAV ----
        if self.state == MissionState.WAIT_FOR_NAV:
            if self.nav_client.server_is_ready():
                if self._use_tf_hub:
                    self.state = MissionState.CAPTURING_HUB
                    self._hub_capture_start = time.time()
                else:
                    self._start_mission()
            return

        # ---- CAPTURING_HUB ----
        if self.state == MissionState.CAPTURING_HUB:
            if self._capture_hub_from_tf():
                self._start_mission()
            elif (time.time() - self._hub_capture_start) > self._hub_tf_timeout_s:
                self.get_logger().warn("Hub TF timeout - using config hub")
                self._start_mission()
            return

        # ---- GO_LEG ----
        if self.state == MissionState.GO_LEG:
            # Check for obstacle pause — only when we have a valid path and the goal
            # has produced a result. When the goal hasn't even produced a path yet,
            # obstacles are irrelevant (the vehicle isn't moving).
            if self._enable_obstacle_detection and not self._motion_enabled:
                has_valid_path = (not self._mpcc_mode) or (self._current_path is not None)
                if has_valid_path:
                    self._pause_for_obstacle()
                    return

            # MPCC mode: periodically republish path to ensure MPCC controller receives it
            if self._mpcc_mode and self._current_path is not None:
                # Republish path every 2 seconds for robustness
                current_time = time.time()
                if not hasattr(self, '_last_path_pub_time') or (current_time - self._last_path_pub_time) > 2.0:
                    self._path_pub.publish(self._current_path)
                    self._last_path_pub_time = current_time

            # MPCC mode: check if we've reached the goal via TF
            if self._mpcc_mode and self._current_target is not None and self._goal_result_received:
                current_pose = self._get_current_pose()
                if current_pose is not None:
                    target_x, target_y, _ = self._current_target
                    cur_x, cur_y, cur_yaw = current_pose
                    dist = math.sqrt((target_x - cur_x)**2 + (target_y - cur_y)**2)

                    # Log position periodically (every 2 seconds) for debugging
                    current_time = time.time()
                    if not hasattr(self, '_last_pos_log_time') or (current_time - self._last_pos_log_time) > 2.0:
                        # Convert current position back to QLabs for reference
                        if self._use_qlabs_coords:
                            qlabs_pos = map_to_qlabs_frame(cur_x, cur_y, cur_yaw, **self._transform_params)
                            self.get_logger().info(
                                f"MPCC tracking: map=({cur_x:.2f}, {cur_y:.2f}) "
                                f"qlabs=({qlabs_pos[0]:.2f}, {qlabs_pos[1]:.2f}) "
                                f"target_map=({target_x:.2f}, {target_y:.2f}) dist={dist:.2f}m"
                            )
                        else:
                            self.get_logger().info(
                                f"MPCC tracking: pos=({cur_x:.2f}, {cur_y:.2f}) "
                                f"target=({target_x:.2f}, {target_y:.2f}) dist={dist:.2f}m"
                            )
                        self._last_pos_log_time = current_time

                    if dist < self._goal_tol_m:
                        self.get_logger().info("MPCC reached goal (dist=%.3f < tol=%.3f)" % (dist, self._goal_tol_m))
                        self._log_behavior("MPCC_GOAL_REACHED", f"dist={dist:.3f} < tol={self._goal_tol_m:.3f}")
                        self.goal_sent_time = None
                        self._on_goal_success()
                        return

            # Check for goal timeout
            if (
                self.goal_sent_time is not None
                and (time.time() - self.goal_sent_time) > self._goal_timeout_s
            ):
                self.get_logger().warn("Goal timeout (%.0fs) - entering recovery" % self._goal_timeout_s)
                self._log_behavior("GOAL_TIMEOUT", f"after {self._goal_timeout_s:.0f}s")
                if self.goal_handle is not None:
                    try:
                        self.goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                self.goal_handle = None
                self.goal_sent_time = None
                self._on_goal_failure()
            return

        # ---- PAUSED_OBSTACLE ----
        if self.state == MissionState.PAUSED_OBSTACLE:
            if self._motion_enabled:
                self._resume_from_pause()
            elif (time.time() - self._obstacle_pause_start) > self._obstacle_pause_timeout_s:
                self.get_logger().warn("Obstacle pause timeout - entering recovery")
                self._obstacle_pause_start = None
                self.state = MissionState.RECOVERING
            return

        # ---- DWELL ----
        if self.state == MissionState.DWELL:
            # In MPCC mode, keep publishing zero velocity to prevent drift
            if self._mpcc_mode:
                stop_cmd = Twist()
                self._cmd_vel_pub.publish(stop_cmd)
            if time.time() >= self.stop_until:
                label = self.legs[self.leg_index][1] if self.leg_index < len(self.legs) else "?"
                self._log_behavior("DWELL_COMPLETE", f"{label} - advancing to next leg")
                self._advance_to_next_leg()
            return

        # ---- RECOVERING ----
        if self.state == MissionState.RECOVERING:
            # Handle backup completion
            if self._backup_start_time is not None:
                backup_duration = self._backup_distance_m / self._backup_speed
                if (time.time() - self._backup_start_time) >= backup_duration:
                    # Stop backing up
                    twist = Twist()
                    self._cmd_vel_pub.publish(twist)
                    self._backup_start_time = None

                    # Now retry
                    target, label, _, _ = self.legs[self.leg_index]
                    self.state = MissionState.GO_LEG
                    self.send_goal(target, label + "_after_backup")
                return

            # Execute next recovery strategy
            self._execute_recovery()
            return

        # ---- DONE / ABORTED ----
        # Nothing to do

    def _start_mission(self):
        """Start the mission after initialization."""
        self.get_logger().info("Starting mission with %d legs" % len(self.legs))
        self._log_behavior("MISSION_START", f"{len(self.legs)} legs")
        # Per scenario: Start at Taxi Hub with Magenta LEDs
        self._set_led(LED_HUB)  # Magenta at start
        self.leg_index = 0
        target, label, _, _ = self.legs[0]
        self.state = MissionState.GO_LEG
        self.send_goal(target, label)


def main():
    rclpy.init()
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
