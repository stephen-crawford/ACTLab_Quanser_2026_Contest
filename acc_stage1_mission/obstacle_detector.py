#!/usr/bin/env python3
from __future__ import annotations

"""
Enhanced Obstacle Detector for ACC Competition

Detects:
- Pedestrians (YOLO class: person)
- Traffic cones (YOLO class: traffic cone / orange objects)
- Stop signs, yield signs
- Traffic lights (red/green state)
- Other vehicles

Publishes:
- /motion_enable (Bool): False when obstacle detected, True when clear
- /obstacle_info (String): JSON with detection details
- /detection_image (Image): Annotated camera image

Configuration:
- Edit DETECTION_CONFIG below to tune detection parameters
- Supports both YOLO (GPU) and fallback color detection (CPU)
"""

import os
import time
import math
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

# Import traffic control state message classes
from acc_stage1_mission.traffic_control_state import (
    TrafficControlState,
    ObstaclePosition,
    ObstaclePositions,
)
from acc_stage1_mission.road_boundaries import RoadBoundarySpline
from acc_stage1_mission.pedestrian_tracker import PedestrianKalmanTracker
from acc_stage1_mission.detection_interface import DetectorBackend, Detection
from acc_stage1_mission.module_config import load_module_config

# Camera horizontal FOV for QCar2 (~60 degrees)
CAMERA_HFOV_RAD = math.radians(60.0)

# =============================================================================
# DETECTION CONFIGURATION - EDIT THESE TO TUNE BEHAVIOR
# Thresholds matched to reference repo (MPC_node.py) for QLabs environment.
# =============================================================================
DETECTION_CONFIG = {
    # Reference repo approach: confidence IS the distance proxy.
    # Higher confidence = closer/larger object. No distance thresholds for stop decisions.

    # Confidence thresholds - tuned for COCO YOLOv8n on QLabs-rendered objects.
    # COCO detections on QLabs signs/lights typically reach ~0.3-0.6.
    # If the custom QLabs-trained model (best.pt) is found, these are overridden
    # with higher thresholds in _init_yolo().
    'pedestrian_confidence': 0.45,
    'cone_confidence': 0.35,
    'sign_confidence': 0.50,
    'red_light_confidence': 0.50,
    'green_light_confidence': 0.50,
    'round_sign_confidence': 0.50,
    'vehicle_confidence': 0.5,

    # Timing (seconds) - tuned for QLabs competition
    'stop_sign_pause': 3.0,        # Stop duration at stop sign (competition requires full stop)
    'stop_sign_cooldown': 15.0,    # Per-sign cooldown (reference: 15s)
    'traffic_light_cooldown': 6.0, # After green: ignore reds for 6s
    'detection_cooldown': 15.0,    # General cooldown for same detection class

    # YOLO settings
    'yolo_classes': [0, 2, 9, 11, 33],  # COCO: person, car, traffic light, stop sign, sports ball
    'image_width': 640,
    'image_height': 480,

    # Custom YOLO model path (QLabs-trained, 9 classes)
    # Will be searched in order; first found wins
    'custom_model_paths': [
        # Package-relative paths (set at runtime in _init_yolo)
        # Source tree path
        # Existing reference paths (backward compat)
        '/workspaces/isaac_ros-dev/ros2/src/polyctrl/polyctrl/best.pt',
        '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/models/best.pt',
    ],

    # Fallback color detection (when YOLO unavailable)
    'use_color_fallback': True,
    'cone_hsv_lower': [5, 150, 150],
    'cone_hsv_upper': [15, 255, 255],

}

# COCO YOLO class name mapping - only competition-relevant classes
YOLO_CLASSES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign',
    33: 'sports ball',  # Traffic cone proxy in QLabs
}

# Custom QLabs-trained YOLO model class mapping (8 classes)
# Matches ACC2025_Quanser_Student_Competition/polyctrl/polyctrl/data.yaml
CUSTOM_YOLO_CLASSES = {
    0: 'cone',
    1: 'green',     # Green traffic light
    2: 'person',    # Pedestrian
    3: 'red',       # Red traffic light
    4: 'round',     # Roundabout sign
    5: 'stop',      # Stop sign
    6: 'yellow',    # Yellow traffic light
    7: 'yield',     # Yield sign
}

# Whitelist of allowed class names - detections not in this set are discarded
ALLOWED_CLASSES = {
    'person', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'sports ball',
    'traffic_cone', 'yield sign',                    # From color detection
    'traffic_light_red', 'traffic_light_green',      # From color detection
    'traffic_light_yellow',                          # From custom model
    # Custom model classes
    'cone', 'green', 'red', 'yellow', 'stop', 'yield', 'round',
}


# =============================================================================
# DETECTION BACKENDS — Wrap existing detection methods as DetectorBackend
# implementations. Each returns list of dicts (legacy format); the
# ObstacleDetector node converts to Detection objects as needed.
# The backends are instantiated inside the node to access YOLO models,
# config, and depth images.
# =============================================================================

class _HSVBackend(DetectorBackend):
    """HSV+Contour color detection backend (CPU-friendly).

    Wraps ObstacleDetector._color_detect(). Does not require YOLO/GPU.
    Benchmark: F1=0.429, 1.82ms average latency.
    """
    name = "hsv"

    def __init__(self, node: 'ObstacleDetector'):
        self._node = node

    def detect(self, image: np.ndarray) -> list:
        return self._node._color_detect()


class _YOLOBackend(DetectorBackend):
    """YOLO detection backend (custom or COCO model).

    Wraps ObstacleDetector._yolo_detect(). Requires ultralytics.
    """
    name = "yolo"

    def __init__(self, node: 'ObstacleDetector'):
        self._node = node

    def detect(self, image: np.ndarray) -> list:
        return self._node._yolo_detect()


class _HybridBackend(DetectorBackend):
    """Hybrid HSV pre-filter + YOLO verification backend.

    Wraps ObstacleDetector._hybrid_detect(). Falls back to HSV-only
    when YOLO is unavailable.
    """
    name = "hybrid"

    def __init__(self, node: 'ObstacleDetector'):
        self._node = node

    def detect(self, image: np.ndarray) -> list:
        return self._node._hybrid_detect()


class _AutoBackend(DetectorBackend):
    """Auto-selection backend: custom YOLO -> COCO YOLO -> HSV fallback.

    Wraps the original _dispatch_detect() 'auto' logic.
    """
    name = "auto"

    def __init__(self, node: 'ObstacleDetector'):
        self._node = node

    def detect(self, image: np.ndarray) -> list:
        if self._node._yolo is not None:
            return self._node._yolo_detect()
        elif DETECTION_CONFIG['use_color_fallback']:
            return self._node._color_detect()
        return []


class ObstacleDetector(Node):
    """
    ROS2 node for obstacle detection using YOLO and/or color detection.

    Detection backends are selected via the detection_mode ROS parameter
    or config/modules.yaml. All backends implement the DetectorBackend ABC
    from detection_interface.py.

    Available backends:
      - 'auto' (default): custom model -> COCO YOLO -> HSV fallback
      - 'hsv':       HSV+Contour color detection (F1=0.429, 1.82ms)
      - 'yolo_coco': Force COCO YOLOv8n
      - 'custom':    Force custom QLabs-trained model
      - 'hybrid':    HSV pre-filter -> YOLO verification on crops
      - 'hough_hsv': HoughCircles for lights + HSV for signs/cones (F1=0.444, 3.65ms)

    To switch backend:
      ROS param:   detection_mode:=hsv
      Launch arg:  ros2 launch ... detection_mode:=hough_hsv
      Config file: config/modules.yaml -> detection.backend
    """

    def __init__(self):
        super().__init__('obstacle_detector')

        # Declare ROS parameters (can be overridden at launch)
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('camera_topic', '/camera/color_image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('detection_mode', 'auto')  # auto|yolo_coco|custom|hsv|hybrid

        self._use_yolo = self.get_parameter('use_yolo').value
        self._debug_viz = self.get_parameter('debug_visualization').value
        self._camera_topic = self.get_parameter('camera_topic').value
        self._depth_topic = self.get_parameter('depth_topic').value
        self._detection_mode = self.get_parameter('detection_mode').value

        # Resolve detection_mode from modules.yaml when param is 'auto'
        if self._detection_mode == 'auto':
            try:
                mod_config = load_module_config()
                yaml_backend = mod_config['detection']['backend']
                if yaml_backend != 'auto':
                    self._detection_mode = yaml_backend
                    self.get_logger().info(
                        f"Detection mode from modules.yaml: {yaml_backend}")
            except Exception:
                pass  # Keep 'auto' default

        # Initialize YOLO if available
        self._yolo = None
        self._yolo_coco = None  # Separate COCO model for hybrid mode
        self._depth_aligner = None
        if self._use_yolo:
            self._init_yolo()

        # Set up active detection backend
        self._active_backend = self._create_backend(self._detection_mode)
        self.get_logger().info(
            f"Active detection backend: {self._active_backend.name}")

        # State tracking
        self._motion_enabled = True
        self._last_detection_time = {}  # Track cooldowns per object type
        self._current_detections = []
        self._pause_until = 0.0

        # Clear-hysteresis: require N consecutive clear frames before re-enabling motion
        # This prevents rapid detect/clear cycling from depth sensor noise
        self._consecutive_clear_frames = 0
        self._clear_hysteresis_frames = 5  # ~167ms at 30Hz (was 10 - too slow to resume)

        # Detection persistence: require N consecutive detection frames before triggering stop
        # This prevents single-frame false positives from triggering stops
        self._consecutive_detection_frames = 0
        self._detection_persistence_threshold = 2  # ~67ms at 30Hz (was 5 - too slow to react)

        # Traffic control state tracking
        self._at_stop_sign = False
        self._stop_sign_start_time = None
        self._stop_sign_wait_complete = False
        self._stop_sign_position = None  # (x, y) in image coordinates
        self._last_traffic_light_state = "unknown"
        self._traffic_light_transition_time = None

        # Traffic light state machine (reference repo approach)
        # States: idle -> cross_waiting (stopped at red) -> cross_cooldown (after green)
        self._cross_waiting = False      # Waiting at red light for green
        self._cross_waiting_start = 0.0  # When cross_waiting started
        self._cross_waiting_timeout = 15.0  # Max time to wait if light disappears from view
        self._cross_waiting_no_red_frames = 0  # Frames since last red detection while waiting
        self._cross_waiting_no_red_threshold = 60  # ~2s at 30Hz: if no red for 2s, assume passed
        self._cross_cooldown = False     # Cooldown after proceeding on green
        self._cross_cooldown_start = 0.0 # When cooldown started

        # Spatial stop sign tracking: track bbox center Y to detect different signs
        # instead of using a single global cooldown
        self._last_stop_sign_bbox_y = None  # Y center of last detected stop sign
        self._stop_sign_bbox_threshold = 50  # Pixel difference to consider a "new" sign

        # Yield sign post-action suppression: after yielding, suppress re-detection
        # for a duration so the vehicle can drive past the sign without re-triggering
        self._yield_suppress_until = 0.0  # Timestamp until yield sign detection is suppressed
        self._yield_suppress_duration = 8.0  # Seconds to suppress after yielding
        self._yield_active = False  # Currently yielding (waiting for pedestrian to clear)

        self._image_width = DETECTION_CONFIG.get('image_width', 640)

        # TF listener for vehicle pose in map frame (for pedestrian road-boundary checks)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._vehicle_x = 0.0
        self._vehicle_y = 0.0
        self._vehicle_theta = 0.0
        self._has_vehicle_pose = False

        # Road boundaries for map-based pedestrian filtering
        self._road_boundaries = None
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('acc_stage1_mission')
            boundary_path = os.path.join(pkg_share, 'config', 'road_boundaries.yaml')
            if os.path.exists(boundary_path):
                self._road_boundaries = RoadBoundarySpline(boundary_path)
                self.get_logger().info(f"Loaded road boundaries from {boundary_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not load road boundaries: {e}")

        # Kalman filter pedestrian tracker (smooths noisy depth, tracks crossing)
        self._ped_tracker = PedestrianKalmanTracker(self._road_boundaries)

        # Mission phase tracking
        self._mission_phase = "unknown"  # "pickup", "dropoff", "hub", "unknown"

        # CV Bridge for image conversion
        self._bridge = CvBridge()
        self._latest_rgb = None
        self._latest_depth = None

        # Publishers
        self._motion_pub = self.create_publisher(Bool, '/motion_enable', 10)
        self._info_pub = self.create_publisher(String, '/obstacle_info', 10)
        if self._debug_viz:
            self._viz_pub = self.create_publisher(Image, '/detection_image', 10)

        # Traffic control state publisher (JSON-encoded TrafficControlState)
        self._traffic_control_pub = self.create_publisher(
            String, '/traffic_control_state', 10)

        # Obstacle positions publisher for MPCC (JSON-encoded ObstaclePositions)
        self._obstacle_positions_pub = self.create_publisher(
            String, '/obstacle_positions', 10)

        # Subscribers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._rgb_sub = self.create_subscription(
            Image, self._camera_topic, self._rgb_callback, qos)

        if self._depth_topic:
            self._depth_sub = self.create_subscription(
                Image, self._depth_topic, self._depth_callback, qos)

        # Mission status subscriber — tracks which leg we're on
        self._mission_status_sub = self.create_subscription(
            String, 'mission/status', self._mission_status_callback, 10)

        # Main processing timer
        self._timer = self.create_timer(1.0/30.0, self._process_frame)

        # Publish initial state
        self._publish_motion(True)

        self.get_logger().info(
            f"ObstacleDetector initialized - YOLO: {self._yolo is not None}, "
            f"Debug viz: {self._debug_viz}, "
            f"detection_mode: {self._detection_mode}"
        )

    def _init_yolo(self):
        """Initialize YOLO detector if available.

        Tries custom QLabs-trained model first (8 classes), falls back to
        standard COCO YOLOv8 nano.
        """
        self._using_custom_model = False
        try:
            from ultralytics import YOLO
            import os
            from ament_index_python.packages import get_package_share_directory

            # Build search paths for custom model
            model_search_paths = []
            try:
                pkg_share = get_package_share_directory('acc_stage1_mission')
                model_search_paths.append(
                    os.path.join(pkg_share, 'models', 'best.pt'))
            except Exception:
                pass
            # Source tree path (relative to this file)
            model_search_paths.append(
                os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt'))
            # Existing reference paths from config (backward compat)
            model_search_paths.extend(
                DETECTION_CONFIG.get('custom_model_paths', []))

            # Try custom QLabs-trained model first
            custom_model_path = None
            for path in model_search_paths:
                if os.path.isfile(path):
                    custom_model_path = path
                    break

            if custom_model_path:
                self._yolo = YOLO(custom_model_path)
                self._using_custom_model = True
                model_desc = f"custom QLabs ({custom_model_path})"
            else:
                # Fall back to standard COCO model
                model_size = 'yolov8n.pt'
                self._yolo = YOLO(model_size)
                model_desc = f"COCO {model_size}"

            # Try CUDA first (competition uses GPU), fall back to CPU
            self._yolo_device = 'cpu'
            try:
                import torch
                if torch.cuda.is_available():
                    self._yolo_device = 'cuda'
            except ImportError:
                pass

            # Override thresholds for custom model — values from competition repo
            # (ACC2025_Quanser_Student_Competition MPC_node.py), tested with this best.pt
            if self._using_custom_model:
                DETECTION_CONFIG['sign_confidence'] = 0.91
                DETECTION_CONFIG['red_light_confidence'] = 0.83
                DETECTION_CONFIG['green_light_confidence'] = 0.84
                DETECTION_CONFIG['pedestrian_confidence'] = 0.70
                DETECTION_CONFIG['cone_confidence'] = 0.80
                DETECTION_CONFIG['round_sign_confidence'] = 0.90
                self.get_logger().info("Using custom model confidence thresholds (from competition repo)")

            # For hybrid mode, also load COCO model if using custom
            if self._using_custom_model and self._detection_mode == 'hybrid':
                try:
                    self._yolo_coco = YOLO('yolov8n.pt')
                    self.get_logger().info("Hybrid mode: loaded COCO model for verification")
                except Exception:
                    self._yolo_coco = None

            self.get_logger().info(f"YOLO initialized: {model_desc}, device={self._yolo_device}, "
                                   f"detection_mode={self._detection_mode}")

        except ImportError as e:
            self.get_logger().warn(f"YOLO not available: {e}")
            self.get_logger().info("Falling back to color-based detection")
            self._yolo = None

    def _rgb_callback(self, msg: Image):
        """Store latest RGB image."""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            if img is None:
                return
            self._latest_rgb = img
            self._image_width = img.shape[1]
        except Exception as e:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000,
                f"RGB conversion failed: {e}")

    def _depth_callback(self, msg: Image):
        """Store latest depth image."""
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def _mission_status_callback(self, msg: String):
        """Track mission phase from mission_manager status."""
        status = msg.data.lower()
        if 'pickup' in status:
            self._mission_phase = 'pickup'
        elif 'dropoff' in status:
            self._mission_phase = 'dropoff'
        elif 'hub' in status:
            self._mission_phase = 'hub'

    def _update_vehicle_pose(self):
        """Update vehicle pose from TF (map -> base_link)."""
        for source, target in [('map', 'base_link'), ('odom', 'base_link')]:
            try:
                t = self._tf_buffer.lookup_transform(
                    source, target,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.02))
                self._vehicle_x = t.transform.translation.x
                self._vehicle_y = t.transform.translation.y
                q = t.transform.rotation
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self._vehicle_theta = math.atan2(siny, cosy)
                self._has_vehicle_pose = True
                return
            except TransformException:
                continue

    def _det_to_map_position(self, det: dict) -> tuple[float, float] | None:
        """Transform a detection to map (x, y) using depth camera + vehicle TF.

        Returns (x_map, y_map) or None if position can't be determined.
        """
        if not self._has_vehicle_pose:
            return None

        bbox = det.get('bbox')
        distance = det.get('distance')
        if bbox is None or distance is None or distance <= 0:
            return None

        bx, by, bw, bh = bbox
        bbox_cx = bx + bw / 2

        # Use depth image for more accurate distance at bbox center
        if self._latest_depth is not None:
            cx_px = int(bx + bw / 2)
            cy_px = int(by + bh / 2)
            if 0 <= cy_px < self._latest_depth.shape[0] and 0 <= cx_px < self._latest_depth.shape[1]:
                depth_val = float(self._latest_depth[cy_px, cx_px])
                if 0.1 < depth_val < 10.0:
                    distance = depth_val

        img_w = self._image_width if self._image_width > 0 else 640
        normalized = (bbox_cx - img_w / 2) / (img_w / 2)
        lateral = normalized * distance * math.tan(CAMERA_HFOV_RAD / 2)

        cos_t = math.cos(self._vehicle_theta)
        sin_t = math.sin(self._vehicle_theta)
        x_map = self._vehicle_x + distance * cos_t - lateral * sin_t
        y_map = self._vehicle_y + distance * sin_t + lateral * cos_t
        return (x_map, y_map)

    def _update_pedestrian_tracker(self, detections: list):
        """Feed pedestrian detections into the Kalman tracker.

        Computes map positions for all pedestrian detections and updates
        the tracker, which smooths positions and estimates velocity.
        """
        cfg = DETECTION_CONFIG
        measurements = []
        for det in detections:
            if det['class'] != 'person':
                continue
            if det.get('confidence', 0) < cfg['pedestrian_confidence']:
                continue
            pos = self._det_to_map_position(det)
            if pos is not None:
                measurements.append(pos)

        self._ped_tracker.predict()
        self._ped_tracker.update(measurements)

    def _process_frame(self):
        """Main processing loop - runs at 30Hz."""
        current_time = time.time()

        # Update vehicle pose for road boundary checks
        self._update_vehicle_pose()

        # Check if we're in a timed pause
        if current_time < self._pause_until:
            return

        if self._latest_rgb is None:
            return

        # Run detection based on detection_mode
        detections = self._dispatch_detect()

        # Update Kalman tracker with pedestrian detections
        self._update_pedestrian_tracker(detections)

        # Process detections and determine action
        self._current_detections = detections
        should_stop, reason, pause_duration = self._process_detections(detections)

        # Update motion state with detection persistence and clear-hysteresis
        if should_stop:
            self._consecutive_clear_frames = 0  # Reset clear counter
            self._consecutive_detection_frames += 1

            # Only trigger stop after N consecutive detection frames (prevents single-frame false positives)
            if self._consecutive_detection_frames >= self._detection_persistence_threshold:
                if self._motion_enabled:
                    self._motion_enabled = False
                    self._publish_motion(False)
                    self.get_logger().info(f"STOP: {reason} (after {self._consecutive_detection_frames} frames)")

                    if pause_duration > 0:
                        self._pause_until = current_time + pause_duration
                        self.get_logger().info(f"Pausing for {pause_duration:.1f}s")

        elif not should_stop and not self._motion_enabled:
            self._consecutive_detection_frames = 0  # Reset detection counter
            self._consecutive_clear_frames += 1
            if self._consecutive_clear_frames >= self._clear_hysteresis_frames:
                self._motion_enabled = True
                self._publish_motion(True)
                self.get_logger().info(
                    f"CLEAR: Motion enabled (after {self._consecutive_clear_frames} clear frames)"
                )

        elif not should_stop and self._motion_enabled:
            self._consecutive_detection_frames = 0  # Reset detection counter

        # Publish detection info
        self._publish_info(detections, reason if should_stop else "clear")

        # Process and publish traffic control state
        traffic_state = self._process_traffic_control(detections)
        self._publish_traffic_control_state(traffic_state)

        # Publish obstacle positions for MPCC
        self._publish_obstacle_positions(detections)

        # Publish debug visualization
        if self._debug_viz and self._latest_rgb is not None:
            self._publish_visualization(detections)

    def _yolo_detect(self) -> list:
        """Run YOLO detection on current frame.

        Handles both custom QLabs-trained model (8 classes) and standard COCO model.
        Custom model classes are mapped to standard names for downstream processing.
        """
        detections = []

        if self._yolo is None or self._latest_rgb is None:
            return detections

        try:
            rgb = self._latest_rgb
            depth = self._latest_depth
            img_shape = rgb.shape

            # Run YOLO inference (ultralytics) on CPU
            results = self._yolo(rgb, verbose=False, conf=0.3, device=self._yolo_device)

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    # Get class info
                    cls_id = int(boxes.cls[i].item())
                    cls_name = result.names[cls_id]
                    confidence = float(boxes.conf[i].item())

                    # For custom model, map class names to standard names
                    if self._using_custom_model:
                        custom_name = CUSTOM_YOLO_CLASSES.get(cls_id, cls_name)
                        # Map custom classes to standard detection names
                        if custom_name == 'red':
                            cls_name = 'traffic_light_red'
                        elif custom_name == 'green':
                            cls_name = 'traffic_light_green'
                        elif custom_name == 'yellow':
                            cls_name = 'traffic_light_yellow'
                        elif custom_name == 'stop':
                            cls_name = 'stop sign'
                        elif custom_name == 'yield':
                            cls_name = 'yield sign'
                        elif custom_name == 'cone':
                            cls_name = 'traffic_cone'
                        elif custom_name == 'round':
                            cls_name = 'round'
                        elif custom_name == 'person':
                            cls_name = 'person'
                        else:
                            cls_name = custom_name

                    # COCO 'traffic light' — use HSV to determine red/green state
                    if cls_name == 'traffic light' and not self._using_custom_model:
                        x1_, y1_, x2_, y2_ = boxes.xyxy[i].tolist()
                        color_label = self._classify_traffic_light_color(
                            rgb, int(x1_), int(y1_), int(x2_), int(y2_))
                        if color_label is not None:
                            cls_name = color_label  # e.g. 'traffic_light_red'

                    # Whitelist filter: skip classes not in ALLOWED_CLASSES
                    if cls_name not in ALLOWED_CLASSES:
                        continue

                    # Get bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                    # Estimate distance from depth image if available
                    distance = None
                    if depth is not None:
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                            distance = float(depth[cy, cx])
                            if distance <= 0 or distance > 10:
                                distance = None

                    # Estimate distance from bbox size if depth unavailable
                    if distance is None:
                        bbox_area = (x2 - x1) * (y2 - y1)
                        distance = self._estimate_distance_from_size(bbox_area, cls_name)

                    det = {
                        'class': cls_name,
                        'confidence': confidence,
                        'distance': distance,
                        'bbox': bbox,
                    }
                    detections.append(det)

        except Exception as e:
            self.get_logger().error(f"YOLO detection failed: {e}")

        return detections

    def _create_backend(self, mode: str) -> DetectorBackend:
        """Create detection backend from mode string.

        Args:
            mode: One of 'auto', 'hsv', 'yolo_coco', 'custom', 'hybrid', 'hough_hsv'.

        Returns:
            DetectorBackend instance.
        """
        if mode == 'hsv' or mode == 'hough_hsv':
            return _HSVBackend(self)
        elif mode == 'hybrid':
            return _HybridBackend(self)
        elif mode in ('yolo_coco', 'custom'):
            if self._yolo is not None:
                return _YOLOBackend(self)
            # Fallback to HSV if YOLO not available
            return _HSVBackend(self)
        else:
            # 'auto' mode
            return _AutoBackend(self)

    def _dispatch_detect(self) -> list:
        """Dispatch detection to the active backend.

        Returns list of detection dicts compatible with downstream processing.
        """
        return self._active_backend.detect(self._latest_rgb)

    def _hybrid_detect(self) -> list:
        """Hybrid detection: HSV pre-filter finds candidates, YOLO verifies on crops.

        Pipeline:
        1. Run HSV color detection to find candidate regions
        2. For each candidate, extract a padded crop
        3. Run YOLO on each crop to verify/reject
        4. Merge results: verified detections get boosted confidence

        Falls back to HSV-only if YOLO is unavailable.
        """
        # Step 1: HSV pre-filter
        hsv_detections = self._color_detect()

        if self._latest_rgb is None:
            return hsv_detections

        # Use COCO model if available for verification, else primary model
        verify_model = self._yolo_coco if self._yolo_coco is not None else self._yolo
        if verify_model is None:
            return hsv_detections

        h_img, w_img = self._latest_rgb.shape[:2]
        verified = []

        for det in hsv_detections:
            bbox = det.get('bbox')
            if bbox is None:
                verified.append(det)
                continue

            bx, by, bw, bh = bbox
            # Expand crop by 50% for YOLO context
            margin_x = bw // 2
            margin_y = bh // 2
            cx1 = max(0, bx - margin_x)
            cy1 = max(0, by - margin_y)
            cx2 = min(w_img, bx + bw + margin_x)
            cy2 = min(h_img, by + bh + margin_y)

            crop = self._latest_rgb[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                verified.append(det)
                continue

            # Resize small crops to min YOLO input size
            scale = max(1.0, 64.0 / min(crop.shape[:2]))
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)

            try:
                results = verify_model(crop, verbose=False, conf=0.15,
                                       device=self._yolo_device)
                yolo_found = False
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        yolo_found = True
                        break

                if yolo_found:
                    # Both HSV and YOLO agree — boost confidence
                    det['confidence'] = min(1.0, det['confidence'] * 1.3)
                    verified.append(det)
                else:
                    # YOLO disagrees — keep with reduced confidence
                    det['confidence'] *= 0.6
                    if det['confidence'] >= 0.3:
                        verified.append(det)
            except Exception:
                # YOLO failed on crop — trust HSV
                verified.append(det)

        return verified

    def _color_detect(self) -> list:
        """Fallback color-based detection (CPU-friendly)."""
        detections = []

        if self._latest_rgb is None:
            return detections

        hsv = cv2.cvtColor(self._latest_rgb, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        # Detect orange cones
        cone_detections = self._detect_orange_cones(hsv)
        detections.extend(cone_detections)

        # Detect red signs (stop/yield)
        sign_detections = self._detect_red_signs(hsv)
        detections.extend(sign_detections)

        # Detect traffic light state
        light_detections = self._detect_traffic_lights(hsv)
        detections.extend(light_detections)

        return detections

    def _detect_orange_cones(self, hsv, y_offset=0) -> list:
        """Detect orange traffic cones using color."""
        detections = []

        lower = np.array(DETECTION_CONFIG['cone_hsv_lower'])
        upper = np.array(DETECTION_CONFIG['cone_hsv_upper'])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                # Adjust y coordinate for ROI offset
                y += y_offset
                # Estimate distance based on apparent size
                estimated_distance = self._estimate_distance_from_size(area, 'cone')

                detections.append({
                    'class': 'traffic_cone',
                    'confidence': min(area / 5000.0, 1.0),
                    'distance': estimated_distance,
                    'bbox': (x, y, w, h),
                })

        return detections

    def _detect_red_signs(self, hsv, y_offset=0) -> list:
        """Detect red signs (stop/yield) using color and shape."""
        detections = []
        height, width = hsv.shape[:2]

        # Focus on right side of image where signs typically appear
        roi = hsv[:, int(2*width/3):]

        # Red color detection
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(roi, lower_red2, upper_red2)
        mask = mask1 | mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue

            # Approximate shape
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            sides = len(approx)

            sign_type = None
            if sides == 3:
                sign_type = 'yield sign'
            elif 7 <= sides <= 9:
                sign_type = 'stop sign'

            if sign_type:
                x, y, w, h = cv2.boundingRect(cnt)
                x += int(2*width/3)  # Adjust for ROI offset (horizontal)
                y += y_offset  # Adjust for ROI offset (vertical)

                detections.append({
                    'class': sign_type,
                    'confidence': 0.8,
                    'distance': self._estimate_distance_from_size(area, 'sign'),
                    'bbox': (x, y, w, h),
                })

        return detections

    def _detect_traffic_lights(self, hsv) -> list:
        """Detect traffic light state (red/green)."""
        detections = []
        height, width = hsv.shape[:2]

        # ROI: upper-center portion of image
        offset = 20
        roi = hsv[offset:int(height/3), int(width/3):int(2*width/3)]

        # Color thresholds
        lower_red = np.array([0, 200, 200])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([90, 255, 255])

        mask_red = cv2.inRange(roi, lower_red, upper_red)
        mask_green = cv2.inRange(roi, lower_green, upper_green)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)

        if red_pixels > 5 and red_pixels > green_pixels:
            detections.append({
                'class': 'traffic_light_red',
                'confidence': min(red_pixels / 100.0, 1.0),
                'distance': None,
                'bbox': None,
            })
        elif green_pixels > 30:
            detections.append({
                'class': 'traffic_light_green',
                'confidence': min(green_pixels / 100.0, 1.0),
                'distance': None,
                'bbox': None,
            })

        return detections

    def _classify_traffic_light_color(self, image, x1, y1, x2, y2) -> str:
        """Classify a COCO-detected traffic light as red or green using HSV analysis.

        Args:
            image: BGR image
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            'traffic_light_red', 'traffic_light_green', or None if indeterminate
        """
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None

        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red detection (two ranges wrapping around hue=0)
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

        # Green detection
        mask_green = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))
        green_pixels = cv2.countNonZero(mask_green)

        # Require minimum pixel count to classify
        min_pixels = 5
        if red_pixels >= min_pixels and red_pixels > green_pixels:
            return 'traffic_light_red'
        elif green_pixels >= min_pixels and green_pixels > red_pixels:
            return 'traffic_light_green'
        return None

    def _estimate_distance_from_size(self, area: float, obj_type: str) -> float:
        """Estimate distance based on apparent object size.
        Conservative: only report close when bbox is very large (at stop line)."""
        if obj_type == 'cone':
            if area > 5000: return 0.2
            elif area > 2500: return 0.3
            elif area > 1000: return 0.5
            else: return 1.0
        elif obj_type == 'sign':
            if area > 5000: return 0.2
            elif area > 3000: return 0.4
            elif area > 1500: return 0.6
            else: return 1.5
        elif 'person' in obj_type:
            if area > 25000: return 0.2
            elif area > 15000: return 0.4
            elif area > 10000: return 0.5
            elif area > 5000: return 0.8
            else: return 2.0
        return 1.5

    def _is_different_stop_sign(self, det: dict) -> bool:
        """Check if this stop sign detection is a spatially different sign from the last one."""
        bbox = det.get('bbox')
        if bbox is None:
            return True  # No bbox info, assume different
        _, y, _, h = bbox
        bbox_center_y = y + h / 2
        if self._last_stop_sign_bbox_y is None:
            return True
        return abs(bbox_center_y - self._last_stop_sign_bbox_y) > self._stop_sign_bbox_threshold

    def _process_detections(self, detections: list) -> tuple:
        """
        Process detections and decide whether to stop.

        Reference repo approach: confidence IS the distance proxy.
        High confidence = object is close/large enough to require action.
        No distance thresholds for stop decisions.
        Yield signs are ignored (reference repo doesn't handle them).

        Returns: (should_stop: bool, reason: str, pause_duration: float)
        """
        current_time = time.time()
        cfg = DETECTION_CONFIG

        # Check if traffic light cooldown has expired
        if self._cross_cooldown:
            cooldown_duration = cfg.get('traffic_light_cooldown', 6.0)
            if current_time - self._cross_cooldown_start >= cooldown_duration:
                self._cross_cooldown = False

        # Track whether we see a red light this frame (for cross_waiting timeout)
        saw_red_this_frame = False
        stop_sign_seen = False

        for det in detections:
            obj_class = det['class']
            confidence = det['confidence']

            # --- Confidence-only stop decisions (reference repo approach) ---

            # Pedestrian — handled by Kalman tracker (checked after loop)
            if obj_class == 'person':
                continue

            # Traffic cone — MPCC handles avoidance via /obstacle_positions
            elif obj_class in ['traffic_cone', 'sports ball', 'cone']:
                continue

            # Stop sign — confidence >= threshold means at the stop line
            elif obj_class in ['stop sign', 'stop']:
                if confidence >= cfg['sign_confidence']:
                    stop_sign_seen = True
                    # Skip if currently at a stop sign or just completed one
                    if self._stop_sign_wait_complete:
                        continue
                    if self._at_stop_sign:
                        continue
                    # Cooldown: check if this is a different sign
                    last_time = self._last_detection_time.get('stop sign', 0)
                    cooldown = cfg.get('stop_sign_cooldown', 15.0)
                    if current_time - last_time < cooldown:
                        if not self._is_different_stop_sign(det):
                            continue
                    # Record this sign's position and trigger stop
                    bbox = det.get('bbox')
                    if bbox is not None:
                        _, y, _, h = bbox
                        self._last_stop_sign_bbox_y = y + h / 2
                    self._last_detection_time['stop sign'] = current_time
                    return True, f"Stop sign (conf={confidence:.2f})", cfg['stop_sign_pause']

            # Yield sign — handled after loop (yield only if obstacle present)
            elif obj_class in ['yield sign', 'yield']:
                continue

            # Red traffic light — confidence >= threshold means close enough
            elif obj_class in ['traffic_light_red', 'red']:
                if confidence >= cfg['red_light_confidence']:
                    if self._cross_cooldown:
                        continue
                    saw_red_this_frame = True
                    if not self._cross_waiting:
                        self._cross_waiting = True
                        self._cross_waiting_start = current_time
                    self._cross_waiting_no_red_frames = 0
                    return True, f"Red light (conf={confidence:.2f})", 0

            # Green traffic light — clear to go
            elif obj_class in ['traffic_light_green', 'green']:
                if confidence >= cfg['green_light_confidence']:
                    if self._cross_waiting:
                        self._cross_waiting = False
                        self._cross_waiting_no_red_frames = 0
                        self._cross_cooldown = True
                        self._cross_cooldown_start = current_time
                        self.get_logger().info(
                            "Traffic light GREEN - proceeding (cooldown %.0fs)" %
                            cfg.get('traffic_light_cooldown', 6.0))
                    continue

            # Round sign - informational only
            elif obj_class == 'round':
                continue

            # COCO traffic light with unknown state
            elif obj_class == 'traffic light':
                continue

            # Vehicle detection — disabled for motion_enable to prevent false
            # positives (own car reflections, parked cars). Obstacle avoidance
            # is handled by MPCC via /obstacle_positions instead.
            elif obj_class == 'car':
                continue

        # Kalman tracker: check if any tracked pedestrian is on the road
        ped_stop, ped_reason = self._ped_tracker.any_on_road()
        if ped_stop:
            return True, ped_reason, 0

        # Yield sign: only yield for tracked pedestrians actually on the road.
        # Post-action suppression prevents livelock from repeated detections.
        has_yield = any(
            d['class'] in ('yield sign', 'yield') and d['confidence'] >= cfg['sign_confidence']
            for d in detections
        )
        if has_yield:
            # Check post-action suppression first (prevents re-triggering after yield)
            if current_time < self._yield_suppress_until:
                pass  # Suppressed — vehicle is driving past the sign
            elif ped_stop:
                self._yield_active = True
                self._last_detection_time['yield sign'] = current_time
                return True, "Yield: pedestrian on road (tracked)", 0
            elif self._yield_active:
                # Pedestrian cleared while yielding — start suppression timer
                self._yield_active = False
                self._yield_suppress_until = current_time + self._yield_suppress_duration
                self.get_logger().info(
                    f"Yield sign: pedestrian cleared, suppressing for {self._yield_suppress_duration:.0f}s")
        else:
            # Yield sign no longer visible — if we were yielding, start suppression
            if self._yield_active:
                self._yield_active = False
                self._yield_suppress_until = current_time + self._yield_suppress_duration
                self.get_logger().info(
                    f"Yield sign: out of view, suppressing for {self._yield_suppress_duration:.0f}s")

        # Reset stop sign completion flag when sign goes out of view
        if not stop_sign_seen and self._stop_sign_wait_complete:
            self._stop_sign_wait_complete = False
            self._at_stop_sign = False

        # If cross_waiting but no red detected this frame, track timeout
        if self._cross_waiting:
            if not saw_red_this_frame:
                self._cross_waiting_no_red_frames += 1
                if self._cross_waiting_no_red_frames >= self._cross_waiting_no_red_threshold:
                    self.get_logger().info("Traffic light: no red visible for 2s, assuming passed")
                    self._cross_waiting = False
                    self._cross_waiting_no_red_frames = 0
                    self._cross_cooldown = True
                    self._cross_cooldown_start = current_time
                    return False, "", 0
            if current_time - self._cross_waiting_start > self._cross_waiting_timeout:
                self.get_logger().info("Traffic light: cross_waiting timeout (15s), resuming")
                self._cross_waiting = False
                self._cross_waiting_no_red_frames = 0
                return False, "", 0
            return True, "Waiting for green light", 0

        return False, "", 0

    def _process_traffic_control(self, detections: list) -> TrafficControlState:
        """
        Process detections for traffic control state.

        Reference repo approach: confidence-only, no distance thresholds.
        Yield signs are ignored (reference repo doesn't handle them).

        Returns:
            TrafficControlState with current traffic control information
        """
        current_time = time.time()
        cfg = DETECTION_CONFIG
        state = TrafficControlState()

        # Check if traffic light cooldown has expired
        if self._cross_cooldown:
            cooldown_duration = cfg.get('traffic_light_cooldown', 6.0)
            if current_time - self._cross_cooldown_start >= cooldown_duration:
                self._cross_cooldown = False

        for det in detections:
            obj_class = det['class']
            confidence = det.get('confidence', 0.0)

            # Red traffic light — confidence-only
            if obj_class in ('traffic_light_red', 'red'):
                if confidence < cfg['red_light_confidence']:
                    continue
                if self._cross_cooldown:
                    continue

                state.control_type = "traffic_light"
                state.light_state = "red"
                state.should_stop = True
                state.stop_duration = 0.0

                if self._last_traffic_light_state != "red":
                    self.get_logger().info(f"Traffic light: RED (conf={confidence:.2f}) - stopping")
                    self._traffic_light_transition_time = current_time
                self._last_traffic_light_state = "red"
                self._cross_waiting = True

                self._at_stop_sign = False
                self._stop_sign_wait_complete = False
                return state

            elif obj_class in ('traffic_light_green', 'green'):
                if confidence < cfg['green_light_confidence']:
                    continue

                state.control_type = "traffic_light"
                state.light_state = "green"
                state.should_stop = False

                if self._cross_waiting:
                    self.get_logger().info("Traffic light turned GREEN - can proceed")
                    self._traffic_light_transition_time = current_time
                    self._cross_waiting = False
                    self._cross_cooldown = True
                    self._cross_cooldown_start = current_time

                self._last_traffic_light_state = "green"
                return state

            elif obj_class == 'traffic light':
                continue

            # Stop sign — confidence-only
            elif obj_class in ('stop sign', 'stop'):
                if confidence < cfg['sign_confidence']:
                    continue

                last_time = self._last_detection_time.get('stop sign', 0)
                cooldown = cfg.get('stop_sign_cooldown', 15.0)
                if current_time - last_time < cooldown:
                    continue

                state.control_type = "stop_sign"

                if not self._at_stop_sign:
                    self._at_stop_sign = True
                    self._stop_sign_start_time = current_time
                    self._stop_sign_wait_complete = False
                    self.get_logger().info(
                        f"Stop sign (conf={confidence:.2f}) - starting {cfg['stop_sign_pause']:.1f}s wait")

                if self._at_stop_sign and not self._stop_sign_wait_complete:
                    elapsed = current_time - self._stop_sign_start_time
                    remaining = cfg['stop_sign_pause'] - elapsed
                    if remaining > 0:
                        state.should_stop = True
                        state.stop_duration = remaining
                    else:
                        self._stop_sign_wait_complete = True
                        self._last_detection_time['stop sign'] = current_time
                        self.get_logger().info("Stop sign wait complete - can proceed")
                        state.should_stop = False
                        state.stop_duration = 0.0
                elif self._at_stop_sign and self._stop_sign_wait_complete:
                    state.should_stop = False
                    state.stop_duration = 0.0

                return state

            # Yield sign — only yield for tracked pedestrians on road.
            # Post-action suppression prevents infinite stop loop.
            elif obj_class in ('yield sign', 'yield'):
                if confidence < cfg['sign_confidence']:
                    continue
                # Check post-action suppression first
                if current_time < self._yield_suppress_until:
                    continue  # Suppressed — vehicle is driving past the sign
                ped_on_road, _ = self._ped_tracker.any_on_road()
                if ped_on_road:
                    self._yield_active = True
                    self._last_detection_time['yield sign'] = current_time
                    state.control_type = "yield_sign"
                    state.should_stop = True
                    state.stop_duration = 0.0
                    return state
                elif self._yield_active:
                    # Pedestrian cleared — start suppression and let vehicle go
                    self._yield_active = False
                    self._yield_suppress_until = current_time + self._yield_suppress_duration
                    self.get_logger().info(
                        f"Yield sign: pedestrian cleared, suppressing for {self._yield_suppress_duration:.0f}s")

        # No traffic control detected this frame
        state.control_type = "none"
        state.should_stop = False

        # If cross_waiting (stopped at red, lost sight), stay stopped
        if self._cross_waiting:
            state.control_type = "traffic_light"
            state.light_state = "red"
            state.should_stop = True
            state.stop_duration = 0.0
            return state

        # Handle stop sign state when sign is no longer visible
        if self._at_stop_sign:
            if self._stop_sign_wait_complete:
                self.get_logger().info("Stop sign: no longer visible, resetting state")
                self._at_stop_sign = False
                self._stop_sign_wait_complete = False
                self._stop_sign_start_time = None
            else:
                elapsed = current_time - self._stop_sign_start_time if self._stop_sign_start_time else 0
                remaining = cfg['stop_sign_pause'] - elapsed
                if remaining > 0:
                    state.control_type = "stop_sign"
                    state.should_stop = True
                    state.stop_duration = remaining
                else:
                    self._stop_sign_wait_complete = True
                    self._last_detection_time['stop sign'] = current_time
                    self.get_logger().info("Stop sign wait complete (sign not visible) - can proceed")
                    state.should_stop = False

        return state

    def _publish_traffic_control_state(self, state: TrafficControlState):
        """Publish traffic control state as JSON-encoded string."""
        msg = String()
        msg.data = state.to_json()
        self._traffic_control_pub.publish(msg)

    def _publish_obstacle_positions(self, detections: list):
        """
        Publish obstacle positions for MPCC controller.

        Extracts pedestrians, cones, and vehicles with position estimates
        for the MPCC to use in obstacle avoidance constraints.
        """
        obstacles = ObstaclePositions()
        obstacles.timestamp = time.time()

        for det in detections:
            obj_class = det['class']
            distance = det.get('distance')
            bbox = det.get('bbox')
            confidence = det.get('confidence', 0.0)

            # Only include obstacles relevant for collision avoidance
            if obj_class not in ['person', 'car', 'traffic_cone', 'sports ball', 'cone']:
                continue

            # Skip if no distance estimate
            if distance is None:
                continue

            # Estimate position relative to camera
            # For now, assume obstacle is directly ahead
            # TODO: Use bbox center to estimate lateral offset
            obs = ObstaclePosition()
            obs.obj_class = obj_class

            # Distance is forward (x in vehicle frame, need transform to map)
            # For simplicity, publish in camera/vehicle frame
            obs.x = distance  # Forward distance
            obs.y = 0.0  # Assume centered (could use bbox to estimate)

            # Set radius based on object type
            if obj_class == 'person':
                obs.radius = 0.3  # Pedestrian radius
            elif obj_class == 'car':
                obs.radius = 0.5  # Vehicle radius
            else:  # cone
                obs.radius = 0.15  # Cone radius

            # Estimate lateral offset from bbox center if available
            if bbox is not None and self._latest_rgb is not None:
                x, y, w, h = bbox
                img_w = self._latest_rgb.shape[1]
                bbox_center_x = x + w / 2
                # Normalize to [-1, 1] range
                lateral_offset = (bbox_center_x - img_w / 2) / (img_w / 2)
                # Rough estimate: scale by distance and FOV
                # Assuming ~60 degree horizontal FOV
                obs.y = lateral_offset * distance * 0.5  # Approximate

            obstacles.add_obstacle(obs)

        # Publish
        msg = String()
        msg.data = obstacles.to_json()
        self._obstacle_positions_pub.publish(msg)

    def _publish_motion(self, enabled: bool):
        """Publish motion enable/disable."""
        msg = Bool()
        msg.data = enabled
        self._motion_pub.publish(msg)

    def _publish_info(self, detections: list, status: str):
        """Publish detection info as JSON."""
        info = {
            'status': status,
            'motion_enabled': self._motion_enabled,
            'timestamp': time.time(),
            'detections': detections,
        }
        msg = String()
        msg.data = json.dumps(info)
        self._info_pub.publish(msg)

    def _publish_visualization(self, detections: list):
        """Publish annotated image for debugging."""
        if self._latest_rgb is None:
            return

        viz = self._latest_rgb.copy()

        for det in detections:
            bbox = det.get('bbox')
            if bbox:
                x, y, w_box, h_box = bbox
                color = (0, 255, 0) if self._motion_enabled else (0, 0, 255)
                cv2.rectangle(viz, (x, y), (x+w_box, y+h_box), color, 2)

                label = f"{det['class']}: {det['confidence']:.2f}"
                if det['distance']:
                    label += f" ({det['distance']:.2f}m)"
                cv2.putText(viz, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Status overlay
        status_color = (0, 255, 0) if self._motion_enabled else (0, 0, 255)
        status_text = "CLEAR" if self._motion_enabled else "STOP"
        cv2.putText(viz, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Publish
        try:
            msg = self._bridge.cv2_to_imgmsg(viz, 'bgr8')
            self._viz_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Viz publish failed: {e}")


def main():
    rclpy.init()
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
