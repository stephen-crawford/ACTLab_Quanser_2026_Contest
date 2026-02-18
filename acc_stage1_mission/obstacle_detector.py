#!/usr/bin/env python3
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

# Import traffic control state message classes
from acc_stage1_mission.traffic_control_state import (
    TrafficControlState,
    ObstaclePosition,
    ObstaclePositions,
)

# =============================================================================
# DETECTION CONFIGURATION - EDIT THESE TO TUNE BEHAVIOR
# Thresholds matched to reference repo (MPC_node.py) for QLabs environment.
# =============================================================================
DETECTION_CONFIG = {
    # Detection distances (meters) - stop if object closer than this
    'pedestrian_stop_distance': 1.0,
    'cone_stop_distance': 0.6,
    'vehicle_stop_distance': 0.8,
    'sign_stop_distance': 0.9,

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
    'yield_sign_pause': 2.0,       # Yield sign pause (competition: -2 stars for failure to yield)
    'pedestrian_clear_delay': 1.0,
    'stop_sign_cooldown': 5.0,     # Per-sign spatial cooldown (was 15s global - too aggressive)
    'yield_sign_cooldown': 5.0,    # Per-sign spatial cooldown
    'traffic_light_cooldown': 6.0, # After green: ignore reds for 6s (was 10s - too long)

    # YOLO settings
    'yolo_classes': [0, 2, 9, 11, 33],  # COCO: person, car, traffic light, stop sign, sports ball
    'image_width': 640,
    'image_height': 480,

    # Custom YOLO model path (QLabs-trained, 8 classes)
    # Will be searched in order; first found wins
    'custom_model_paths': [
        '/workspaces/isaac_ros-dev/ros2/src/polyctrl/polyctrl/best.pt',
        '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/models/best.pt',
    ],

    # Fallback color detection (when YOLO unavailable)
    'use_color_fallback': True,
    'cone_hsv_lower': [5, 150, 150],
    'cone_hsv_upper': [15, 255, 255],

    # CAMERA MASK
    'mask_bottom_fraction': 0.18,
    'mask_left_fraction': 0.0,
    'mask_right_fraction': 0.0,
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
CUSTOM_YOLO_CLASSES = {
    0: 'cone',
    1: 'green',     # Green traffic light
    2: 'red',       # Red traffic light
    3: 'yellow',    # Yellow traffic light
    4: 'stop',      # Stop sign
    5: 'yield',     # Yield sign
    6: 'round',     # Roundabout sign
    7: 'person',    # Pedestrian
}

# Whitelist of allowed class names - detections not in this set are discarded
ALLOWED_CLASSES = {
    'person', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'sports ball',
    'traffic_cone', 'yield sign',                    # From color detection
    'traffic_light_red', 'traffic_light_green',      # From color detection
    # Custom model classes
    'cone', 'green', 'red', 'yellow', 'stop', 'yield', 'round',
}


class ObstacleDetector(Node):
    """
    ROS2 node for obstacle detection using YOLO and/or color detection.

    To modify detection behavior:
    1. Edit DETECTION_CONFIG above
    2. Override methods like _process_pedestrian(), _process_cone(), etc.
    3. Add new detection types by extending _process_detections()
    """

    def __init__(self):
        super().__init__('obstacle_detector')

        # Declare ROS parameters (can be overridden at launch)
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('camera_topic', '/camera/color_image')
        self.declare_parameter('depth_topic', '/camera/depth_image')

        self._use_yolo = self.get_parameter('use_yolo').value
        self._debug_viz = self.get_parameter('debug_visualization').value
        self._camera_topic = self.get_parameter('camera_topic').value
        self._depth_topic = self.get_parameter('depth_topic').value

        # Initialize YOLO if available
        self._yolo = None
        self._depth_aligner = None
        if self._use_yolo:
            self._init_yolo()

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

        # Main processing timer
        self._timer = self.create_timer(1.0/30.0, self._process_frame)

        # Publish initial state
        self._publish_motion(True)

        self.get_logger().info(
            f"ObstacleDetector initialized - YOLO: {self._yolo is not None}, "
            f"Debug viz: {self._debug_viz}"
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

            # Try custom QLabs-trained model first
            custom_model_path = None
            for path in DETECTION_CONFIG.get('custom_model_paths', []):
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

            # Force CPU - RTX 5080 (sm_120) needs PyTorch with Python 3.10+ for CUDA support
            self._yolo_device = 'cpu'

            # Override thresholds for custom model (trained on QLabs, higher confidence)
            if self._using_custom_model:
                DETECTION_CONFIG['sign_confidence'] = 0.91
                DETECTION_CONFIG['red_light_confidence'] = 0.83
                DETECTION_CONFIG['green_light_confidence'] = 0.84
                DETECTION_CONFIG['pedestrian_confidence'] = 0.70
                DETECTION_CONFIG['cone_confidence'] = 0.80
                DETECTION_CONFIG['round_sign_confidence'] = 0.90
                self.get_logger().info("Using custom model confidence thresholds")

            self.get_logger().info(f"YOLO initialized: {model_desc}, device={self._yolo_device}")

        except ImportError as e:
            self.get_logger().warn(f"YOLO not available: {e}")
            self.get_logger().info("Falling back to color-based detection")
            self._yolo = None

    def _rgb_callback(self, msg: Image):
        """Store latest RGB image."""
        try:
            self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def _get_detection_roi(self, image):
        """
        Get the region of interest for detection, excluding masked areas.
        Returns (roi_image, y_offset) where y_offset is the top of the ROI.

        This masks out the car's camera housing visible at the bottom of frame.
        """
        if image is None:
            return None, 0

        h, w = image.shape[:2]
        cfg = DETECTION_CONFIG

        # Calculate mask boundaries
        top = 0
        bottom = int(h * (1 - cfg.get('mask_bottom_fraction', 0.15)))
        left = int(w * cfg.get('mask_left_fraction', 0))
        right = int(w * (1 - cfg.get('mask_right_fraction', 0)))

        # Extract ROI
        roi = image[top:bottom, left:right]
        return roi, top

    def _bbox_in_masked_region(self, bbox, image_shape) -> bool:
        """Check if a bounding box is primarily in the masked (ignored) region."""
        if bbox is None:
            return False

        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        cfg = DETECTION_CONFIG

        # Check if bbox center is in bottom masked region
        bbox_center_y = y + h / 2
        mask_top = int(img_h * (1 - cfg.get('mask_bottom_fraction', 0.15)))

        return bbox_center_y > mask_top

    def _depth_callback(self, msg: Image):
        """Store latest depth image."""
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def _process_frame(self):
        """Main processing loop - runs at 30Hz."""
        current_time = time.time()

        # Check if we're in a timed pause
        if current_time < self._pause_until:
            return

        if self._latest_rgb is None:
            return

        # Run detection
        detections = []

        if self._yolo is not None:
            detections = self._yolo_detect()
        elif DETECTION_CONFIG['use_color_fallback']:
            detections = self._color_detect()

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

                    # SKIP detections in the masked region (car's camera housing)
                    if self._bbox_in_masked_region(bbox, img_shape):
                        continue

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

    def _color_detect(self) -> list:
        """Fallback color-based detection (CPU-friendly)."""
        detections = []

        if self._latest_rgb is None:
            return detections

        # Get ROI (excluding camera housing at bottom)
        roi, y_offset = self._get_detection_roi(self._latest_rgb)
        if roi is None:
            return detections

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        # Detect orange cones
        cone_detections = self._detect_orange_cones(hsv, y_offset)
        detections.extend(cone_detections)

        # Detect red signs (stop/yield)
        sign_detections = self._detect_red_signs(hsv, y_offset)
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
        Larger area = closer object. Tuned for earlier detection."""
        if obj_type == 'cone':
            if area > 5000: return 0.2
            elif area > 2500: return 0.4
            elif area > 1000: return 0.6
            elif area > 400: return 0.8
            else: return 1.0
        elif obj_type == 'sign':
            if area > 4000: return 0.3
            elif area > 2000: return 0.5
            elif area > 800: return 0.7
            else: return 1.0
        elif 'person' in obj_type:
            if area > 25000: return 0.3
            elif area > 10000: return 0.5
            elif area > 4000: return 0.8
            else: return 1.2
        return 1.0

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

        Returns: (should_stop: bool, reason: str, pause_duration: float)

        Traffic light state machine:
        - RED (conf>=threshold) -> stop, set cross_waiting=True
        - While cross_waiting: stay stopped until GREEN detected
        - On GREEN -> proceed, enter cross_cooldown for 6s
        - During cooldown: ignore red detections (prevents re-triggering same light)
        - cross_waiting auto-expires after timeout or if no red seen for 2s

        Stop sign detection:
        - Uses per-sign spatial cooldown (bbox Y position) instead of global timer
        - This allows detecting multiple stop signs in quick succession
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

        for det in detections:
            obj_class = det['class']
            confidence = det['confidence']
            distance = det['distance']

            # Discard implausibly close readings (depth noise, camera housing, self-detection)
            if distance is not None and distance < 0.25:
                continue

            # Pedestrian detection
            if obj_class == 'person':
                if confidence >= cfg['pedestrian_confidence']:
                    if distance is None or distance < cfg['pedestrian_stop_distance']:
                        return True, f"Pedestrian detected (dist={distance})", 0

            # Traffic cone detection
            elif obj_class in ['traffic_cone', 'sports ball', 'cone']:
                if confidence >= cfg['cone_confidence']:
                    if distance is None or distance < cfg['cone_stop_distance']:
                        return True, f"Obstacle/cone detected (dist={distance})", 0

            # Stop sign - spatial cooldown instead of global timer
            elif obj_class in ['stop sign', 'stop']:
                if confidence >= cfg['sign_confidence']:
                    if distance is None or distance < cfg['sign_stop_distance']:
                        # Skip if currently at a stop sign or just completed one
                        if self._stop_sign_wait_complete:
                            continue
                        if self._at_stop_sign:
                            continue
                        # Spatial cooldown: check if this is a different sign
                        last_time = self._last_detection_time.get('stop sign', 0)
                        cooldown = cfg.get('stop_sign_cooldown', 5.0)
                        if current_time - last_time < cooldown:
                            # Within cooldown - only stop if it's a spatially different sign
                            if not self._is_different_stop_sign(det):
                                continue
                        # Record this sign's position and trigger stop
                        bbox = det.get('bbox')
                        if bbox is not None:
                            _, y, _, h = bbox
                            self._last_stop_sign_bbox_y = y + h / 2
                        self._last_detection_time['stop sign'] = current_time
                        return True, "Stop sign detected", cfg['stop_sign_pause']

            # Yield sign - spatial cooldown
            elif obj_class in ['yield sign', 'yield']:
                if confidence >= cfg['sign_confidence']:
                    if distance is None or distance < cfg['sign_stop_distance']:
                        last_time = self._last_detection_time.get('yield sign', 0)
                        cooldown = cfg.get('yield_sign_cooldown', 5.0)
                        if current_time - last_time < cooldown:
                            continue
                        self._last_detection_time['yield sign'] = current_time
                        return True, "Yield sign detected", cfg['yield_sign_pause']

            # Red traffic light
            elif obj_class in ['traffic_light_red', 'red']:
                if confidence >= cfg['red_light_confidence']:
                    # During cooldown after green, ignore red detections
                    if self._cross_cooldown:
                        continue
                    saw_red_this_frame = True
                    if not self._cross_waiting:
                        self._cross_waiting = True
                        self._cross_waiting_start = current_time
                    self._cross_waiting_no_red_frames = 0
                    return True, "Red light detected", 0

            # Green traffic light
            elif obj_class in ['traffic_light_green', 'green']:
                if confidence >= cfg['green_light_confidence']:
                    if self._cross_waiting:
                        # Was waiting at red -> green detected -> proceed
                        self._cross_waiting = False
                        self._cross_waiting_no_red_frames = 0
                        self._cross_cooldown = True
                        self._cross_cooldown_start = current_time
                        self.get_logger().info(
                            "Traffic light GREEN - proceeding (cooldown %.0fs)" %
                            cfg.get('traffic_light_cooldown', 6.0))
                    # Green means go - don't stop
                    continue

            # Round sign - informational only
            elif obj_class == 'round':
                continue

            # COCO 'traffic light' with unknown state - try HSV classification
            # Do NOT blindly stop for unknown state (was causing false stops)
            elif obj_class == 'traffic light':
                continue  # Skip - rely on HSV classification in _yolo_detect

            # Vehicle detection
            elif obj_class == 'car':
                if confidence >= cfg['vehicle_confidence']:
                    if distance is None or distance < cfg['vehicle_stop_distance']:
                        return True, f"Vehicle detected (dist={distance})", 0

        # If cross_waiting but no red detected this frame, track timeout
        if self._cross_waiting:
            if not saw_red_this_frame:
                self._cross_waiting_no_red_frames += 1
                # If no red seen for threshold frames (~2s), assume we've passed the light
                if self._cross_waiting_no_red_frames >= self._cross_waiting_no_red_threshold:
                    self.get_logger().info("Traffic light: no red visible for 2s, assuming passed")
                    self._cross_waiting = False
                    self._cross_waiting_no_red_frames = 0
                    self._cross_cooldown = True
                    self._cross_cooldown_start = current_time
                    return False, "", 0
            # Also check absolute timeout
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

        Uses the reference repo's traffic light state machine:
        - RED (conf>=0.83) -> stop, cross_waiting=True
        - While cross_waiting: stay stopped until GREEN (conf>=0.84)
        - On GREEN -> proceed, cross_cooldown for 10s
        - During cooldown: ignore red detections

        Args:
            detections: List of detection dictionaries

        Returns:
            TrafficControlState with current traffic control information
        """
        current_time = time.time()
        cfg = DETECTION_CONFIG
        state = TrafficControlState()

        for det in detections:
            obj_class = det['class']
            distance = det.get('distance')
            confidence = det.get('confidence', 0.0)

            # Handle traffic light detections (custom model: 'red'/'green'; COCO+HSV: 'traffic_light_red'/'traffic_light_green')
            if obj_class in ('traffic_light_red', 'red'):
                if confidence < cfg['red_light_confidence']:
                    continue
                # During cooldown after green, ignore red
                if self._cross_cooldown:
                    cooldown_duration = cfg.get('traffic_light_cooldown', 10.0)
                    if current_time - self._cross_cooldown_start < cooldown_duration:
                        continue
                    else:
                        self._cross_cooldown = False

                state.control_type = "traffic_light"
                state.light_state = "red"
                state.distance = distance if distance else 1.0
                state.should_stop = True
                state.stop_duration = 0.0

                if self._last_traffic_light_state != "red":
                    self.get_logger().info("Traffic light: RED - stopping")
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
                state.distance = distance if distance else 1.0
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
                # COCO YOLO detected traffic light but state unknown
                # Do NOT stop for unknown state - rely on HSV color classification
                continue

            # Handle stop sign detections
            elif obj_class in ('stop sign', 'stop'):
                if confidence < cfg['sign_confidence']:
                    continue

                # Spatial cooldown: check if this is a different sign
                last_time = self._last_detection_time.get('stop sign', 0)
                cooldown = cfg.get('stop_sign_cooldown', 5.0)
                if current_time - last_time < cooldown:
                    continue

                state.control_type = "stop_sign"
                state.distance = distance if distance else 0.5

                stop_distance_threshold = cfg['sign_stop_distance']

                if distance is not None and distance < stop_distance_threshold:
                    if not self._at_stop_sign:
                        self._at_stop_sign = True
                        self._stop_sign_start_time = current_time
                        self._stop_sign_wait_complete = False
                        self.get_logger().info(
                            f"Arrived at stop sign - starting {cfg['stop_sign_pause']:.1f}s wait")

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
                else:
                    if self._at_stop_sign and self._stop_sign_wait_complete:
                        self.get_logger().info("Stop sign: moved past, resetting state")
                        self._at_stop_sign = False
                        self._stop_sign_wait_complete = False
                        self._stop_sign_start_time = None
                    state.should_stop = False
                    return state

            # Handle yield sign
            elif obj_class in ('yield sign', 'yield'):
                if confidence < cfg['sign_confidence']:
                    continue

                state.control_type = "yield_sign"
                state.distance = distance if distance else 0.5

                if distance is not None and distance < cfg['sign_stop_distance']:
                    state.should_stop = True
                    state.stop_duration = cfg['yield_sign_pause']
                else:
                    state.should_stop = False

                return state

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
            if obj_class not in ['person', 'car', 'traffic_cone', 'sports ball']:
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
        h, w = viz.shape[:2]
        cfg = DETECTION_CONFIG

        # Draw masked region (camera housing) with semi-transparent overlay
        mask_top = int(h * (1 - cfg.get('mask_bottom_fraction', 0.15)))
        overlay = viz.copy()
        cv2.rectangle(overlay, (0, mask_top), (w, h), (128, 128, 128), -1)
        cv2.addWeighted(overlay, 0.5, viz, 0.5, 0, viz)
        cv2.line(viz, (0, mask_top), (w, mask_top), (255, 255, 0), 2)
        cv2.putText(viz, "MASKED (camera)", (10, mask_top + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
