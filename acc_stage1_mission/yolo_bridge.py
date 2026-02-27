#!/usr/bin/env python3
from __future__ import annotations

"""
YOLO Bridge Node - Bidirectional bridge between ROS2 and standalone Python 3.10 YOLO detector.

This node:
1. Subscribes to camera images from QLabs and sends them to the GPU detector
2. Receives detection results from the GPU detector and publishes to ROS2 topics:
   - /obstacle_positions (String): JSON-encoded obstacle positions for MPCC avoidance
   - /obstacle_info (String): JSON detection details

NOTE: /motion_enable and /traffic_control_state are handled exclusively by the
C++ sign_detector_node. This bridge only provides supplementary obstacle position
data for MPCC path planning.

Usage:
    ros2 run acc_stage1_mission yolo_bridge
"""

import json
import math
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

from acc_stage1_mission.traffic_control_state import (
    TrafficControlState,
    ObstaclePosition,
    ObstaclePositions,
)
from acc_stage1_mission.road_boundaries import RoadBoundarySpline
from acc_stage1_mission.pedestrian_tracker import PedestrianKalmanTracker

# Camera horizontal FOV for QCar2 (~60 degrees)
CAMERA_HFOV_RAD = math.radians(60.0)

# Whitelist of allowed detection classes - all others are discarded
ALLOWED_CLASSES = {
    'person', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'sports ball',
    'traffic_cone', 'yield sign',
    'traffic_light_red', 'traffic_light_green', 'traffic_light_yellow',
    # Custom model classes
    'cone', 'green', 'red', 'yellow', 'stop', 'yield', 'round',
}

# Socket configuration
BRIDGE_HOST = '0.0.0.0'
RESULT_PORT = 9999   # Receive detection results
FRAME_PORT = 9998    # Send camera frames

# Detection config (mirrors standalone detector — confidence-only, no distance thresholds)
DETECTION_CONFIG = {
    'stop_sign_pause': 3.0,
    'detection_cooldown': 15.0,
    'sign_confidence': 0.91,
    'red_light_confidence': 0.83,
    'green_light_confidence': 0.84,
    'pedestrian_confidence': 0.70,
}


class YoloBridge(Node):
    def __init__(self):
        super().__init__('yolo_bridge')

        # Parameters
        self.declare_parameter('camera_topic', '/camera/color_image')
        self._camera_topic = self.get_parameter('camera_topic').value

        # CV Bridge
        self._bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Publishers
        # NOTE: /motion_enable and /traffic_control_state are handled exclusively
        # by the C++ sign_detector_node. The YOLO bridge only publishes raw
        # detection data for MPCC obstacle avoidance. Publishing to those topics
        # here would cause competing publishers and motion enable flickering.
        self._info_pub = self.create_publisher(String, '/yolo_bridge/status', 10)
        self._obstacle_info_pub = self.create_publisher(String, '/obstacle_info', 10)
        self._obstacle_positions_pub = self.create_publisher(
            String, '/obstacle_positions', 10)

        # Subscribe to camera and depth
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_callback, qos)
        self._latest_depth = None
        self._depth_lock = threading.Lock()
        self._depth_sub = self.create_subscription(
            Image, '/camera/depth_image', self._depth_callback, qos)

        # UDP socket to receive results from standalone detector
        self._result_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._result_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._result_socket.bind((BRIDGE_HOST, RESULT_PORT))
        self._result_socket.settimeout(0.1)

        # TCP socket to send frames to standalone detector
        self._frame_socket = None
        self._client_connected = False

        # TF listener for vehicle pose in map frame
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
            else:
                self.get_logger().warn(f"Road boundaries not found: {boundary_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not load road boundaries: {e}")

        # Kalman filter pedestrian tracker
        self._ped_tracker = PedestrianKalmanTracker(self._road_boundaries)

        # Traffic control state tracking
        self._at_stop_sign = False
        self._stop_sign_start_time = None
        self._stop_sign_wait_complete = False
        self._last_traffic_light_state = "unknown"
        self._traffic_light_transition_time = None
        self._cross_waiting = False
        self._cross_cooldown = False
        self._cross_cooldown_start = 0.0
        self._last_detection_time = {}  # Per-sign cooldown tracking

        self.get_logger().info("=" * 50)
        self.get_logger().info("YOLO Bridge Starting (enhanced)")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Result listener:  UDP port {RESULT_PORT}")
        self.get_logger().info(f"Frame server:     TCP port {FRAME_PORT}")
        self.get_logger().info(f"Camera topic:     {self._camera_topic}")
        self.get_logger().info("")
        self.get_logger().info("Publishing to:")
        self.get_logger().info("  /obstacle_positions, /obstacle_info")
        self.get_logger().info("  (motion_enable + traffic_control handled by C++ sign_detector)")
        self.get_logger().info("")
        self.get_logger().info("Waiting for:")
        self.get_logger().info("  1. Camera frames from QLabs")
        self.get_logger().info("  2. GPU detector to connect")
        self.get_logger().info("=" * 50)

        # Start receiver thread
        self._running = True
        self._result_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._result_thread.start()

        # Start frame server thread
        self._frame_thread = threading.Thread(target=self._frame_server, daemon=True)
        self._frame_thread.start()

        # Heartbeat timer
        self._timer = self.create_timer(1.0, self._heartbeat)
        self._last_msg_time = 0
        self._frame_count = 0

    def _camera_callback(self, msg: Image):
        """Store latest camera frame."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    def _depth_callback(self, msg: Image):
        """Store latest depth image for pedestrian distance estimation."""
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
            with self._depth_lock:
                self._latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def _frame_server(self):
        """TCP server to send frames to standalone detector."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((BRIDGE_HOST, FRAME_PORT))
        server.listen(1)
        server.settimeout(1.0)

        self.get_logger().info(f"Frame server listening on port {FRAME_PORT}")

        while self._running:
            try:
                client, addr = server.accept()
                self.get_logger().info(f"GPU detector connected from {addr}")
                self._client_connected = True

                # Send frames to connected client
                last_frame_count = -1
                frames_sent = 0
                no_frame_warned = False
                connect_time = time.time()
                while self._running:
                    try:
                        with self._frame_lock:
                            if self._latest_frame is None or self._frame_count == last_frame_count:
                                # No new frame yet — warn once if waiting too long
                                if not no_frame_warned and self._latest_frame is None and \
                                   (time.time() - connect_time) > 5.0:
                                    self.get_logger().warn(
                                        "Detector connected but no camera frames yet "
                                        f"(waiting on {self._camera_topic})")
                                    no_frame_warned = True
                                time.sleep(0.01)
                                continue
                            frame = self._latest_frame.copy()
                            last_frame_count = self._frame_count

                        # Encode frame as JPEG
                        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        data = encoded.tobytes()

                        # Send size header (4 bytes) + data
                        header = struct.pack('>I', len(data))
                        client.sendall(header + data)
                        frames_sent += 1

                    except (BrokenPipeError, ConnectionResetError):
                        elapsed = time.time() - connect_time
                        self.get_logger().warn(
                            f"GPU detector disconnected after {elapsed:.1f}s "
                            f"({frames_sent} frames sent)")
                        break
                    except Exception as e:
                        self.get_logger().error(f"Frame send error: {e}")
                        break

                client.close()
                self._client_connected = False

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.get_logger().error(f"Frame server error: {e}")

        server.close()

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

    def _is_pedestrian_on_road(self, det: dict) -> bool:
        """Check if a detected pedestrian is within road boundaries (map frame).

        Uses depth image for accurate distance, bbox center for lateral position,
        vehicle TF for map transform, and road boundaries for on-road check.

        Returns True if pedestrian is on the road (should stop).
        Returns True if we can't determine (fail-safe).
        """
        if self._road_boundaries is None or not self._has_vehicle_pose:
            return True  # Can't check — assume on road (safe)

        bbox = det.get('bbox')
        distance = det.get('distance')
        if bbox is None or distance is None or distance <= 0:
            return True  # No spatial info — assume on road

        x, y, w, h = bbox
        bbox_cx = x + w / 2

        # Use depth image for more accurate distance at bbox center
        with self._depth_lock:
            if self._latest_depth is not None:
                cx_px = int(x + w / 2)
                cy_px = int(y + h / 2)
                if (0 <= cy_px < self._latest_depth.shape[0]
                        and 0 <= cx_px < self._latest_depth.shape[1]):
                    depth_val = float(self._latest_depth[cy_px, cx_px])
                    if 0.1 < depth_val < 10.0:
                        distance = depth_val

        img_w = 640
        with self._frame_lock:
            if self._latest_frame is not None:
                img_w = self._latest_frame.shape[1]

        # Lateral offset from image center, scaled by FOV and distance
        normalized = (bbox_cx - img_w / 2) / (img_w / 2)
        lateral = normalized * distance * math.tan(CAMERA_HFOV_RAD / 2)

        # Transform from vehicle frame to map frame
        cos_t = math.cos(self._vehicle_theta)
        sin_t = math.sin(self._vehicle_theta)
        ped_x_map = self._vehicle_x + distance * cos_t - lateral * sin_t
        ped_y_map = self._vehicle_y + distance * sin_t + lateral * cos_t

        # Check if pedestrian position is within any road segment
        segment = self._road_boundaries.get_active_segment(ped_x_map, ped_y_map)
        if segment is not None:
            return True

        for seg in self._road_boundaries.segments:
            if seg.contains_point(ped_x_map, ped_y_map, margin=0.15):
                return True

        return False

    def _det_to_map_position(self, det: dict) -> tuple[float, float] | None:
        """Transform a detection to map (x, y) using depth + vehicle TF."""
        if not self._has_vehicle_pose:
            return None

        bbox = det.get('bbox')
        distance = det.get('distance')
        if bbox is None or distance is None or distance <= 0:
            return None

        x, y, w, h = bbox
        bbox_cx = x + w / 2

        # Use depth for more accurate distance
        with self._depth_lock:
            if self._latest_depth is not None:
                cx_px = int(x + w / 2)
                cy_px = int(y + h / 2)
                if (0 <= cy_px < self._latest_depth.shape[0]
                        and 0 <= cx_px < self._latest_depth.shape[1]):
                    depth_val = float(self._latest_depth[cy_px, cx_px])
                    if 0.1 < depth_val < 10.0:
                        distance = depth_val

        img_w = 640
        with self._frame_lock:
            if self._latest_frame is not None:
                img_w = self._latest_frame.shape[1]

        normalized = (bbox_cx - img_w / 2) / (img_w / 2)
        lateral = normalized * distance * math.tan(CAMERA_HFOV_RAD / 2)

        cos_t = math.cos(self._vehicle_theta)
        sin_t = math.sin(self._vehicle_theta)
        x_map = self._vehicle_x + distance * cos_t - lateral * sin_t
        y_map = self._vehicle_y + distance * sin_t + lateral * cos_t
        return (x_map, y_map)

    def _update_pedestrian_tracker(self, detections: list):
        """Feed pedestrian detections into the Kalman tracker."""
        cfg = DETECTION_CONFIG
        measurements = []
        for det in detections:
            if det.get('class') != 'person':
                continue
            if det.get('confidence', 0) < cfg['pedestrian_confidence']:
                continue
            pos = self._det_to_map_position(det)
            if pos is not None:
                measurements.append(pos)

        self._ped_tracker.predict()
        self._ped_tracker.update(measurements)

    def _receive_loop(self):
        """Receive detection results from standalone YOLO detector."""
        while self._running:
            try:
                data, addr = self._result_socket.recvfrom(65536)
                msg = json.loads(data.decode())

                if 'motion_enable' in msg:
                    # Update vehicle pose for road boundary checks
                    self._update_vehicle_pose()

                    self._last_msg_time = time.time()

                    # Filter and process detections for obstacle positions only
                    # NOTE: /motion_enable and /traffic_control_state are handled
                    # by the C++ sign_detector_node. We only publish obstacle
                    # positions/info for MPCC avoidance.
                    detections = msg.get('detections', [])
                    if detections:
                        # Whitelist filter: only keep allowed classes
                        detections = [d for d in detections
                                      if d.get('class', '').lower() in
                                      {c.lower() for c in ALLOWED_CLASSES}]

                    # Update Kalman pedestrian tracker with new detections
                    if detections:
                        self._update_pedestrian_tracker(detections)

                    if detections:
                        self._process_and_publish_detections(detections)

            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                self.get_logger().warn(f"Invalid JSON: {e}")
            except Exception as e:
                self.get_logger().error(f"Receive error: {e}")

    def _process_and_publish_detections(self, detections: list):
        """Process raw detections and publish obstacle info + positions for MPCC avoidance."""
        # Publish obstacle_info (same format as obstacle_detector.py)
        info = {
            'status': 'active',
            'motion_enabled': True,
            'timestamp': time.time(),
            'detections': detections,
        }
        info_msg = String()
        info_msg.data = json.dumps(info)
        self._obstacle_info_pub.publish(info_msg)

        # NOTE: traffic control state is handled by C++ sign_detector_node.
        # We only publish obstacle positions for MPCC avoidance.

        # Process obstacle positions
        self._publish_obstacle_positions(detections)

    def _process_traffic_control(self, detections: list) -> TrafficControlState:
        """
        Process detections for traffic control state.

        Reference repo approach: confidence-only, no distance thresholds.
        High confidence = object is close enough to require action.
        Yield signs are ignored (reference repo doesn't handle them).
        """
        current_time = time.time()
        cfg = DETECTION_CONFIG
        state = TrafficControlState()

        # Check if traffic light cooldown has expired
        if self._cross_cooldown:
            if current_time - self._cross_cooldown_start >= 6.0:
                self._cross_cooldown = False

        for det in detections:
            obj_class = det.get('class', '')
            confidence = det.get('confidence', 0.0)

            # Red traffic light (custom model: 'red')
            if obj_class == 'red' and confidence >= cfg['red_light_confidence']:
                if self._cross_cooldown:
                    continue
                state.control_type = "traffic_light"
                state.light_state = "red"
                state.should_stop = True
                state.stop_duration = 0.0
                self._cross_waiting = True
                return state

            # Green traffic light (custom model: 'green')
            elif obj_class == 'green' and confidence >= cfg['green_light_confidence']:
                state.control_type = "traffic_light"
                state.light_state = "green"
                state.should_stop = False
                if self._cross_waiting:
                    self._cross_waiting = False
                    self._cross_cooldown = True
                    self._cross_cooldown_start = current_time
                    self.get_logger().info("Traffic light GREEN - proceeding")
                return state

            # COCO traffic light (fallback model)
            elif obj_class == 'traffic light':
                state.control_type = "traffic_light"
                state.light_state = "detected"
                state.should_stop = False
                return state

            # Stop sign (custom: 'stop', COCO: 'stop sign')
            elif obj_class in ('stop sign', 'stop') and confidence >= cfg['sign_confidence']:
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
                        self.get_logger().info("Stop sign wait complete - can proceed")
                        state.should_stop = False
                        state.stop_duration = 0.0
                elif self._at_stop_sign and self._stop_sign_wait_complete:
                    state.should_stop = False
                    state.stop_duration = 0.0

                return state

            # Yield sign — only yield for tracked pedestrians on road,
            # with cooldown to prevent livelock. Vehicle co-detection disabled
            # (too many false positives from own car / parked cars).
            elif 'yield' in obj_class.lower():
                if confidence < cfg['sign_confidence']:
                    continue
                yield_cooldown = cfg.get('yield_sign_cooldown', 10.0)
                last_yield = self._last_detection_time.get('yield sign', 0)
                if current_time - last_yield < yield_cooldown:
                    continue  # In cooldown
                ped_on_road, _ = self._ped_tracker.any_on_road()
                if ped_on_road:
                    self._last_detection_time['yield sign'] = current_time
                    state.control_type = "yield_sign"
                    state.should_stop = True
                    state.stop_duration = 0.0
                    self.get_logger().info("Yield: pedestrian on road - stopping")
                    return state

        # No traffic control detected
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
                    state.should_stop = False

        return state

    def _publish_obstacle_positions(self, detections: list):
        """Publish obstacle positions for MPCC controller."""
        obstacles = ObstaclePositions()
        obstacles.timestamp = time.time()

        for det in detections:
            obj_class = det.get('class', '')
            distance = det.get('distance')
            bbox = det.get('bbox')
            confidence = det.get('confidence', 0.0)

            if obj_class not in ['person', 'car', 'traffic cone', 'sports ball']:
                continue

            if distance is None:
                continue

            map_pos = self._det_to_map_position(det)
            if map_pos is None:
                continue

            obs = ObstaclePosition()
            obs.obj_class = obj_class
            obs.x = map_pos[0]
            obs.y = map_pos[1]
            obs.frame = "map"

            if obj_class == 'person':
                obs.radius = 0.3
            elif obj_class == 'car':
                obs.radius = 0.5
            else:
                obs.radius = 0.15

            obstacles.add_obstacle(obs)

        msg = String()
        msg.data = obstacles.to_json()
        self._obstacle_positions_pub.publish(msg)

    def _heartbeat(self):
        """Publish bridge status."""
        connected = (time.time() - self._last_msg_time) < 2.0

        msg = String()
        msg.data = json.dumps({
            'connected': connected,
            'client_connected': self._client_connected,
            'last_msg_age': time.time() - self._last_msg_time if self._last_msg_time > 0 else -1,
            'frames_received': self._frame_count,
        })
        self._info_pub.publish(msg)

        # Periodic status logging
        if self._frame_count > 0 and self._frame_count % 30 == 0:
            status = f"Camera: {self._frame_count} frames | Detector: {'connected' if self._client_connected else 'waiting'}"
            self.get_logger().info(status)
        elif self._frame_count == 0:
            self.get_logger().warn(f"No camera frames received yet on {self._camera_topic}")

        if not connected and self._last_msg_time > 0:
            # Throttle: only log every 10s instead of every 1s heartbeat
            if not hasattr(self, '_last_disconnect_log') or \
               (time.time() - self._last_disconnect_log) >= 10.0:
                age = time.time() - self._last_msg_time
                self.get_logger().warn(
                    f"YOLO detector disconnected (last msg {age:.0f}s ago)")
                self._last_disconnect_log = time.time()

    def destroy_node(self):
        self._running = False
        self._result_socket.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
