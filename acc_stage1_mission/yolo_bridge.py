#!/usr/bin/env python3
"""
YOLO Bridge Node - Bidirectional bridge between ROS2 and standalone Python 3.10 YOLO detector.

This node:
1. Subscribes to camera images from QLabs and sends them to the GPU detector
2. Receives detection results from the GPU detector and publishes to ROS2 topics:
   - /motion_enable (Bool): stop/go signal
   - /traffic_control_state (String): JSON-encoded traffic control info for MPCC
   - /obstacle_positions (String): JSON-encoded obstacle positions for MPCC
   - /obstacle_info (String): JSON detection details

Usage:
    ros2 run acc_stage1_mission yolo_bridge
"""

import json
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from acc_stage1_mission.traffic_control_state import (
    TrafficControlState,
    ObstaclePosition,
    ObstaclePositions,
)

# Whitelist of allowed detection classes - all others are discarded
ALLOWED_CLASSES = {
    'person', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'sports ball',
}

# Socket configuration
BRIDGE_HOST = '0.0.0.0'
RESULT_PORT = 9999   # Receive detection results
FRAME_PORT = 9998    # Send camera frames

# Detection config (mirrors standalone detector for traffic control processing)
DETECTION_CONFIG = {
    'stop_sign_pause': 3.0,
    'yield_sign_pause': 1.5,
    'sign_stop_distance': 0.9,
    'sign_confidence': 0.5,
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
        self._motion_pub = self.create_publisher(Bool, '/motion_enable', 10)
        self._info_pub = self.create_publisher(String, '/yolo_bridge/status', 10)
        self._obstacle_info_pub = self.create_publisher(String, '/obstacle_info', 10)
        self._traffic_control_pub = self.create_publisher(
            String, '/traffic_control_state', 10)
        self._obstacle_positions_pub = self.create_publisher(
            String, '/obstacle_positions', 10)

        # Subscribe to camera
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_callback, qos)

        # UDP socket to receive results from standalone detector
        self._result_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._result_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._result_socket.bind((BRIDGE_HOST, RESULT_PORT))
        self._result_socket.settimeout(0.1)

        # TCP socket to send frames to standalone detector
        self._frame_socket = None
        self._client_connected = False

        # Traffic control state tracking
        self._at_stop_sign = False
        self._stop_sign_start_time = None
        self._stop_sign_wait_complete = False
        self._last_traffic_light_state = "unknown"
        self._traffic_light_transition_time = None

        self.get_logger().info("=" * 50)
        self.get_logger().info("YOLO Bridge Starting (enhanced)")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Result listener:  UDP port {RESULT_PORT}")
        self.get_logger().info(f"Frame server:     TCP port {FRAME_PORT}")
        self.get_logger().info(f"Camera topic:     {self._camera_topic}")
        self.get_logger().info("")
        self.get_logger().info("Publishing to:")
        self.get_logger().info("  /motion_enable, /traffic_control_state,")
        self.get_logger().info("  /obstacle_positions, /obstacle_info")
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
                while self._running:
                    try:
                        with self._frame_lock:
                            if self._latest_frame is None or self._frame_count == last_frame_count:
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

                    except (BrokenPipeError, ConnectionResetError):
                        self.get_logger().warn("GPU detector disconnected")
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

    def _receive_loop(self):
        """Receive detection results from standalone YOLO detector."""
        while self._running:
            try:
                data, addr = self._result_socket.recvfrom(65536)
                msg = json.loads(data.decode())

                if 'motion_enable' in msg:
                    self._publish_motion(msg['motion_enable'])
                    self._last_msg_time = time.time()

                    # Log detections if present
                    if 'reason' in msg and msg['reason']:
                        self.get_logger().info(f"Detection: {msg['reason']}")

                    # Filter and process detections for traffic control and obstacle positions
                    detections = msg.get('detections', [])
                    if detections:
                        # Whitelist filter: only keep allowed classes
                        detections = [d for d in detections
                                      if d.get('class', '').lower() in
                                      {c.lower() for c in ALLOWED_CLASSES}]
                        self._process_and_publish_detections(detections)

            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                self.get_logger().warn(f"Invalid JSON: {e}")
            except Exception as e:
                self.get_logger().error(f"Receive error: {e}")

    def _process_and_publish_detections(self, detections: list):
        """Process raw detections and publish traffic control state + obstacle positions."""
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

        # Process traffic control state
        traffic_state = self._process_traffic_control(detections)
        tc_msg = String()
        tc_msg.data = traffic_state.to_json()
        self._traffic_control_pub.publish(tc_msg)

        # Process obstacle positions
        self._publish_obstacle_positions(detections)

    def _process_traffic_control(self, detections: list) -> TrafficControlState:
        """
        Process detections for traffic control state.
        Mirrors the logic from obstacle_detector.py.
        """
        current_time = time.time()
        cfg = DETECTION_CONFIG
        state = TrafficControlState()

        # Check for traffic lights first (by analyzing detection classes)
        has_traffic_light = False
        has_stop_sign = False
        has_yield_sign = False

        for det in detections:
            obj_class = det.get('class', '')
            distance = det.get('distance')
            confidence = det.get('confidence', 0.0)

            # Traffic light detection (YOLO detects "traffic light")
            if obj_class == 'traffic light':
                has_traffic_light = True
                state.control_type = "traffic_light"
                state.distance = distance if distance else 1.0
                # YOLO doesn't directly tell us red vs green;
                # the stop decision from standalone detector handles this.
                # If motion was disabled, treat as red. Otherwise green.
                # This is a simplification - color analysis is done in standalone.
                state.light_state = "detected"
                state.should_stop = False  # Let motion_enable handle the stop
                return state

            # Stop sign detection
            elif obj_class == 'stop sign':
                has_stop_sign = True
                if confidence < cfg['sign_confidence']:
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

            # Yield sign (may be detected as generic sign or not at all by YOLO)
            elif 'yield' in obj_class.lower():
                has_yield_sign = True
                state.control_type = "yield_sign"
                state.distance = distance if distance else 0.5
                if distance is not None and distance < cfg['sign_stop_distance']:
                    state.should_stop = True
                    state.stop_duration = cfg['yield_sign_pause']
                else:
                    state.should_stop = False
                return state

        # No traffic control detected
        state.control_type = "none"
        state.should_stop = False

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

            obs = ObstaclePosition()
            obs.obj_class = obj_class
            obs.x = distance  # Forward distance
            obs.y = 0.0

            if obj_class == 'person':
                obs.radius = 0.3
            elif obj_class == 'car':
                obs.radius = 0.5
            else:
                obs.radius = 0.15

            # Estimate lateral offset from bbox if available
            if bbox is not None and self._latest_frame is not None:
                x, y, w, h = bbox
                with self._frame_lock:
                    if self._latest_frame is not None:
                        img_w = self._latest_frame.shape[1]
                        bbox_center_x = x + w / 2
                        lateral_offset = (bbox_center_x - img_w / 2) / (img_w / 2)
                        obs.y = lateral_offset * distance * 0.5

            obstacles.add_obstacle(obs)

        msg = String()
        msg.data = obstacles.to_json()
        self._obstacle_positions_pub.publish(msg)

    def _publish_motion(self, enabled: bool):
        """Publish motion enable message."""
        msg = Bool()
        msg.data = enabled
        self._motion_pub.publish(msg)

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
            self.get_logger().warn("YOLO detector disconnected - enabling motion")
            self._publish_motion(True)
            # Publish clear traffic control state
            clear_state = TrafficControlState()
            tc_msg = String()
            tc_msg.data = clear_state.to_json()
            self._traffic_control_pub.publish(tc_msg)

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
