#!/usr/bin/env python3
"""
ROS2 Data Capture Node for YOLO Training

Captures camera frames from the QLabs simulation for annotation and training.

Usage:
    # Manual capture (press Enter to save frame):
    ros2 run acc_stage1_mission capture_data

    # Auto capture every 0.5 seconds while driving:
    ros2 run acc_stage1_mission capture_data --ros-args -p auto:=true -p interval:=0.5

After capturing:
    1. Upload images from training/images/ to Roboflow or CVAT
    2. Annotate with 9 classes: cone, green_light, red_light, yellow_light,
       stop_sign, yield_sign, round_sign, person, car
    3. Export in YOLO format to training/dataset/
    4. Run: python3 training/train.py
"""

import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CaptureData(Node):
    def __init__(self):
        super().__init__('capture_data')

        self.declare_parameter('camera_topic', '/camera/color_image')
        self.declare_parameter('auto', False)
        self.declare_parameter('interval', 0.5)
        self.declare_parameter('output_dir', '')

        self._camera_topic = self.get_parameter('camera_topic').value
        self._auto_mode = self.get_parameter('auto').value
        self._interval = self.get_parameter('interval').value
        output_dir = self.get_parameter('output_dir').value

        # Resolve output directory
        if not output_dir:
            # Default: training/images/ relative to package source
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(pkg_dir, 'training', 'images')
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._latest_frame = None
        self._frame_count = 0
        self._save_count = 0
        self._last_save_time = 0.0

        # Subscribe to camera
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Image, self._camera_topic, self._camera_callback, qos)

        if self._auto_mode:
            self._timer = self.create_timer(self._interval, self._auto_capture)
            self.get_logger().info(
                f"Auto capture mode: saving every {self._interval}s to {self._output_dir}")
        else:
            self._timer = self.create_timer(0.1, self._manual_capture_prompt)
            self._prompted = False
            self.get_logger().info(
                f"Manual capture mode: press Enter to save frame to {self._output_dir}")

        self.get_logger().info(f"Subscribing to {self._camera_topic}")
        self.get_logger().info(f"Output directory: {self._output_dir}")

    def _camera_callback(self, msg: Image):
        try:
            self._latest_frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Frame conversion failed: {e}")

    def _save_frame(self):
        if self._latest_frame is None:
            self.get_logger().warn("No frame available yet")
            return False

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        ms = int((time.time() % 1) * 1000)
        filename = f"frame_{timestamp}_{ms:03d}.jpg"
        filepath = os.path.join(self._output_dir, filename)

        cv2.imwrite(filepath, self._latest_frame)
        self._save_count += 1
        self.get_logger().info(
            f"Saved [{self._save_count}]: {filename} "
            f"({self._latest_frame.shape[1]}x{self._latest_frame.shape[0]})")
        return True

    def _auto_capture(self):
        if self._latest_frame is not None:
            self._save_frame()

    def _manual_capture_prompt(self):
        if not self._prompted and self._frame_count > 0:
            self._prompted = True
            self.get_logger().info("Camera connected. Press Enter in terminal to capture frames.")
            self.get_logger().info("Press Ctrl+C to stop.")

        # Non-blocking check for Enter key (via timer, user interaction via terminal)
        import select
        import sys
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            self._save_frame()


def main():
    rclpy.init()
    node = CaptureData()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Capture complete. Saved {node._save_count} frames to {node._output_dir}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
