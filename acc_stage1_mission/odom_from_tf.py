#!/usr/bin/env python3
"""
Odom From TF Publisher

This node converts TF transforms (odom -> base_link) to Odometry messages
on the /odom topic. It also broadcasts TF frames when Cartographer doesn't
provide them (creating the odom frame so Nav2's local_costmap works).

This is necessary because:
1. Cartographer may or may not provide odom frame depending on config
2. Nav2's local_costmap REQUIRES the odom frame TF to exist
3. AMCL, Nav2 controller, and MPCC controller expect /odom topic

Usage:
    ros2 run acc_stage1_mission odom_from_tf
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformException


class OdomFromTF(Node):
    """
    Publishes Odometry messages by listening to TF transforms.
    Also broadcasts TF if needed to create the odom frame.

    Cartographer with provide_odom_frame=true publishes:
    - map -> odom (localization correction)
    - odom -> base_link (odometry estimate)

    If Cartographer only provides map -> base_link, this node will:
    - Broadcast map -> odom (identity) and odom -> base_link
    - Publish /odom topic from odom -> base_link

    This ensures Nav2's local_costmap always has the odom frame it needs.
    """

    def __init__(self):
        super().__init__('odom_from_tf')

        # Parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('broadcast_tf', True)  # Broadcast TF if odom frame missing

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        publish_rate = self.get_parameter('publish_rate').value
        self._broadcast_tf = self.get_parameter('broadcast_tf').value

        # TF listener and broadcaster
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publisher - use BEST_EFFORT to match typical odom QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._odom_pub = self.create_publisher(Odometry, '/odom', qos)

        # State for velocity estimation
        self._last_x = None
        self._last_y = None
        self._last_theta = None
        self._last_time = None

        # Track which TF source is being used
        self._using_map_frame = False
        self._tf_source_logged = False
        self._odom_frame_exists = False

        # Timer for publishing
        period = 1.0 / publish_rate
        self._timer = self.create_timer(period, self._publish_odom)

        self.get_logger().info(f"OdomFromTF started: {self.odom_frame} -> {self.base_frame} -> /odom @ {publish_rate}Hz")
        self.get_logger().info(f"  Fallback: {self.map_frame} -> {self.base_frame} if odom frame unavailable")
        if self._broadcast_tf:
            self.get_logger().info(f"  Will broadcast TF to create odom frame if needed")

    def _lookup_transform(self):
        """
        Try to look up TF transform, with fallback options.
        Returns (transform, frame_id, needs_tf_broadcast) or (None, None, False) if unavailable.
        """
        # Try primary source: odom -> base_link
        try:
            t = self._tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.05)
            )
            if self._using_map_frame:
                self.get_logger().info(f"Switched to native {self.odom_frame} -> {self.base_frame} TF (Cartographer providing odom)")
            self._using_map_frame = False
            self._odom_frame_exists = True
            return t, self.odom_frame, False  # No TF broadcast needed
        except TransformException:
            pass

        # Fallback: map -> base_link (Cartographer without odom frame)
        try:
            t = self._tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.05)
            )
            if not self._using_map_frame:
                self.get_logger().info(f"Using fallback {self.map_frame} -> {self.base_frame} TF")
                self.get_logger().info(f"  Will broadcast {self.map_frame} -> {self.odom_frame} and {self.odom_frame} -> {self.base_frame}")
            self._using_map_frame = True
            self._odom_frame_exists = False
            return t, self.odom_frame, True  # Need to broadcast TF to create odom frame
        except TransformException:
            pass

        return None, None, False

    def _broadcast_odom_tf(self, map_to_base: TransformStamped):
        """
        Broadcast TF frames to create odom frame when Cartographer doesn't provide it.

        We create:
        - map -> odom (identity transform - odom frame at same location as map origin)
        - odom -> base_link (same as map -> base_link since odom=map)

        This way Nav2's local_costmap gets the odom frame it needs.
        """
        now = self.get_clock().now().to_msg()

        # Broadcast map -> odom (identity - odom frame is at map origin)
        # This is a simplification that works when we don't have wheel odometry
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = now
        map_to_odom.header.frame_id = self.map_frame
        map_to_odom.child_frame_id = self.odom_frame
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.x = 0.0
        map_to_odom.transform.rotation.y = 0.0
        map_to_odom.transform.rotation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0

        # Broadcast odom -> base_link (same as map -> base_link since map=odom)
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = now
        odom_to_base.header.frame_id = self.odom_frame
        odom_to_base.child_frame_id = self.base_frame
        odom_to_base.transform = map_to_base.transform

        self._tf_broadcaster.sendTransform([map_to_odom, odom_to_base])

    def _publish_odom(self):
        """Look up TF, broadcast TF if needed, and publish Odometry message."""
        t, frame_id, needs_tf_broadcast = self._lookup_transform()

        if t is None:
            # Only warn occasionally to avoid spam
            if not hasattr(self, '_last_warn_time'):
                self._last_warn_time = 0
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_warn_time > 5.0:
                self.get_logger().warn(
                    f"TF not available: tried {self.odom_frame}->{self.base_frame} "
                    f"and {self.map_frame}->{self.base_frame}"
                )
                self._last_warn_time = current_time
            return

        # Broadcast TF to create odom frame if Cartographer doesn't provide it
        if needs_tf_broadcast and self._broadcast_tf:
            self._broadcast_odom_tf(t)

        # Extract position
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z

        # Extract orientation (quaternion)
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w

        # Extract yaw for velocity calculation
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        theta = np.arctan2(siny_cosp, cosy_cosp)

        # Estimate velocity from position change
        current_time = self.get_clock().now().nanoseconds / 1e9
        vx = 0.0
        vy = 0.0
        vtheta = 0.0

        if self._last_time is not None and self._last_x is not None:
            dt = current_time - self._last_time
            if dt > 0.001:  # Avoid division by very small dt
                # World frame velocities
                dx = x - self._last_x
                dy = y - self._last_y

                # Convert to body frame velocity (forward velocity)
                # vx_body = dx*cos(theta) + dy*sin(theta)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                vx = (dx * cos_t + dy * sin_t) / dt
                vy = (-dx * sin_t + dy * cos_t) / dt

                # Angular velocity
                dtheta = theta - self._last_theta
                # Normalize angle difference to [-pi, pi]
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi
                vtheta = dtheta / dt

        # Store for next iteration
        self._last_x = x
        self._last_y = y
        self._last_theta = theta
        self._last_time = current_time

        # Create and publish Odometry message
        # Use the actual frame_id so consumers know which frame the data is in
        # When using map->base_link fallback, frame_id will be 'map'
        odom_msg = Odometry()
        odom_msg.header.stamp = t.header.stamp
        odom_msg.header.frame_id = frame_id  # 'odom' or 'map' depending on TF source
        odom_msg.child_frame_id = self.base_frame

        # Position
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Velocity (in body frame)
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.angular.z = vtheta

        # Covariance (small values indicating relatively good estimates)
        # Position covariance [x, y, z, roll, pitch, yaw] - 6x6 matrix flattened
        pose_cov = [0.01] * 36
        pose_cov[0] = 0.01   # x
        pose_cov[7] = 0.01   # y
        pose_cov[35] = 0.01  # yaw
        odom_msg.pose.covariance = pose_cov

        # Velocity covariance
        twist_cov = [0.01] * 36
        twist_cov[0] = 0.1   # vx - higher uncertainty
        twist_cov[35] = 0.1  # vyaw
        odom_msg.twist.covariance = twist_cov

        self._odom_pub.publish(odom_msg)


def main():
    rclpy.init()
    node = OdomFromTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
