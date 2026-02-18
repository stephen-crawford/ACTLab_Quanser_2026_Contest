#!/usr/bin/env python3
"""
Utility node to log current robot pose from TF.

Use this to find coordinates for mission.yaml:
1. Launch Nav2: ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
2. Drive/navigate the QCar2 to a location (pickup, dropoff, hub)
3. Run: ros2 run acc_stage1_mission pose_logger
4. Copy the logged (x, y, yaw) values to mission.yaml
"""
import math
import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rad) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseLogger(Node):
    def __init__(self):
        super().__init__("pose_logger")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.timer = self.create_timer(1.0, self.log_pose)
        self.get_logger().info("PoseLogger started - will log map->base_link every 1s")
        self.get_logger().info("Navigate the QCar2 to a location, then copy the coordinates below:")

    def log_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            yaw_deg = math.degrees(yaw)

            self.get_logger().info(
                f"  x: {x:.3f}\n  y: {y:.3f}\n  yaw: {yaw:.3f}  # ({yaw_deg:.1f} degrees)"
            )
        except TransformException as e:
            self.get_logger().warn(f"TF not available: {e}")


def main():
    rclpy.init()
    node = PoseLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
