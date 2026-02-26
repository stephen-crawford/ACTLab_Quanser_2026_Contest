#!/usr/bin/env python3
"""
Real-Time Path Overlay Visualizer

Draws the planned path (from /plan) and vehicle position (from TF) on top
of a bird's-eye view of the SDCS competition track. Helps diagnose whether
path following issues are caused by bad planning or bad controller tracking.

Shows:
- Road graph network (gray lines)
- Road boundary centerlines (from road_boundaries.yaml)
- Mission waypoints (Hub, Pickup, Dropoff)
- Traffic controls (stop signs, traffic lights)
- Planned path from /plan topic (cyan, updated live)
- Vehicle position + heading (red arrow, updated live)
- Vehicle trail (fading red dots)

All coordinates displayed in QLabs world frame.

Usage:
    ros2 run acc_stage1_mission path_overlay
"""

import math
import os
from collections import deque

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException

import matplotlib

def _select_backend():
    """
    Pick a matplotlib backend that can actually display a window.

    Tests both X11 socket reachability AND X11 authorization (xhost).
    Falls back to Agg if either check fails.
    """
    import os
    import socket as _sock
    import subprocess

    display = os.environ.get('DISPLAY', '')
    if not display:
        return 'Agg'

    def _x11_authorized():
        """Check X11 reachability AND authorization via xdpyinfo."""
        try:
            result = subprocess.run(
                ['xdpyinfo'], capture_output=True, timeout=3,
                env={**os.environ, 'DISPLAY': display})
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # xdpyinfo not available — try python-xlib or Qt probe
        try:
            from PyQt5.QtWidgets import QApplication
            import sys
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv[:1] or [''])
            # If we got here without crashing, Qt connected to X11
            return True
        except Exception:
            pass
        # Last resort: socket-level check (can't verify auth, but better than nothing)
        try:
            parts = display.split(':')
            disp_num = int(parts[1].split('.')[0])
            sock_path = f'/tmp/.X11-unix/X{disp_num}'
            if os.path.exists(sock_path):
                s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
                s.settimeout(1.0)
                try:
                    s.connect(sock_path)
                    s.close()
                    return True
                except Exception:
                    s.close()
        except Exception:
            pass
        return False

    if not _x11_authorized():
        return 'Agg'

    for backend, probe in [
        ('Qt5Agg', 'PyQt5.QtWidgets'),
        ('GTK3Agg', 'gi'),
        ('TkAgg', 'tkinter'),
    ]:
        try:
            __import__(probe)
            return backend
        except ImportError:
            continue
    return 'Agg'

_chosen_backend = _select_backend()
matplotlib.use(_chosen_backend)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from acc_stage1_mission.road_graph import SDCSRoadMap


# QLabs coordinates for mission locations
HUB = (-1.205, -0.83)
PICKUP = (0.125, 4.395)
DROPOFF = (-0.905, 0.800)

# Transform parameters (map frame -> QLabs)
ORIGIN_X = -1.205
ORIGIN_Y = -0.83
ORIGIN_HEADING_RAD = 0.7177


def map_to_qlabs(map_x, map_y):
    """Transform map frame coordinates to QLabs world frame."""
    theta = ORIGIN_HEADING_RAD
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Inverse rotation: R(-theta)
    qlabs_x = map_x * cos_t + map_y * sin_t + ORIGIN_X
    qlabs_y = -map_x * sin_t + map_y * cos_t + ORIGIN_Y
    return qlabs_x, qlabs_y


def map_yaw_to_qlabs_yaw(map_yaw):
    """Transform a heading from map frame to QLabs frame."""
    return map_yaw - ORIGIN_HEADING_RAD


class PathOverlayNode(Node):
    """ROS2 node that subscribes to /plan and TF for live visualization."""

    def __init__(self):
        super().__init__('path_overlay')

        # Planned path (QLabs coordinates)
        self.planned_path_x = []
        self.planned_path_y = []

        # Vehicle position trail (QLabs)
        self.trail_x = deque(maxlen=500)
        self.trail_y = deque(maxlen=500)

        # Current vehicle pose (QLabs)
        self.vehicle_x = None
        self.vehicle_y = None
        self.vehicle_yaw = None

        # Mission leg info
        self.mission_status = ""

        # TF diagnostics
        self._has_tf = False
        self._tf_fail_count = 0

        # Subscribe to /plan with volatile durability (NOT transient_local).
        # transient_local causes QoS DURABILITY incompatibility with Nav2's
        # planner which also publishes on /plan with volatile durability.
        # Mission manager republishes every 2s, so latching isn't needed.
        plan_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Path, '/plan', self._plan_callback, plan_qos)

        # Subscribe to mission status
        self.create_subscription(
            String, '/mission/status', self._status_callback, 10)

        # TF for vehicle position
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Timer to poll TF at 10 Hz
        self.create_timer(0.1, self._update_vehicle_pose)

        self.get_logger().info("Path overlay initialized, waiting for /plan and TF...")

    def _plan_callback(self, msg: Path):
        """Receive planned path from mission manager and convert to QLabs."""
        xs = []
        ys = []
        # Also collect raw map-frame coords for validation
        map_xs = []
        map_ys = []
        for pose in msg.poses:
            mx = pose.pose.position.x
            my = pose.pose.position.y
            map_xs.append(mx)
            map_ys.append(my)
            qx, qy = map_to_qlabs(mx, my)
            xs.append(qx)
            ys.append(qy)
        self.planned_path_x = xs
        self.planned_path_y = ys

        if not xs:
            return

        # Diagnostic: bounding box in QLabs frame
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        span_x = x_max - x_min
        span_y = y_max - y_min

        # Diagnostic: bounding box in map frame
        mx_min, mx_max = min(map_xs), max(map_xs)
        my_min, my_max = min(map_ys), max(map_ys)
        map_span_x = mx_max - mx_min
        map_span_y = my_max - my_min

        # Check for large jumps
        max_step = 0.0
        n_jumps = 0
        for i in range(1, len(xs)):
            step = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            if step > max_step:
                max_step = step
            if step > 0.1:
                n_jumps += 1

        self.get_logger().info(
            "Path: %d poses, qlabs_bbox=(%.2f..%.2f, %.2f..%.2f), "
            "map_bbox=(%.2f..%.2f, %.2f..%.2f), max_step=%.3fm, jumps=%d" % (
                len(msg.poses), x_min, x_max, y_min, y_max,
                mx_min, mx_max, my_min, my_max, max_step, n_jumps))

        # Detect common coordinate transform bugs
        # Track spans 3.5m x 6m in QLabs. A correct path should not span
        # much more than that. If it does, the C++ qlabs_to_map transform
        # is likely using the wrong rotation direction.
        if span_x > 5.0 or span_y > 8.0:
            self.get_logger().error(
                "PATH TRANSFORM ERROR: QLabs bbox spans %.1f x %.1fm "
                "(expected <3.5 x 6m). The C++ mission_manager likely needs "
                "to be REBUILT: colcon build --packages-select "
                "acc_mpcc_controller_cpp" % (span_x, span_y))
        if map_span_x > 8.0 or map_span_y > 10.0:
            self.get_logger().error(
                "MAP FRAME PATH ERROR: spans %.1f x %.1fm. Check "
                "coordinate_transform.h qlabs_to_map uses R(+theta), "
                "not R(-theta)" % (map_span_x, map_span_y))

    def _status_callback(self, msg: String):
        self.mission_status = msg.data

    def _update_vehicle_pose(self):
        """Poll TF for current vehicle position."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1))
            mx = t.transform.translation.x
            my = t.transform.translation.y
            q = t.transform.rotation
            map_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            if not self._has_tf:
                self._has_tf = True
                self.get_logger().info(
                    "First TF received: map(%.3f, %.3f) -> qlabs(%.3f, %.3f)" % (
                        mx, my, *map_to_qlabs(mx, my)))
                self._tf_fail_count = 0

            qx, qy = map_to_qlabs(mx, my)
            self.vehicle_x = qx
            self.vehicle_y = qy
            self.vehicle_yaw = map_yaw_to_qlabs_yaw(map_yaw)

            # Append to trail (only if moved enough)
            if len(self.trail_x) == 0 or (
                abs(qx - self.trail_x[-1]) > 0.005 or
                abs(qy - self.trail_y[-1]) > 0.005
            ):
                self.trail_x.append(qx)
                self.trail_y.append(qy)
        except TransformException:
            self._tf_fail_count += 1
            # Log every 5 seconds (50 ticks at 10Hz) so the user knows TF is down
            if self._tf_fail_count % 50 == 1:
                self.get_logger().warn(
                    "TF map->base_link unavailable (%d consecutive failures). "
                    "Vehicle position frozen. Is SLAM running?" %
                    self._tf_fail_count)


def load_road_boundaries():
    """Load road boundary centerlines from config.

    Tries multiple paths:
    1. ament_index (colcon install share directory)
    2. Relative to this source file (development)
    3. Hardcoded Docker workspace path (fallback)
    """
    config_paths = []
    # 1. ament_index — works for colcon-installed packages
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('acc_stage1_mission')
        config_paths.append(os.path.join(pkg_share, 'config', 'road_boundaries.yaml'))
    except Exception:
        pass
    # 2. Relative to source file (development layout)
    config_paths.append(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'road_boundaries.yaml'))
    # 3. Docker workspace (hardcoded fallback)
    config_paths.append(
        '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/config/road_boundaries.yaml')

    for path in config_paths:
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    return None


def draw_static_map(ax):
    """Draw the static track elements (road network, boundaries, waypoints)."""

    # --- Road graph edges (from SDCSRoadMap) ---
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    for edge in roadmap.edges:
        if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
            ax.plot(edge.waypoints[0, :], edge.waypoints[1, :],
                    color='#555555', linewidth=0.8, alpha=0.5, zorder=1)

    # Road graph nodes
    for i, node in enumerate(roadmap.nodes):
        x, y = node.pose[0, 0], node.pose[1, 0]
        ax.plot(x, y, 'o', color='#777777', markersize=3, zorder=2)
        ax.annotate(str(i), (x, y), fontsize=5, color='#999999',
                    ha='center', va='bottom', zorder=2)

    # --- Road boundary centerlines ---
    boundaries = load_road_boundaries()
    if boundaries and 'road_segments' in boundaries:
        for seg in boundaries['road_segments']:
            if 'centerline' in seg:
                pts = seg['centerline']
                xs = [p['x'] for p in pts]
                ys = [p['y'] for p in pts]
                wl = [p.get('width_left', 0.3) for p in pts]
                wr = [p.get('width_right', 0.3) for p in pts]

                # Draw centerline
                ax.plot(xs, ys, '--', color='#888888', linewidth=1.0,
                        alpha=0.6, zorder=1)

                # Draw approximate boundaries by offsetting perpendicular
                xs_arr = np.array(xs)
                ys_arr = np.array(ys)
                if len(xs_arr) >= 2:
                    dx = np.gradient(xs_arr)
                    dy = np.gradient(ys_arr)
                    norms = np.sqrt(dx**2 + dy**2)
                    norms[norms < 1e-9] = 1.0
                    nx = -dy / norms  # perpendicular
                    ny = dx / norms

                    wl_arr = np.array(wl)
                    wr_arr = np.array(wr)

                    left_x = xs_arr + nx * wl_arr
                    left_y = ys_arr + ny * wl_arr
                    right_x = xs_arr - nx * wr_arr
                    right_y = ys_arr - ny * wr_arr

                    ax.plot(left_x, left_y, '-', color='#444444',
                            linewidth=0.6, alpha=0.4, zorder=1)
                    ax.plot(right_x, right_y, '-', color='#444444',
                            linewidth=0.6, alpha=0.4, zorder=1)

            elif seg.get('type') == 'circular':
                center = seg['center']
                radius = seg['radius']
                circle = plt.Circle(
                    (center['x'], center['y']), radius,
                    fill=False, color='#444444', linewidth=0.6,
                    linestyle='--', alpha=0.4, zorder=1)
                ax.add_patch(circle)

    # --- Traffic controls ---
    if boundaries and 'traffic_controls' in boundaries:
        for ctrl in boundaries['traffic_controls']:
            px = ctrl['position']['x']
            py = ctrl['position']['y']
            if ctrl['type'] == 'traffic_light':
                ax.plot(px, py, 's', color='#FFFF00', markersize=8,
                        markeredgecolor='#888800', markeredgewidth=1,
                        zorder=5, label='Traffic Light' if ctrl == boundaries['traffic_controls'][0] else None)
            elif ctrl['type'] == 'stop_sign':
                ax.plot(px, py, '^', color='#FF4444', markersize=8,
                        markeredgecolor='#880000', markeredgewidth=1,
                        zorder=5, label='Stop Sign' if 'hub_exit' in ctrl.get('name', '') else None)

    # --- Obstacle zones ---
    if boundaries and 'obstacle_zones' in boundaries:
        for zone in boundaries['obstacle_zones']:
            cx = zone['center']['x']
            cy = zone['center']['y']
            if zone['type'] == 'circle':
                circle = plt.Circle(
                    (cx, cy), zone['radius'],
                    fill=True, color='#FF8800', alpha=0.08,
                    linewidth=0.5, linestyle=':', edgecolor='#FF8800',
                    zorder=1)
                ax.add_patch(circle)
            elif zone['type'] == 'rectangle':
                w = zone['width']
                h = zone['height']
                rect = plt.Rectangle(
                    (cx - w/2, cy - h/2), w, h,
                    fill=True, color='#FF8800', alpha=0.08,
                    linewidth=0.5, linestyle=':', edgecolor='#FF8800',
                    zorder=1)
                ax.add_patch(rect)

    # --- Mission waypoints ---
    ax.plot(*HUB, 'D', color='#FF00FF', markersize=12,
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.annotate('Hub', HUB, fontsize=9, fontweight='bold', color='#FF00FF',
                ha='left', va='bottom', xytext=(5, 5),
                textcoords='offset points', zorder=10)

    ax.plot(*PICKUP, 's', color='#00AAFF', markersize=12,
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.annotate('Pickup', PICKUP, fontsize=9, fontweight='bold', color='#00AAFF',
                ha='left', va='bottom', xytext=(5, 5),
                textcoords='offset points', zorder=10)

    ax.plot(*DROPOFF, 'p', color='#FF8800', markersize=12,
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.annotate('Dropoff', DROPOFF, fontsize=9, fontweight='bold', color='#FF8800',
                ha='left', va='bottom', xytext=(5, 5),
                textcoords='offset points', zorder=10)

    # --- Known traffic light positions (from Setup_Real_Scenario_Interleaved) ---
    tl_positions = [(0.6, 1.55), (-0.6, 1.28), (-0.37, 0.3), (0.75, 0.48)]
    for pos in tl_positions:
        ax.plot(*pos, 's', color='#FFFF00', markersize=5, alpha=0.3,
                markeredgecolor='#888800', markeredgewidth=0.5, zorder=3)

    # --- Known cone position ---
    ax.plot(-1.977, 2.784, '^', color='#FF6600', markersize=7,
            markeredgecolor='#993300', markeredgewidth=1, zorder=3)
    ax.annotate('Cone', (-1.977, 2.784), fontsize=6, color='#FF6600',
                ha='center', va='bottom', xytext=(0, 5),
                textcoords='offset points', zorder=3)


def main():
    rclpy.init()
    node = PathOverlayNode()

    active_backend = matplotlib.get_backend()
    is_interactive = active_backend.lower() != 'agg'
    node.get_logger().info("Matplotlib backend: %s (interactive=%s)" % (
        active_backend, is_interactive))
    if not is_interactive:
        import os as _os
        display = _os.environ.get('DISPLAY', '<not set>')
        node.get_logger().warn(
            "No GUI display available (DISPLAY=%s). "
            "Falling back to Agg — will save snapshots to /tmp/path_overlay.png every 2s. "
            "To enable live window: run 'xhost +local:' on host, then ensure Docker has "
            "-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY" % display)

    # --- Set up matplotlib figure ---
    # Wrap in try/except to handle X11 auth failures that only manifest at
    # window creation time (e.g. "Authorization required" when xhost not set)
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    except Exception as e:
        if is_interactive:
            node.get_logger().warn(
                "GUI backend '%s' failed: %s. Falling back to Agg (headless)." % (
                    active_backend, e))
            node.get_logger().warn(
                "Fix: run 'xhost +local:' on the host before launching Docker")
            matplotlib.use('Agg')
            is_interactive = False
            fig, ax = plt.subplots(1, 1, figsize=(10, 14))
        else:
            raise
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.set_aspect('equal')
    ax.set_title('Path Overlay - QLabs World Frame',
                 fontsize=14, fontweight='bold', color='white', pad=12)
    ax.set_xlabel('X (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.grid(True, alpha=0.15, color='#555555')

    # Draw static map elements
    draw_static_map(ax)

    # Set axis limits with margin around the track
    ax.set_xlim(-2.8, 1.5)
    ax.set_ylim(-1.8, 5.2)

    # Dynamic plot elements (updated each frame)
    line_plan, = ax.plot([], [], '-', color='#00FFFF', linewidth=2.5,
                          alpha=0.85, zorder=8, label='Planned Path')
    line_trail, = ax.plot([], [], '.', color='#FF4444', markersize=2,
                           alpha=0.4, zorder=6)
    vehicle_marker, = ax.plot([], [], 'o', color='#FF0000', markersize=8,
                               markeredgecolor='white', markeredgewidth=1.5,
                               zorder=12)
    # Vehicle heading arrow (will be recreated each frame)
    vehicle_arrow = [None]

    # Status text
    status_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        fontsize=9, fontfamily='monospace', color='white',
        verticalalignment='top', zorder=15,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                  edgecolor='#444444', alpha=0.9))

    # Legend
    legend_handles = [
        mpatches.Patch(color='#00FFFF', label='Planned Path'),
        mpatches.Patch(color='#FF4444', label='Vehicle Trail'),
        mpatches.Patch(color='#FF0000', label='Vehicle Position'),
        mpatches.Patch(color='#555555', label='Road Graph'),
        mpatches.Patch(color='#888888', label='Road Boundaries'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#444444', labelcolor='white')

    def update(frame):
        # Spin ROS to process callbacks
        rclpy.spin_once(node, timeout_sec=0.01)

        # Update planned path
        if node.planned_path_x:
            line_plan.set_data(node.planned_path_x, node.planned_path_y)
        else:
            line_plan.set_data([], [])

        # Update vehicle trail
        if node.trail_x:
            line_trail.set_data(list(node.trail_x), list(node.trail_y))

        # Update vehicle position
        if node.vehicle_x is not None:
            vehicle_marker.set_data([node.vehicle_x], [node.vehicle_y])

            # Remove old arrow
            if vehicle_arrow[0] is not None:
                vehicle_arrow[0].remove()
                vehicle_arrow[0] = None

            # Draw heading arrow
            if node.vehicle_yaw is not None:
                arrow_len = 0.15
                dx = arrow_len * math.cos(node.vehicle_yaw)
                dy = arrow_len * math.sin(node.vehicle_yaw)
                vehicle_arrow[0] = ax.annotate(
                    '', xy=(node.vehicle_x + dx, node.vehicle_y + dy),
                    xytext=(node.vehicle_x, node.vehicle_y),
                    arrowprops=dict(arrowstyle='->', color='#FF0000',
                                    lw=2.0),
                    zorder=12)

        # Update status text
        lines = []
        if node.mission_status:
            lines.append("Mission: %s" % node.mission_status)
        if not node._has_tf:
            lines.append("TF: WAITING (no map->base_link)")
        elif node.vehicle_x is not None:
            lines.append("Vehicle: (%.3f, %.3f)" % (
                node.vehicle_x, node.vehicle_y))
            if node.vehicle_yaw is not None:
                lines.append("Heading: %.1f deg" % math.degrees(node.vehicle_yaw))
        if node.planned_path_x:
            lines.append("Path: %d pts" % len(node.planned_path_x))
            # Distance from vehicle to nearest path point
            if node.vehicle_x is not None:
                path_arr = np.array(list(zip(
                    node.planned_path_x, node.planned_path_y)))
                veh = np.array([node.vehicle_x, node.vehicle_y])
                dists = np.linalg.norm(path_arr - veh, axis=1)
                min_dist = np.min(dists)
                nearest_idx = int(np.argmin(dists))
                progress_pct = 100.0 * nearest_idx / max(len(dists) - 1, 1)
                lines.append("Cross-track: %.3f m" % min_dist)
                lines.append("Progress: %.0f%%" % progress_pct)
        if len(node.trail_x) > 0:
            lines.append("Trail: %d pts" % len(node.trail_x))

        status_text.set_text('\n'.join(lines))

        return line_plan, line_trail, vehicle_marker, status_text

    plt.tight_layout()

    if is_interactive:
        ani = animation.FuncAnimation(  # noqa: F841
            fig, update, interval=100, blit=False, cache_frame_data=False)
        plt.show()
    else:
        # Agg fallback: spin ROS + save snapshot periodically
        save_path = '/tmp/path_overlay.png'
        node.get_logger().info("Saving snapshots to %s" % save_path)
        try:
            while rclpy.ok():
                update(0)
                fig.savefig(save_path, dpi=100, facecolor=fig.get_facecolor())
                for _ in range(20):  # ~2s between saves
                    rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
