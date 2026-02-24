#!/usr/bin/env python3
"""
Real-Time Telemetry Dashboard for MPCC Controller

Live-updating matplotlib window showing velocity, cross-track error,
heading error, and mission status during execution.

Subscribes to /mpcc/telemetry (JSON String) published by the MPCC controller.

Layout (2x2 matplotlib figure):
┌────────────────────────────┬────────────────────────────┐
│   Velocity (m/s) vs Time   │  Cross-Track Error vs Time │
│  ─── v_reference (blue)    │  ─── cross_track (red)     │
│  ─── v_measured (green)    │  --- ±0.15m bounds (gray)  │
│  ─── v_command (orange)    │                            │
├────────────────────────────┼────────────────────────────┤
│   Heading Error vs Time    │      Status Panel          │
│  ─── heading_err (purple)  │  Progress: ██████░░ 72%    │
│  --- ±15° bounds (gray)    │  Leg: hub_to_pickup        │
│                            │  State: tracking           │
└────────────────────────────┴────────────────────────────┘

Usage:
    ros2 run acc_stage1_mission dashboard
"""

import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


WINDOW_SIZE = 300  # 15 seconds at 20Hz


class DashboardNode(Node):
    """Real-time telemetry dashboard for MPCC controller."""

    def __init__(self):
        super().__init__('dashboard')

        # Ring buffers
        self._timestamps = deque(maxlen=WINDOW_SIZE)
        self._v_ref = deque(maxlen=WINDOW_SIZE)
        self._v_measured = deque(maxlen=WINDOW_SIZE)
        self._v_command = deque(maxlen=WINDOW_SIZE)
        self._cross_track = deque(maxlen=WINDOW_SIZE)
        self._heading_err = deque(maxlen=WINDOW_SIZE)

        # Latest status
        self._progress_pct = 0.0
        self._mission_leg = ""
        self._state = ""
        self._obstacles = 0
        self._steering_angle = 0.0

        # Subscribe to telemetry
        self.create_subscription(
            String, '/mpcc/telemetry', self._telemetry_callback, 10)

        self.get_logger().info("Dashboard initialized, waiting for /mpcc/telemetry...")

    def _telemetry_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        t = data.get('timestamp', 0.0)
        self._timestamps.append(t)
        self._v_ref.append(data.get('v_reference', 0.0))
        self._v_measured.append(data.get('v_measured', 0.0))
        self._v_command.append(data.get('v_command', 0.0))
        self._cross_track.append(data.get('cross_track_error', 0.0))
        self._heading_err.append(math.degrees(data.get('heading_error', 0.0)))

        self._progress_pct = data.get('progress_pct', 0.0)
        self._mission_leg = data.get('mission_leg', '')
        self._state = data.get('state', '')
        self._obstacles = data.get('obstacles', 0)
        self._steering_angle = math.degrees(data.get('steering_angle', 0.0))


def main():
    rclpy.init()
    node = DashboardNode()

    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('MPCC Telemetry Dashboard', fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('#1e1e1e')
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    ax_vel = axes[0, 0]
    ax_ct = axes[0, 1]
    ax_he = axes[1, 0]
    ax_status = axes[1, 1]

    for ax in [ax_vel, ax_ct, ax_he]:
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.grid(True, alpha=0.3, color='#555555')

    ax_status.set_facecolor('#2d2d2d')
    ax_status.axis('off')

    # Velocity plot setup
    ax_vel.set_title('Velocity (m/s)')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('m/s')
    line_vref, = ax_vel.plot([], [], 'b-', label='v_ref', linewidth=1.5)
    line_vmeas, = ax_vel.plot([], [], 'g-', label='v_measured', linewidth=1.5)
    line_vcmd, = ax_vel.plot([], [], color='orange', linestyle='-',
                              label='v_command', linewidth=1.5)
    ax_vel.legend(loc='upper left', fontsize=8, facecolor='#2d2d2d',
                  edgecolor='#555555', labelcolor='white')
    ax_vel.set_ylim(-0.05, 0.8)

    # Cross-track error plot setup
    ax_ct.set_title('Cross-Track Error (m)')
    ax_ct.set_xlabel('Time (s)')
    ax_ct.set_ylabel('meters')
    line_ct, = ax_ct.plot([], [], 'r-', label='cross_track', linewidth=1.5)
    ax_ct.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, label='±0.15m')
    ax_ct.axhline(y=-0.15, color='gray', linestyle='--', alpha=0.5)
    ax_ct.fill_between([], 0.15, 0.3, alpha=0.1, color='red')
    ax_ct.legend(loc='upper left', fontsize=8, facecolor='#2d2d2d',
                 edgecolor='#555555', labelcolor='white')
    ax_ct.set_ylim(-0.3, 0.3)

    # Heading error plot setup
    ax_he.set_title('Heading Error (deg)')
    ax_he.set_xlabel('Time (s)')
    ax_he.set_ylabel('degrees')
    line_he, = ax_he.plot([], [], color='purple', linestyle='-',
                           label='heading_err', linewidth=1.5)
    ax_he.axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='±15°')
    ax_he.axhline(y=-15, color='gray', linestyle='--', alpha=0.5)
    ax_he.legend(loc='upper left', fontsize=8, facecolor='#2d2d2d',
                 edgecolor='#555555', labelcolor='white')
    ax_he.set_ylim(-30, 30)

    def update_plots(frame):
        # Spin ROS to process callbacks
        rclpy.spin_once(node, timeout_sec=0.01)

        if len(node._timestamps) < 2:
            return line_vref, line_vmeas, line_vcmd, line_ct, line_he

        # Relative timestamps (seconds from first)
        t0 = node._timestamps[0]
        times = [t - t0 for t in node._timestamps]

        # Update velocity plot
        line_vref.set_data(times, list(node._v_ref))
        line_vmeas.set_data(times, list(node._v_measured))
        line_vcmd.set_data(times, list(node._v_command))
        ax_vel.set_xlim(times[0], max(times[-1], times[0] + 1))

        # Update cross-track plot
        line_ct.set_data(times, list(node._cross_track))
        ax_ct.set_xlim(times[0], max(times[-1], times[0] + 1))

        # Update heading error plot
        line_he.set_data(times, list(node._heading_err))
        ax_he.set_xlim(times[0], max(times[-1], times[0] + 1))

        # Update status panel
        ax_status.clear()
        ax_status.set_facecolor('#2d2d2d')
        ax_status.axis('off')

        # Progress bar
        pct = node._progress_pct
        bar_len = 20
        filled = int(bar_len * pct / 100.0)
        bar = '\u2588' * filled + '\u2591' * (bar_len - filled)

        status_lines = [
            f"Progress: {bar} {pct:.0f}%",
            f"Leg: {node._mission_leg}",
            f"State: {node._state}",
            f"Obstacles: {node._obstacles}",
            f"Steering: {node._steering_angle:.1f}\u00b0",
        ]

        if node._v_measured:
            status_lines.append(f"Speed: {node._v_measured[-1]:.3f} m/s")
        if node._cross_track:
            status_lines.append(f"CT Error: {node._cross_track[-1]:.3f} m")

        text = '\n'.join(status_lines)
        ax_status.text(0.1, 0.85, text,
                       transform=ax_status.transAxes,
                       fontsize=12, fontfamily='monospace',
                       verticalalignment='top',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.5',
                                 facecolor='#3d3d3d', edgecolor='#555555'))

        return line_vref, line_vmeas, line_vcmd, line_ct, line_he

    ani = animation.FuncAnimation(  # noqa: F841
        fig, update_plots, interval=100, blit=False, cache_frame_data=False)

    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
