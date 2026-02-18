#!/usr/bin/env bash
# =============================================================================
# Stage I Mission - DEV Container Runner
# =============================================================================
# Run inside the DEV (Isaac-ROS) container after QLabs + Setup_Competition_Map.py.
# Builds acc_stage1_mission, starts SLAM+Nav, optionally starts detector, then mission.
#
# Usage:
#   ./run_stage1_dev.sh              # Nav2 mission (no obstacle detection)
#   ./run_stage1_dev.sh --detect     # With traffic system detector (CPU)
#   ./run_stage1_dev.sh --yolo       # With YOLO detector (GPU required)
#   ./run_stage1_dev.sh --mpcc       # MPCC controller (path following only)
#   ./run_stage1_dev.sh --mpcc --yolo  # MPCC + YOLO detector
# =============================================================================

set -e

# Parse arguments
USE_DETECTOR=""
USE_MPCC=""
for arg in "$@"; do
  case "$arg" in
    --detect)  USE_DETECTOR="traffic_system_detector" ;;
    --yolo)    USE_DETECTOR="yolo_detector" ;;
    --mpcc|--mpcc-gpu) USE_MPCC="true" ;;
  esac
done

# Workspace: when run from run_dev.sh, Development is mounted at /workspaces/isaac_ros-dev
if [ -n "${WORKSPACE}" ]; then
  ROS2_WS="${WORKSPACE}/ros2"
elif [ -d /workspaces/isaac_ros-dev/ros2 ]; then
  ROS2_WS=/workspaces/isaac_ros-dev/ros2
else
  ROS2_WS="$(cd "$(dirname "$0")/../.." && pwd)"
fi

cd "$ROS2_WS"
echo "=============================================="
echo "  Stage I DEV: workspace $ROS2_WS"
echo "=============================================="

# Source ROS2 and existing install if present
source /opt/ros/humble/setup.bash
[ -f install/setup.bash ] && source install/setup.bash

# Build mission package
echo "== Building acc_stage1_mission..."
colcon build --packages-select acc_stage1_mission --symlink-install
source install/setup.bash

# Check if SLAM+Nav is already running
if ! ros2 node list 2>/dev/null | grep -q bt_navigator; then
  echo "== Starting SLAM + Nav2 (background)..."
  ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py &
  NAV_PID=$!
  echo "   Waiting 25s for Nav2 to initialize..."
  sleep 25
else
  echo "== SLAM+Nav already running."
fi

# Start detector if requested
if [ -n "$USE_DETECTOR" ]; then
  echo "== Starting $USE_DETECTOR (background)..."
  ros2 run qcar2_autonomy "$USE_DETECTOR" &
  DETECTOR_PID=$!
  sleep 2
fi

# Select launch file
if [ -n "$USE_MPCC" ]; then
  LAUNCH_FILE="mpcc_mission_launch.py"
else
  LAUNCH_FILE="mission_launch.py"
fi

echo "=============================================="
echo "  Launching Stage I Mission"
echo "  Controller: $([ -n "$USE_MPCC" ] && echo 'MPCC' || echo 'Nav2 MPPI')"
if [ -n "$USE_DETECTOR" ]; then
  echo "  Detector: $USE_DETECTOR"
fi
echo "  Launch: $LAUNCH_FILE"
echo ""
LOG_DIR="$ROS2_WS/src/acc_stage1_mission/logs"
echo "  Log files will be written to:"
echo "    Container: /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/logs/"
echo "    Host:      ~/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission/logs/"
echo "    - behavior_*.log   (timestamped events)"
echo "    - coordinates_*.csv (1 Hz position trace)"
echo "=============================================="

# Cleanup on exit
cleanup() {
  echo ""
  echo "== Shutting down..."
  [ -n "$DETECTOR_PID" ] && kill $DETECTOR_PID 2>/dev/null || true
  [ -n "$NAV_PID" ] && kill $NAV_PID 2>/dev/null || true
  echo ""
  echo "== Log files saved in: $LOG_DIR"
  ls -lt "$LOG_DIR" 2>/dev/null | head -5 || ls -lt /tmp/mission_logs/ 2>/dev/null | head -5
}
trap cleanup EXIT

# Launch mission
exec ros2 launch acc_stage1_mission "$LAUNCH_FILE"
