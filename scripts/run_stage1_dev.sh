#!/usr/bin/env bash
# =============================================================================
# Stage I Mission - DEV Container Runner
# =============================================================================
# Run inside the DEV (Isaac-ROS) container after QLabs + Setup_Competition_Map.py.
# Builds acc_stage1_mission (+ C++ nodes for MPCC mode), starts SLAM+Nav, then mission.
#
# Usage:
#   ./run_stage1_dev.sh              # Nav2 mission (no obstacle detection)
#   ./run_stage1_dev.sh --detect     # With traffic system detector (CPU)
#   ./run_stage1_dev.sh --yolo       # With YOLO detector (GPU required)
#   ./run_stage1_dev.sh --mpcc       # Full C++ MPCC stack (recommended)
#   ./run_stage1_dev.sh --mpcc-2025  # PolyCtrl 2025 Python stack (CasADi + YOLO)
#   ./run_stage1_dev.sh --mpcc --yolo  # MPCC + YOLO detector
#
# The --mpcc mode uses C++ nodes by default (controller, mission, sign detector).
# Module configuration: config/modules.yaml
# =============================================================================

set -e

# Parse arguments
USE_DETECTOR=""
USE_MPCC=""
USE_PRESET=""
USE_DASHBOARD=""
for arg in "$@"; do
  case "$arg" in
    --detect)  USE_DETECTOR="traffic_system_detector" ;;
    --yolo)    USE_DETECTOR="yolo_detector" ;;
    --mpcc|--mpcc-gpu) USE_MPCC="true" ;;
    --mpcc-2025|--2025|--legacy) USE_MPCC="true"; USE_PRESET="legacy_2025" ;;
    --dashboard) USE_DASHBOARD="true" ;;
  esac
done

# Workspace: when run from run_dev.sh, Development is mounted at /workspaces/isaac_ros-dev
# If running inside Docker container, use the container path
# If running on host, use the workspace directory
if [ -d /workspaces/isaac_ros-dev/ros2 ]; then
  ROS2_WS=/workspaces/isaac_ros-dev/ros2
elif [ -n "${WORKSPACE}" ]; then
  ROS2_WS="${WORKSPACE}/ros2"
else
  ROS2_WS="$HOME/Documents/ACC_Development/Development/ros2"
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

# Build C++ nodes for MPCC mode
if [ -n "$USE_MPCC" ]; then
  echo "== Building C++ nodes (acc_mpcc_controller_cpp)..."
  colcon build --packages-select acc_mpcc_controller_cpp 2>&1 | tail -10
  source install/setup.bash

  # Build C++ MPCC solver shared library (for Python ctypes fallback)
  CPP_DIR="$ROS2_WS/src/acc_stage1_mission/cpp"
  if [ -f "$CPP_DIR/mpcc_solver.cpp" ]; then
    echo "== Building C++ MPCC solver library..."
    EIGEN_INCLUDE=$(pkg-config --cflags eigen3 2>/dev/null || echo '-I/usr/include/eigen3')
    g++ -shared -fPIC -O2 $EIGEN_INCLUDE -std=c++17 \
        -o "$CPP_DIR/libmpcc_solver.so" "$CPP_DIR/mpcc_solver.cpp" 2>&1 || \
        echo "   Warning: C++ solver library build failed (CasADi fallback available)"
  fi
fi

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

# Start detector if requested (non-MPCC mode only; MPCC includes C++ sign_detector)
DETECTOR_PID=""
if [ -n "$USE_DETECTOR" ] && [ -z "$USE_MPCC" ]; then
  echo "== Starting $USE_DETECTOR (background)..."
  ros2 run qcar2_autonomy "$USE_DETECTOR" &
  DETECTOR_PID=$!
  sleep 2
elif [ -n "$USE_DETECTOR" ] && [ -n "$USE_MPCC" ]; then
  echo "== MPCC mode: C++ sign_detector handles detection (skipping $USE_DETECTOR)"
  echo "   To use YOLO instead, pass use_cpp_sign_detector:=false to the launch"
fi

# Select launch file and args
if [ -n "$USE_MPCC" ]; then
  LAUNCH_FILE="mpcc_mission_launch.py"
  LAUNCH_ARGS=""

  # Load preset if specified (e.g., --mpcc-2025)
  if [ -n "$USE_PRESET" ]; then
    PRESET_FILE="$ROS2_WS/src/acc_stage1_mission/config/presets/${USE_PRESET}.yaml"
    if [ -f "$PRESET_FILE" ]; then
      echo "== Loading preset: $USE_PRESET"
      # Extract top-level scalar key:value pairs as launch args
      while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]] ]] && continue
        [[ "$line" =~ ^[a-z_]+:[[:space:]]*$ ]] && continue
        key="${line%%:*}"
        value="${line#*: }"
        [[ "$key" == "detection" || "$key" == "path_planning" || "$key" == "controller" || "$key" == "pedestrian_tracking" ]] && continue
        value=$(echo "$value" | sed 's/[[:space:]]*#.*//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed "s/^['\"]//;s/['\"]$//")
        [[ -z "$value" ]] && continue
        LAUNCH_ARGS="$LAUNCH_ARGS ${key}:=${value}"
      done < "$PRESET_FILE"
    else
      echo "WARNING: Preset file not found: $PRESET_FILE"
    fi
  fi

  # Override sign detector if YOLO requested
  if [ "$USE_DETECTOR" = "yolo_detector" ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS use_cpp_sign_detector:=false"
  fi

  # Enable dashboard if requested
  if [ -n "$USE_DASHBOARD" ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS use_dashboard:=true"
  fi
else
  LAUNCH_FILE="mission_launch.py"
  LAUNCH_ARGS=""
fi

echo "=============================================="
echo "  Launching Stage I Mission"
echo "  Controller: $([ -n "$USE_MPCC" ] && echo 'MPCC (C++ stack)' || echo 'Nav2 MPPI')"
if [ -n "$USE_MPCC" ]; then
  echo "  Nodes:      C++ MPCC controller, mission manager, sign detector"
  echo "  Config:     config/modules.yaml"
fi
if [ -n "$USE_DETECTOR" ] && [ -z "$USE_MPCC" ]; then
  echo "  Detector:   $USE_DETECTOR"
fi
echo "  Launch:     $LAUNCH_FILE $LAUNCH_ARGS"
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
exec ros2 launch acc_stage1_mission "$LAUNCH_FILE" $LAUNCH_ARGS
