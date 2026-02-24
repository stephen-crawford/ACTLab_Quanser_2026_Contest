#!/usr/bin/env bash
# =============================================================================
# Stage I Contest - Host Runner
# =============================================================================
# Run from host (Ubuntu 24.04) to orchestrate containers.
# 1) Ensures ENV container is running, optionally runs Setup_Competition_Map.py
# 2) Starts DEV container in a new terminal (or prints commands)
# 3) You run the scenario in ENV and run_stage1_dev.sh in DEV
#
# Usage:
#   ./run_stage1.sh              # Just setup containers
#   ./run_stage1.sh --scenario   # Also run Setup_Competition_Map.py
# =============================================================================

set -e

# Paths — quanser-acc is the canonical repo, Docker infrastructure in ACC_Development
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUANSER_ACC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ACC_ROOT="${ACC_DEVELOPMENT:-$HOME/Documents/ACC_Development}"

ENV_IMAGE="quanser/virtual-qcar2"
ENV_NAME="virtual-qcar2"
DEV_SCRIPT="$ACC_ROOT/isaac_ros_common/scripts/run_dev.sh"
DEV_MOUNT="$ACC_ROOT/Development"
ROS2_WS="$ACC_ROOT/Development/ros2"
DOCKER_PKG_DIR="$ROS2_WS/src/acc_stage1_mission"

echo "=============================================="
echo "  Stage I contest – run everything"
echo "  ACC root: $ACC_ROOT"
echo "=============================================="

# --- 1) ENV container ---
if ! docker ps -a --format '{{.Names}}' | grep -q "^${ENV_NAME}$"; then
  echo "== Starting ENV container (background)..."
  docker run -d --name "$ENV_NAME" --network host "$ENV_IMAGE" sleep infinity
  echo "   Started. Open another terminal and run: docker exec -it $ENV_NAME bash"
elif ! docker ps --format '{{.Names}}' | grep -q "^${ENV_NAME}$"; then
  echo "== Starting existing ENV container..."
  docker start "$ENV_NAME"
fi

echo ""
echo "ENV container: $ENV_NAME (run 'docker exec -it $ENV_NAME bash' for a shell)"
echo "  In that shell, after QLabs Plane World is open, run:"
echo "  python3 /home/qcar2_scripts/python/Base_Scenarios_Python/Setup_Competition_Map.py 2>/dev/null || python3 /home/qcar2_scripts/python/Setup_Competition_Map.py"
echo ""

# --- 2) Run scenario in ENV (optional) ---
if [ "$1" = "--scenario" ]; then
  echo "== Running Setup_Competition_Map in ENV container..."
  docker exec "$ENV_NAME" python3 /home/qcar2_scripts/python/Base_Scenarios_Python/Setup_Competition_Map.py 2>/dev/null || \
  docker exec "$ENV_NAME" python3 /home/qcar2_scripts/python/Setup_Competition_Map.py 2>/dev/null || true
  echo "   Done. Ensure QLabs Plane World is open."
  echo ""
fi

# --- 2b) Sync quanser-acc to Docker workspace ---
echo "== Syncing quanser-acc → Docker workspace..."
mkdir -p "$ROS2_WS/src"
rsync -a --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='*.pyc' --exclude='run_mission.sh' \
    --exclude='logs/*.log' --exclude='logs/*.csv' \
    "$QUANSER_ACC_DIR/" "$DOCKER_PKG_DIR/"
echo "   Done."
echo ""

# --- 3) DEV container ---
if [ ! -x "$DEV_SCRIPT" ]; then
  echo "DEV script not found: $DEV_SCRIPT"
  echo "Start the DEV container manually, then inside it run:"
  echo "  cd /workspaces/isaac_ros-dev/ros2 && ./src/acc_stage1_mission/scripts/run_stage1_dev.sh"
  exit 0
fi

# Try to open a new terminal for DEV
OPEN_CMD=""
if command -v gnome-terminal &>/dev/null; then
  OPEN_CMD="gnome-terminal -- bash -c 'cd $ACC_ROOT/isaac_ros_common && ./scripts/run_dev.sh $DEV_MOUNT; exec bash'"
elif command -v xterm &>/dev/null; then
  OPEN_CMD="xterm -e \"cd $ACC_ROOT/isaac_ros_common && ./scripts/run_dev.sh $DEV_MOUNT; exec bash\""
fi

if [ -n "$OPEN_CMD" ]; then
  echo "== Opening new terminal for DEV container..."
  eval "$OPEN_CMD"
  echo ""
  echo "In the DEV container shell that just opened, run:"
  echo "  cd /workspaces/isaac_ros-dev/ros2 && ./src/acc_stage1_mission/scripts/run_stage1_dev.sh"
else
  echo "== Start DEV container in another terminal:"
  echo "  cd $ACC_ROOT/isaac_ros_common"
  echo "  ./scripts/run_dev.sh $DEV_MOUNT"
  echo ""
  echo "Then inside the DEV container:"
  echo "  cd /workspaces/isaac_ros-dev/ros2 && ./src/acc_stage1_mission/scripts/run_stage1_dev.sh"
fi

echo ""
echo "Summary:"
echo "  1) QLabs → Plane World"
echo "  2) ENV: docker exec -it $ENV_NAME bash → run Setup_Competition_Map.py"
echo "  3) DEV: run_dev.sh → run_stage1_dev.sh"
echo "=============================================="
