#!/bin/bash
#
# ACC Competition - Full Mission Runner
# Spawns all required terminals and launches the complete C++ stack.
# GPU YOLO detection is auto-detected and enabled if available.
#
# Per Virtual Detailed Scenario:
# https://quanser.github.io/student-competitions/events/common/Rules_and_Objectives/Virtual_Detailed_Scenario.html
#
# Mission Route:
#   1. Start at Taxi Hub [-1.205, -0.83] - LED: Magenta
#   2. Navigate to Pickup [0.125, 4.395] - LED: Green->Blue
#   3. Navigate to Dropoff [-0.905, 0.800] - LED: Green->Orange
#   4. Return to Taxi Hub - LED: Magenta
#
# Usage:
#   ./run_mission.sh             - Launch full stack (auto-detects GPU)
#   ./run_mission.sh --no-gpu    - Launch without GPU YOLO (C++ HSV only)
#   ./run_mission.sh --2025      - Use PolyCtrl 2025 MPCC weights
#   ./run_mission.sh --dashboard - Enable real-time telemetry dashboard
#   ./run_mission.sh --overlay   - Enable path overlay map visualizer
#   ./run_mission.sh --stop      - Stop all nodes (sends zero motor commands first)
#   ./run_mission.sh --stop --report - Stop + generate planned-vs-executed path report
#   ./run_mission.sh --reset     - Reset scenario (sync code + rebuild + reset car)
#   ./run_mission.sh --logs      - Show latest session logs
#   ./run_mission.sh --logs <s>  - Show logs for specific session
#
# C++ Stack (always):
#   - mpcc_controller_node:   C++ SQP MPCC controller (Eigen + gradient projection)
#   - mission_manager_node:   C++ mission state machine + road graph path planning
#   - sign_detector_node:     C++ HSV contour-based sign/light/cone detection
#   - odom_from_tf_node:      C++ TF -> /odom bridge
#   - state_estimator_node:   C++ EKF state estimator
#   - obstacle_tracker_node:  C++ multi-class Kalman + lidar tracker
#   - traffic_light_map_node: C++ spatial traffic light mapping
#
# GPU YOLO (auto-detected, alongside C++ HSV):
#   - yolo_bridge:             ROS2 bridge to standalone GPU YOLO detector
#   - yolo_detector_standalone: Python 3.10 + CUDA YOLOv8 (custom QLabs model)
#

set -e

USE_GPU_YOLO=auto   # "auto" = detect at launch, "true" = force on, "false" = force off
USE_PRESET=""
USE_DASHBOARD=false
USE_OVERLAY=false
GENERATE_REPORT=false
SHOW_LOGS_SESSION=""  # specific session to view with --logs <session>
PID_FILE="/tmp/acc_mission_pids.txt"
WID_FILE="/tmp/acc_mission_wids.txt"

# Capture the invoking terminal's window ID immediately at script start.
# This is used by stop_all() to avoid closing the terminal the user ran the script from.
INVOKING_WID=""
INVOKING_PID=$$
if command -v xdotool &> /dev/null; then
    INVOKING_WID=$(xdotool getactivewindow 2>/dev/null || true)
fi

# Configuration
# This script lives in the quanser-acc repo (single source of truth).
# The Docker infrastructure (isaac_ros_common, run_dev.sh) still lives in ACC_Development.
# We rsync our code to the Docker workspace before each launch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANSER_ACC_DIR="$SCRIPT_DIR"
ACC_DEV_DIR="$HOME/Documents/ACC_Development"
ISAAC_ROS_DIR="$ACC_DEV_DIR/isaac_ros_common"
DEV_WORKSPACE="$ACC_DEV_DIR/Development"
DOCKER_PKG_DIR="$DEV_WORKSPACE/ros2/src/acc_stage1_mission"
QUANSER_SCRIPTS="$ACC_DEV_DIR/docker/quanser_docker/python"
SCENARIO_SCRIPT="Setup_Real_Scenario_Interleaved.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║        ACC Competition - Full Mission Runner                  ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

# Sync quanser-acc repo → Docker workspace path
# The Docker container mounts Development/ as /workspaces/isaac_ros-dev/
# so we need our code at Development/ros2/src/acc_stage1_mission/
sync_to_workspace() {
    print_info "Syncing quanser-acc → Docker workspace..."

    # Ensure the target directory structure exists
    mkdir -p "$DEV_WORKSPACE/ros2/src"

    # Rsync with delete to ensure workspace matches repo exactly
    # Exclude .git, __pycache__, build artifacts, and logs (logs stay local)
    rsync -a --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='*.pyc' \
        --exclude='run_mission.sh' \
        --exclude='logs/' \
        --exclude='mpcc_launch_args.txt' \
        "$QUANSER_ACC_DIR/" "$DOCKER_PKG_DIR/"

    print_status "Code synced: $QUANSER_ACC_DIR → $DOCKER_PKG_DIR"
}

# Launch a terminal and track it for cleanup
# Usage: launch_terminal "Title" "geometry" "command"
launch_terminal() {
    local title="$1"
    local geometry="$2"
    local cmd="$3"

    # Determine session log file path (if session logging is enabled)
    local log_file=""
    local use_tee="0"
    if [ -n "${SESSION_DIR:-}" ]; then
        log_file="${SESSION_DIR}/${title}.log"
        use_tee="1"
    fi

    # Write the bash child PID to a file so stop_all() can kill it directly.
    # gnome-terminal uses a single server process for all windows, so we can't
    # track by window ID. Instead, each terminal writes its shell PID to a
    # known file, and we also set ACC_MISSION_TERMINAL for pgrep matching.
    local pid_marker="/tmp/acc_terminal_${title}.pid"

    # Launch terminal with environment marker for easy identification
    gnome-terminal --title="$title" --geometry="$geometry" -- bash -c "
        # Record shell PID for stop_all() cleanup
        echo \$\$ > '${pid_marker}'
        echo \$\$ >> '${PID_FILE}'
        # Set title via escape sequences
        echo -ne '\033]0;$title\007'
        echo -ne '\033]2;$title\007'
        export ACC_MISSION_TERMINAL='$title'
        # Keep setting title periodically to override any changes
        (while true; do echo -ne '\033]0;$title\007'; sleep 5; done) &
        TITLE_PID=\$!
        trap \"kill \$TITLE_PID 2>/dev/null; rm -f '${pid_marker}'\" EXIT
        # Run the command (with tee for session logging if enabled)
        if [ ${use_tee} = 1 ]; then
            ($cmd) 2>&1 | tee -a ${log_file}
        else
            $cmd
        fi
        # Keep terminal open briefly so user can see exit status, then close
        echo ''
        echo 'Command exited. Terminal will close in 5 seconds (Ctrl+C to keep open)...'
        sleep 5
    " &

    # gnome-terminal returns immediately (it delegates to gnome-terminal-server),
    # so $! is the launcher PID, not the shell PID. The real shell PID is written
    # to pid_marker by the bash -c block above.
    # Wait briefly for the shell to start and record its PID
    sleep 1
    if [ -f "$pid_marker" ]; then
        local shell_pid=$(cat "$pid_marker" 2>/dev/null)
        print_status "Window '$title' launched (shell PID: ${shell_pid:-?})"
    else
        print_status "Window '$title' launched"
    fi
}

# Check if simulation container is running, start it if not
check_simulation() {
    if docker ps --format '{{.Names}}' | grep -q "virtual-qcar2"; then
        print_status "Simulation container is running"
        return 0
    fi

    print_warning "Simulation container (virtual-qcar2) not running. Starting it..."

    # Check if QLabs is running
    if ! pgrep -f "QLabs" > /dev/null 2>&1 && ! pgrep -f "Quanser" > /dev/null 2>&1; then
        print_warning "QLabs may not be running. Make sure to:"
        echo "    1. Launch QLabs from your applications"
        echo "    2. Select 'Open World' -> 'Plane'"
        echo ""
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Remove any stopped container with the same name
    docker rm virtual-qcar2 2>/dev/null || true

    print_info "Starting container with scenario: $SCENARIO_SCRIPT"
    docker run --rm -d \
        --network host \
        --name virtual-qcar2 \
        -v "$QUANSER_SCRIPTS:/home/qcar2_scripts/python:rw" \
        quanser/virtual-qcar2:latest \
        bash -c "cd /home/qcar2_scripts/python && python3 Base_Scenarios_Python/$SCENARIO_SCRIPT"

    # Wait for container to initialize and verify it's running.
    # The scenario script connects to QLabs, destroys old actors, and spawns
    # floor + walls + car + signs + cameras + RT model. This takes 5-10s.
    print_info "Waiting for simulation scenario to initialize..."
    sleep 5
    if ! docker ps --format '{{.Names}}' | grep -q "virtual-qcar2"; then
        print_error "Simulation container exited — QLabs may not be running or accessible"
        print_info "Check logs with: docker logs virtual-qcar2"
        print_info "Ensure QLabs is running with Plane world selected"
        exit 1
    fi

    # Verify the scenario script is still running (it enters an infinite traffic
    # light loop after setup — if it exited, setup failed).
    if ! docker exec virtual-qcar2 pgrep -f "$SCENARIO_SCRIPT" > /dev/null 2>&1; then
        print_error "Scenario script exited during initialization"
        print_info "The scenario connects to QLabs and spawns the competition environment."
        print_info "If QLabs is not running or on the wrong world, the script exits immediately."
        print_info "Check container logs: docker logs virtual-qcar2"
        exit 1
    fi

    # Wait for ALL virtual hardware ports to be available.
    # QLabs spawns sensors progressively — the lidar (port 18966) is critical
    # for Cartographer SLAM. If it's not ready when the lidar node starts,
    # the lidar node exits immediately and Cartographer never gets scan data.
    # Ports: 18960=HIL, 18962=CSI, 18965=RGBD, 18966=Lidar, 18969=LED
    local required_ports="18960 18965 18966"
    print_info "Waiting for virtual hardware ports: HIL=18960, RGBD=18965, Lidar=18966..."
    local port_wait=0
    local port_max=90
    local all_ready=false
    while [ "$all_ready" = false ]; do
        if [ $port_wait -ge $port_max ]; then
            print_warning "Not all virtual hardware ports detected after ${port_max}s — proceeding anyway"
            break
        fi
        all_ready=true
        for port in $required_ports; do
            if ! ss -tln 2>/dev/null | grep -q ":${port} " && ! nc -z localhost "$port" 2>/dev/null; then
                all_ready=false
                break
            fi
        done
        if [ "$all_ready" = false ]; then
            sleep 2
            port_wait=$((port_wait + 2))
            printf "\r${BLUE}[i]${NC} Waiting for simulation hardware... %ds" "$port_wait"
        fi
    done
    echo ""
    # Extra settle time — even after ports open, QLabs may still be initializing sensors
    if [ "$all_ready" = true ]; then
        print_status "All virtual hardware ports ready"
        sleep 3
    fi
    print_status "Simulation container started — $SCENARIO_SCRIPT active"
}

# Check if dev container is running
check_dev_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "isaac_ros_dev"; then
        print_warning "Dev container not running. Starting it now..."
        cd "$ISAAC_ROS_DIR"
        gnome-terminal --title="Dev Container" -- bash -c "./scripts/run_dev.sh $DEV_WORKSPACE; exec bash"
        print_info "Waiting for dev container to start (this may take a minute on first run)..."
        local max_wait=120
        local elapsed=0
        while ! docker ps --format '{{.Names}}' | grep -q "isaac_ros_dev"; do
            if [ $elapsed -ge $max_wait ]; then
                print_error "Dev container failed to start within ${max_wait}s"
                exit 1
            fi
            sleep 3
            elapsed=$((elapsed + 3))
            printf "\r${BLUE}[i]${NC} Waiting for dev container... %ds" "$elapsed"
        done
        echo ""
        print_info "Container is up. Waiting for ROS2 workspace to be ready..."
        # Give the container a moment to finish its entrypoint/setup
        local container_id
        container_id=$(docker ps -qf "name=isaac_ros_dev" | head -1)
        local setup_wait=60
        local setup_elapsed=0
        while ! docker exec "$container_id" test -f /opt/ros/humble/setup.bash 2>/dev/null; do
            if [ $setup_elapsed -ge $setup_wait ]; then
                print_warning "Timed out waiting for ROS2 setup inside container, proceeding anyway..."
                break
            fi
            sleep 3
            setup_elapsed=$((setup_elapsed + 3))
            printf "\r${BLUE}[i]${NC} Waiting for container setup... %ds" "$setup_elapsed"
        done
        echo ""
    fi
    print_status "Dev container is running"
}

# Get the dev container ID
get_container_id() {
    docker ps -qf "name=isaac_ros_dev" | head -1
}

# Command to run inside dev container
dev_cmd() {
    local cmd="$1"
    local container_id=$(get_container_id)
    echo "source /opt/ros/humble/setup.bash && source /workspace/cartographer_ws/install/setup.bash && source /workspaces/isaac_ros-dev/ros2/install/setup.bash && $cmd"
}

reset_car() {
    print_header
    print_info "Resetting full scenario (car, map, obstacles, pedestrians)..."

    # Sync code changes first so any rebuild picks up latest source
    sync_to_workspace

    # Rebuild packages if dev container is running
    CONTAINER_ID=$(get_container_id)
    if [ -n "$CONTAINER_ID" ]; then
        print_info "Rebuilding packages with latest code changes..."
        docker exec "$CONTAINER_ID" bash -c '
            source /opt/ros/humble/setup.bash
            cd /workspaces/isaac_ros-dev/ros2
            colcon build --packages-select acc_stage1_mission --symlink-install 2>&1 | tail -5
        ' && print_status "Python package rebuilt" || print_warning "Python build had warnings"

        # Rebuild C++ nodes (clean rebuild to ensure source changes take effect)
        docker exec "$CONTAINER_ID" bash -c '
            source /opt/ros/humble/setup.bash
            cd /workspaces/isaac_ros-dev/ros2
            ln -sfn /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/cpp src/acc_mpcc_controller_cpp
            rm -rf build/acc_mpcc_controller_cpp install/acc_mpcc_controller_cpp
            find src/acc_mpcc_controller_cpp -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs touch 2>/dev/null
            colcon build --packages-select acc_mpcc_controller_cpp \
                --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 2>&1 | tail -10
        ' && print_status "C++ nodes rebuilt (clean)" || print_warning "C++ build had warnings"
    fi

    # Ensure simulation container is running
    check_simulation

    # Stop any running scenario scripts in the container
    print_info "Stopping current scenario..."
    docker exec virtual-qcar2 pkill -f "python3.*Setup_" 2>/dev/null || true
    docker exec virtual-qcar2 pkill -f "python3.*reset_qcar2" 2>/dev/null || true
    sleep 1

    # Run the competition scenario script (same one used at launch)
    print_info "Starting fresh scenario: $SCENARIO_SCRIPT"
    docker exec -d virtual-qcar2 bash -c \
        "cd /home/qcar2_scripts/python && python3 Base_Scenarios_Python/$SCENARIO_SCRIPT"

    sleep 5  # Competition scenario needs more time (spawns floor, walls, signs, etc.)

    # Check if it's running
    if docker exec virtual-qcar2 pgrep -f "$SCENARIO_SCRIPT" > /dev/null 2>&1; then
        print_status "Scenario reset complete!"
        echo ""
        echo "  - QCar2 at starting position [-1.205, -0.83]"
        echo "  - Traffic lights cycling (tied to pedestrian movement)"
        echo "  - Pedestrians spawned (competition behavior)"
        echo "  - Code synced and rebuilt"
        echo ""
        print_warning "NOTE: You should also restart the ROS2 stack for a clean SLAM map:"
        echo "    ./run_mission.sh --stop"
        echo "    ./run_mission.sh"
    else
        print_error "Failed to start scenario"
        print_info "Check container logs: docker logs virtual-qcar2"
    fi
}

generate_report() {
    print_info "Generating mission report..."

    # CSV logs are written to the Docker-mounted workspace, so they appear on the host at:
    #   $DOCKER_PKG_DIR/logs/mpcc_*.csv
    # The quanser-acc/logs/ symlink or direct path also works after rsync.
    local local_csv=""

    # Primary: check the Docker workspace mount on the host (logs written directly here)
    local_csv=$(ls -t "$DOCKER_PKG_DIR"/logs/mpcc_*.csv 2>/dev/null | head -1 || true)

    # Fallback: check quanser-acc/logs/ (in case of local symlink or manual copy)
    if [ -z "$local_csv" ]; then
        local_csv=$(ls -t "$QUANSER_ACC_DIR"/logs/mpcc_*.csv 2>/dev/null | head -1 || true)
    fi

    # Last resort: try docker cp from /tmp/mission_logs (old log location)
    if [ -z "$local_csv" ]; then
        CONTAINER_ID=$(get_container_id)
        if [ -n "$CONTAINER_ID" ]; then
            local container_csv
            container_csv=$(docker exec "$CONTAINER_ID" bash -c \
                'ls -t /tmp/mission_logs/mpcc_*.csv 2>/dev/null | head -1' 2>/dev/null || true)
            if [ -n "$container_csv" ]; then
                mkdir -p "$QUANSER_ACC_DIR/logs"
                local_csv="$QUANSER_ACC_DIR/logs/$(basename "$container_csv")"
                docker cp "$CONTAINER_ID:$container_csv" "$local_csv" 2>/dev/null && {
                    print_status "Copied CSV from container: $(basename "$container_csv")"
                } || {
                    print_error "Failed to copy CSV from container"
                    local_csv=""
                }
            fi
        fi
    fi

    if [ -z "$local_csv" ] || [ ! -f "$local_csv" ]; then
        print_warning "No MPCC CSV log found — cannot generate report"
        print_info "Run a mission first, then use --stop --report."
        return 1
    fi

    print_info "Using CSV: $local_csv"

    # Run the report generation script
    if [ -f "$QUANSER_ACC_DIR/scripts/generate_report.py" ]; then
        python3 "$QUANSER_ACC_DIR/scripts/generate_report.py" "$local_csv" && {
            print_status "Report generated successfully"
        } || {
            print_error "Report generation failed"
            return 1
        }
    else
        print_error "Report script not found: scripts/generate_report.py"
        return 1
    fi
}

stop_all() {
    print_header
    print_info "Stopping all mission components..."

    # Check for window management tools
    if ! command -v xdotool &> /dev/null && ! command -v wmctrl &> /dev/null; then
        print_warning "xdotool and wmctrl not installed. Terminal windows may not close automatically."
        print_info "Install with: sudo apt install xdotool wmctrl"
    fi

    # Use the invoking terminal's WID saved at script start (more reliable
    # than xdotool getactivewindow which may return a spawned terminal's WID)
    CURRENT_WID="${INVOKING_WID:-}"

    CONTAINER_ID=$(get_container_id)

    # Step 0: Kill ROS2 processes FIRST so qcar2_hardware releases the HIL port.
    # We need the HIL port free to send zero motor commands directly.
    if [ -n "$CONTAINER_ID" ]; then
        print_info "Stopping ROS2 nodes in container..."

        docker exec "$CONTAINER_ID" bash -c '
            pkill -9 -f "ros2" 2>/dev/null
            pkill -9 -f "python.*mission" 2>/dev/null
            pkill -9 -f "python.*obstacle" 2>/dev/null
            pkill -9 -f "python.*yolo" 2>/dev/null
            pkill -9 -f "python.*qcar" 2>/dev/null
            pkill -9 -f "python.*mpcc" 2>/dev/null
            pkill -9 -f "python.*odom" 2>/dev/null
            pkill -9 -f "python3.10" 2>/dev/null
            pkill -9 -f "cartographer" 2>/dev/null
            pkill -9 -f "nav2" 2>/dev/null
            pkill -9 -f "component_container" 2>/dev/null
            pkill -9 -f "_node" 2>/dev/null
            pkill -9 -f "static_transform_publisher" 2>/dev/null
            true
        ' 2>/dev/null || true

        print_status "ROS2 nodes stopped"

        # Brief pause for qcar2_hardware to release the HIL port
        sleep 0.5
    else
        print_warning "Dev container not running"
    fi

    # Step 0b: Zero motors via HIL + teleport to hub via QLabs API.
    # qcar2_hardware is now dead so we can open the HIL port directly.
    # HIL channels: 1000=steering_angle, 11000=motor_throttle (same as qcar2_hardware.cpp)
    print_info "Zeroing motors via HIL and resetting car..."
    timeout 5 docker exec virtual-qcar2 python3 -c "
import numpy as np
from quanser.hardware import HIL
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# 1. Zero motors through HIL (clears retained steering/throttle)
try:
    card = HIL()
    card.open('qcar2', '0@tcpip://localhost:18960')
    ch = np.array([1000, 11000], dtype=np.uint32)
    card.write_other(ch, 2, np.array([0.0, 0.0], dtype=np.float64))
    card.close()
    print('HIL motors zeroed')
except Exception as e:
    print(f'HIL zero skipped: {e}')

# 2. Teleport to hub with dynamics OFF (stops any remaining motion)
try:
    qlabs = QuanserInteractiveLabs()
    qlabs.open('localhost')
    car = QLabsQCar2(qlabs)
    car.actorNumber = 0
    car.set_transform_and_request_state_degrees(
        location=[-1.205, -0.83, 0.005], rotation=[0, 0, -44.7],
        enableDynamics=False, headlights=False,
        leftTurnSignal=False, rightTurnSignal=False,
        brakeSignal=False, reverseSignal=False,
        waitForConfirmation=True)
    qlabs.close()
    print('Car frozen at hub')
except Exception as e:
    print(f'QLabs reset skipped: {e}')
" 2>&1 || true

    print_info "Closing terminal windows..."

    # Step 2: Close windows using saved window IDs (most reliable)
    if [ -f "$WID_FILE" ]; then
        while read -r wid; do
            if [ -n "$wid" ]; then
                # Skip the current terminal window
                [ "$wid" = "$CURRENT_WID" ] && continue
                if command -v xdotool &> /dev/null; then
                    xdotool windowclose "$wid" 2>/dev/null || true
                elif command -v wmctrl &> /dev/null; then
                    wmctrl -i -c "$wid" 2>/dev/null || true
                fi
            fi
        done < "$WID_FILE"
        rm -f "$WID_FILE"
        print_status "Closed tracked windows"
    fi

    # Step 3: Kill ALL bash processes that are running docker exec commands
    pkill -9 -f "docker exec -it.*ros2 launch" 2>/dev/null || true
    pkill -9 -f "docker exec -it.*ros2 run" 2>/dev/null || true
    pkill -9 -f "docker exec -it.*source /opt/ros" 2>/dev/null || true
    pkill -9 -f "docker exec -it.*bash -c" 2>/dev/null || true
    pkill -9 -f "ACC_MISSION_TERMINAL" 2>/dev/null || true

    # Step 4: Kill processes by the specific command patterns from our script
    pkill -9 -f "Starting QCar2 hardware" 2>/dev/null || true
    pkill -9 -f "Starting SLAM and Navigation" 2>/dev/null || true
    pkill -9 -f "Starting Obstacle Detector" 2>/dev/null || true
    pkill -9 -f "Starting Mission Manager" 2>/dev/null || true
    pkill -9 -f "Starting MPCC Controller" 2>/dev/null || true
    pkill -9 -f "Starting YOLO Bridge" 2>/dev/null || true
    pkill -9 -f "GPU-accelerated YOLO" 2>/dev/null || true
    pkill -9 -f "Waiting.*seconds.*Nav2" 2>/dev/null || true

    # Step 5: Use xdotool to close windows by our exact terminal names (backup)
    # IMPORTANT: Only use exact names we assigned in launch_terminal().
    # Uses regex anchors (^...$) for exact matching to avoid substring collisions.
    if command -v xdotool &> /dev/null; then
        for title in "QCar2-Hardware" "SLAM-Nav2" "SLAM" "YOLO-Bridge" "GPU-YOLO" "Obstacle-Detector" "MPCC-Mission" "Mission-Manager" "Path-Overlay"; do
            for wid in $(xdotool search --name "^${title}$" 2>/dev/null); do
                # Skip the current terminal window
                [ "$wid" = "$CURRENT_WID" ] && continue
                xdotool windowclose "$wid" 2>/dev/null || true
            done
        done
    fi

    # Step 6: Use wmctrl as backup (close by exact window titles only)
    # wmctrl -c does substring matching, so we verify with wmctrl -l first
    if command -v wmctrl &> /dev/null; then
        for title in "QCar2-Hardware" "SLAM-Nav2" "SLAM" "YOLO-Bridge" "GPU-YOLO" "Obstacle-Detector" "MPCC-Mission" "Mission-Manager" "Path-Overlay"; do
            # Only close if the window title is an exact match
            # wmctrl -l format: "0x04800003  0 hostname Window Title"
            wmctrl -l | awk -v t="$title" '{
                # Title is everything after field 4 (host)
                wtitle = ""; for(i=4;i<=NF;i++) wtitle = (wtitle ? wtitle " " : "") $i
                if (wtitle == t) print $1
            }' | while read -r wid; do
                [ "$wid" = "$CURRENT_WID" ] && continue
                wmctrl -i -c "$wid" 2>/dev/null || true
            done
        done
    fi

    # Step 7: Kill any remaining bash -c processes that look like ours
    for pid in $(pgrep -f "bash -c.*\(docker exec\|ACC_MISSION\)"); do
        kill -9 "$pid" 2>/dev/null || true
    done

    # Step 8: Clean up files
    rm -f "$PID_FILE"
    rm -f "$WID_FILE"

    sleep 0.5
    print_status "All mission components stopped"
    print_status "Terminal windows closed"
    echo ""
    print_info "Dev container is still running (for quick restart)"
    print_info "To stop container: docker stop isaac_ros_dev-x86_64-container"
}

# Load a preset YAML and build launch arguments string
# Usage: LAUNCH_ARGS=$(load_preset "legacy_2025")
load_preset() {
    local preset_name="$1"
    local preset_file="$QUANSER_ACC_DIR/config/presets/${preset_name}.yaml"

    if [ ! -f "$preset_file" ]; then
        print_error "Preset not found: $preset_file"
        print_info "Available presets:"
        ls "$QUANSER_ACC_DIR/config/presets/"*.yaml 2>/dev/null | while read f; do
            echo "    $(basename "$f" .yaml)"
        done
        return 1
    fi

    print_info "Loading preset: $preset_name ($preset_file)"

    # Extract launch-level parameters from YAML (top-level scalar keys only)
    # These become ros2 launch arguments like key:=value
    # Skip indented lines (nested sections), comments, and section headers
    local args=""
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        # Skip indented lines (belong to nested sections)
        [[ "$line" =~ ^[[:space:]] ]] && continue
        # Skip section headers (lines that are just "key:" with no scalar value)
        [[ "$line" =~ ^[a-z_]+:[[:space:]]*$ ]] && continue
        # Parse "key: value" from non-indented lines
        local key="${line%%:*}"
        local value="${line#*: }"
        # Skip known section keys
        [[ "$key" == "detection" || "$key" == "path_planning" || "$key" == "controller" || "$key" == "pedestrian_tracking" ]] && continue
        # Clean value (remove trailing comments and whitespace)
        value=$(echo "$value" | sed 's/[[:space:]]*#.*//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed "s/^['\"]//;s/['\"]$//")
        [[ -z "$value" ]] && continue
        args="$args ${key}:=${value}"
    done < "$preset_file"

    echo "$args"
}

# Detect GPU availability inside the dev container.
# Returns 0 (true) if CUDA + torch + YOLO model are available, 1 (false) otherwise.
detect_gpu() {
    local container_id="$1"

    # Check 1: Does nvidia-smi work inside the container?
    if ! docker exec "$container_id" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -qi "nvidia\|rtx\|gtx\|tesla\|quadro\|geforce"; then
        return 1
    fi

    # Check 2: Can Python 3.10 import torch with CUDA?
    if ! docker exec "$container_id" python3.10 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        return 1
    fi

    # Check 3: Does the standalone YOLO detector script exist?
    if ! docker exec "$container_id" test -f /workspaces/isaac_ros-dev/yolo_detector_standalone.py 2>/dev/null; then
        return 1
    fi

    return 0
}

build_mpcc_solver() {
    local container_id="$1"

    print_info "Building C++ MPCC solver..."

    local cpp_dir="/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/cpp"

    docker exec "$container_id" bash -c "
        cd $cpp_dir
        EIGEN_INCLUDE=\$(pkg-config --cflags eigen3 2>/dev/null || echo '-I/usr/include/eigen3')
        g++ -shared -fPIC -O2 \$EIGEN_INCLUDE -std=c++17 \
            -o libmpcc_solver.so mpcc_solver.cpp 2>&1
    "

    if [ $? -eq 0 ]; then
        # Verify symbols
        if docker exec "$container_id" nm -D "$cpp_dir/libmpcc_solver.so" 2>/dev/null | grep -q mpcc_create; then
            print_status "C++ MPCC solver built successfully"
        else
            print_error "C++ MPCC solver built but symbols missing"
            return 1
        fi
    else
        print_error "C++ MPCC solver build failed"
        return 1
    fi
}

launch_mission() {
    print_header

    # Check for window management tools (needed for terminal cleanup)
    if ! command -v xdotool &> /dev/null; then
        print_warning "xdotool not installed. Installing for terminal management..."
        sudo apt-get install -y xdotool wmctrl 2>/dev/null || {
            print_warning "Could not install xdotool. Terminal cleanup may be manual."
        }
    fi

    # Clean up any previous run (kill ROS nodes, close spawned terminals)
    if [ -f "$WID_FILE" ] || [ -f "$PID_FILE" ]; then
        print_info "Cleaning up previous mission run..."
        stop_all
        echo ""
        print_header
    fi

    # Check prerequisites
    check_simulation

    # Verify the simulation container is still running (it may have exited between
    # check_simulation() and here if the scenario script crashed on startup).
    if ! docker ps --format '{{.Names}}' | grep -q "virtual-qcar2"; then
        print_error "Simulation container exited after check_simulation() — QLabs may not be running"
        print_info "Ensure QLabs is running with Plane world selected, then retry"
        exit 1
    fi

    # Check if scenario process is running, and if so, verify the QLabs actors
    # are still alive AND virtual hardware ports are open. If QLabs was restarted
    # since the scenario was launched, the process is still running its traffic-light
    # loop but all actors (car, signs, floor, etc.) were destroyed by the QLabs restart.
    # The car position reset serves as a health check for actor liveness.
    # The port check catches cases where actors exist but hardware is stale.
    local scenario_needs_restart=false
    if docker exec virtual-qcar2 pgrep -f "$SCENARIO_SCRIPT" > /dev/null 2>&1; then
        print_info "Scenario process is running — verifying QLabs actors are alive..."
        if docker exec virtual-qcar2 python3 /home/qcar2_scripts/python/reset_car_position.py 2>&1; then
            # Actor exists — now verify virtual hardware ports are actually open
            # Ports: 18960=HIL, 18965=RGBD, 18966=Lidar
            local ports_ok=true
            for port in 18960 18965 18966; do
                if ! nc -z localhost "$port" 2>/dev/null; then
                    ports_ok=false
                    print_warning "Port $port not open — virtual hardware is stale"
                    break
                fi
            done
            if [ "$ports_ok" = true ]; then
                print_status "Scenario healthy: actors verified, hardware ports open"
            else
                print_warning "Scenario actors exist but hardware ports are stale"
                print_info "Restarting scenario to re-initialize virtual hardware..."
                scenario_needs_restart=true
            fi
        else
            print_warning "Scenario process is running but QCar2 actor is missing"
            print_info "QLabs was likely restarted — all actors need to be re-spawned"
            scenario_needs_restart=true
        fi
    else
        print_info "Scenario is not running"
        scenario_needs_restart=true
    fi

    if [ "$scenario_needs_restart" = true ]; then
        # The scenario script is PID 1's child inside the container (started by
        # docker run). Killing it causes the container to exit and be removed
        # (--rm flag). We must stop the whole container and start a fresh one.
        print_info "Stopping stale simulation container..."
        docker stop virtual-qcar2 2>/dev/null || true
        docker rm -f virtual-qcar2 2>/dev/null || true
        sleep 2

        # Start a fresh container with the scenario (calls check_simulation
        # which handles container creation, scenario startup verification,
        # and port readiness checks).
        print_info "Starting fresh simulation with scenario..."
        check_simulation

        # Verify scenario is running after fresh start
        if ! docker exec virtual-qcar2 pgrep -f "$SCENARIO_SCRIPT" > /dev/null 2>&1; then
            print_error "Scenario failed to start after container restart"
            print_info "Check: docker logs virtual-qcar2"
            exit 1
        fi

        # Reset car to hub after fresh scenario start
        print_info "Resetting car to taxi hub position..."
        if docker exec virtual-qcar2 python3 /home/qcar2_scripts/python/reset_car_position.py 2>&1; then
            print_status "Car reset to hub [-1.205, -0.83]"
        else
            print_info "Car position reset skipped (car spawns at hub from scenario script)"
        fi
    fi

    check_dev_container

    CONTAINER_ID=$(get_container_id)

    if [ -z "$CONTAINER_ID" ]; then
        print_error "Could not find dev container ID"
        exit 1
    fi

    print_info "Using container: $CONTAINER_ID"
    echo ""

    # Auto-detect GPU if not explicitly set
    if [ "$USE_GPU_YOLO" = "auto" ]; then
        print_info "Detecting GPU availability..."
        if detect_gpu "$CONTAINER_ID"; then
            GPU_NAME=$(docker exec "$CONTAINER_ID" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            USE_GPU_YOLO=true
            print_status "GPU detected: $GPU_NAME — enabling YOLO detection"
        else
            USE_GPU_YOLO=false
            print_info "No GPU/CUDA/YOLO available — using C++ HSV detection only"
        fi
    elif [ "$USE_GPU_YOLO" = true ]; then
        print_info "GPU YOLO: forced on"
    else
        print_info "GPU YOLO: disabled (--no-gpu)"
    fi
    echo ""

    # Sync code from quanser-acc repo to Docker workspace
    sync_to_workspace

    # Build C++ nodes (MPCC controller, sign detector, mission manager, odom_from_tf)
    # The C++ stack is the primary codebase — no Python fallbacks.
    # Colcon won't discover acc_mpcc_controller_cpp inside acc_stage1_mission/cpp/
    # (nested packages are not supported), so create a sibling symlink.
    #
    # IMPORTANT: We force a clean C++ rebuild every time to ensure source changes
    # take effect. Without this, colcon's build cache can skip recompilation if
    # file timestamps don't appear newer (rsync -a preserves mtime). This was
    # the root cause of "source fixes not reaching the running binary" bugs.
    print_info "Building C++ nodes (acc_mpcc_controller_cpp) — CLEAN rebuild..."
    docker exec "$CONTAINER_ID" bash -c '
        source /opt/ros/humble/setup.bash
        cd /workspaces/isaac_ros-dev/ros2

        # Create symlink so colcon discovers the C++ package
        ln -sfn /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/cpp src/acc_mpcc_controller_cpp

        # Force clean rebuild: remove stale build/install artifacts.
        # This ensures ALL source changes are compiled into the binary.
        rm -rf build/acc_mpcc_controller_cpp install/acc_mpcc_controller_cpp

        # Touch all C++ source files to ensure timestamps are newer than any
        # cached objects (rsync -a preserves mtime which can fool cmake)
        find src/acc_mpcc_controller_cpp -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs touch 2>/dev/null

        colcon build --packages-select acc_mpcc_controller_cpp \
            --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 2>&1
    '
    if [ $? -eq 0 ]; then
        print_status "C++ nodes built successfully (clean rebuild)"
    else
        print_error "C++ node build FAILED — cannot proceed without C++ stack"
        print_info "Check build output above for errors"
        print_info "Common fixes:"
        echo "    - Missing dependency: sudo apt install libopencv-dev libeigen3-dev libyaml-cpp-dev"
        echo "    - Missing ROS2 package: sudo apt install ros-humble-cv-bridge ros-humble-nav2-msgs"
        exit 1
    fi

    # Verify C++ executables actually exist (catch silent build failures)
    print_info "Verifying C++ executables..."
    local cpp_verify_failed=false
    for exe in mpcc_controller_node mission_manager_node sign_detector_node odom_from_tf_node state_estimator_node obstacle_tracker_node traffic_light_map_node; do
        if ! docker exec "$CONTAINER_ID" bash -c "source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/ros2/install/setup.bash && ros2 pkg executables acc_mpcc_controller_cpp 2>/dev/null | grep -q $exe" 2>/dev/null; then
            print_error "C++ executable NOT found: $exe"
            cpp_verify_failed=true
        fi
    done
    if [ "$cpp_verify_failed" = true ]; then
        print_error "Some C++ executables are missing — cannot proceed"
        print_info "The launch file references acc_mpcc_controller_cpp executables."
        print_info "If they are missing, ros2 launch will fail or fall back to Python nodes."
        print_info "Try rebuilding: docker exec $CONTAINER_ID bash -c 'cd /workspaces/isaac_ros-dev/ros2 && colcon build --packages-select acc_mpcc_controller_cpp'"
        exit 1
    else
        print_status "All 7 C++ executables verified"
    fi

    # Build Python package (needed for launch files and GPU YOLO bridge)
    print_info "Building acc_stage1_mission package (launch files)..."
    docker exec "$CONTAINER_ID" bash -c '
        source /opt/ros/humble/setup.bash
        cd /workspaces/isaac_ros-dev/ros2
        colcon build --packages-select acc_stage1_mission --symlink-install 2>&1 | tail -5
    '
    if [ $? -ne 0 ]; then
        print_warning "Python package build had warnings (launch files may be stale)"
    fi

    # Create session log directory
    SESSION_DIR="$DOCKER_PKG_DIR/logs/session_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SESSION_DIR"
    export SESSION_DIR

    # Write session info
    {
        echo "Session: $(basename "$SESSION_DIR")"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Host: $(hostname)"
        echo "GPU: ${USE_GPU_YOLO}"
        echo "Preset: ${USE_PRESET:-default}"
        echo "Dashboard: ${USE_DASHBOARD}"
        echo "Overlay: ${USE_OVERLAY}"
        echo "Git commit: $(cd "$QUANSER_ACC_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        echo "Git branch: $(cd "$QUANSER_ACC_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
        echo "Git dirty: $(cd "$QUANSER_ACC_DIR" && git diff --quiet 2>/dev/null && echo 'no' || echo 'yes')"
    } > "$SESSION_DIR/session_info.txt"
    print_status "Session log directory: $SESSION_DIR"

    # Clear old tracking files
    rm -f "$PID_FILE" "$WID_FILE"
    touch "$PID_FILE" "$WID_FILE"

    # Terminal 1: Cartographer SLAM + Hardware Nodes
    # Launch qcar2_cartographer_virtual_launch.py directly (NOT the full
    # qcar2_slam_and_nav_bringup_virtual_launch.py). The bringup launch adds
    # AMCL (conflicts with Cartographer), static_odom_tf (conflicts with
    # Cartographer's provide_odom_frame=true), and Nav2 stack (unused — we have
    # our own MPCC path planner). The reference team (PolyCtrl 2025) also
    # launches qcar2_cartographer_virtual_launch.py directly.
    #
    # qcar2_cartographer_virtual_launch.py includes:
    #   - qcar2_virtual_launch.py (lidar, cameras, qcar2_hardware)
    #   - fixed_lidar_frame_virtual (base_link → base_scan TF)
    #   - cartographer_node (SLAM: map → odom → base_link TF)
    #   - cartographer_occupancy_grid_node (map publisher)
    # Fix Cartographer tracking_frame to match reference (PolyCtrl 2025).
    # The 2026 Quanser config changed tracking_frame to "base_scan" but the reference
    # uses "base_link". With base_scan, Cartographer initializes vehicle heading at 0
    # instead of ~41 deg, causing a 35 deg mismatch with the planned path.
    docker exec "$CONTAINER_ID" bash -c '
        CARTO_CFG="/workspaces/isaac_ros-dev/ros2/install/qcar2_nodes/share/qcar2_nodes/config/qcar2_2d.lua"
        if [ -f "$CARTO_CFG" ]; then
            sed -i "s/tracking_frame = \"base_scan\"/tracking_frame = \"base_link\"/" "$CARTO_CFG"
            echo "Cartographer config: tracking_frame = base_link (reference-matched)"
        else
            echo "WARNING: Cartographer config not found at $CARTO_CFG"
        fi
    '

    # Write SLAM launch helper script to Docker workspace (avoids nested quoting issues)
    docker exec "$CONTAINER_ID" bash -c 'cat > /tmp/slam_launch.sh << '\''SCRIPT'\''
#!/bin/bash
source /opt/ros/humble/setup.bash
source /workspace/cartographer_ws/install/setup.bash
source /workspaces/isaac_ros-dev/ros2/install/setup.bash

echo "=== Cartographer SLAM + Hardware ==="
echo "Checking QLabs virtual hardware ports from inside container..."
for p in 18960 18962 18965 18966 18969; do
    if timeout 1 bash -c "echo > /dev/tcp/localhost/$p" 2>/dev/null; then
        echo "  Port $p: OPEN"
    else
        echo "  Port $p: CLOSED"
    fi
done

echo ""
echo "Waiting for critical ports: 18960=HIL, 18965=RGBD, 18966=Lidar..."
WAIT=0
MAX_WAIT=90
while true; do
    ALL_OK=true
    for p in 18960 18965 18966; do
        if ! timeout 1 bash -c "echo > /dev/tcp/localhost/$p" 2>/dev/null; then
            ALL_OK=false
            break
        fi
    done
    if [ "$ALL_OK" = true ]; then
        echo "All critical ports ready!"
        break
    fi
    WAIT=$((WAIT + 2))
    if [ $WAIT -ge $MAX_WAIT ]; then
        echo "WARNING: Not all ports ready after ${MAX_WAIT}s - launching anyway"
        echo "Final port status:"
        for p in 18960 18962 18965 18966 18969; do
            if timeout 1 bash -c "echo > /dev/tcp/localhost/$p" 2>/dev/null; then
                echo "  Port $p: OPEN"
            else
                echo "  Port $p: CLOSED"
            fi
        done
        break
    fi
    sleep 2
    printf "\r  Waiting for hardware ports... %ds" $WAIT
done
echo ""

echo "Launching Cartographer SLAM + hardware nodes..."
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
SCRIPT
chmod +x /tmp/slam_launch.sh'

    print_info "Launching Cartographer SLAM + Hardware (wait ~30s for init)..."
    launch_terminal "SLAM" "100x20+0+400" "
        docker exec -it $CONTAINER_ID bash /tmp/slam_launch.sh
    "
    sleep 10

    # Terminal 3: Detection
    # C++ sign_detector always runs inside the MPCC launch file.
    # GPU YOLO runs alongside it in separate terminals when GPU is available.
    print_status "Detection: C++ sign_detector (launched with MPCC stack)"
    if [ "$USE_GPU_YOLO" = true ]; then
        print_info "Launching YOLO Bridge (additional GPU detection)..."
        launch_terminal "YOLO-Bridge" "100x20+800+0" "
            sleep 10
            docker exec -it $CONTAINER_ID bash -c '
                source /opt/ros/humble/setup.bash
                source /workspace/cartographer_ws/install/setup.bash
                source /workspaces/isaac_ros-dev/ros2/install/setup.bash
                echo Starting YOLO Bridge...
                ros2 run acc_stage1_mission yolo_bridge
            '
        "

        # Terminal 3b: GPU YOLO Detector (Python 3.10)
        # Allow Docker containers to access the host X server (for annotated frame display)
        xhost +local: > /dev/null 2>&1 || true
        print_info "Launching GPU YOLO Detector (Python 3.10 + CUDA 12.8)..."
        launch_terminal "GPU-YOLO" "100x20+1200+0" "
            sleep 20
            docker exec -it \
                -e DISPLAY=\$DISPLAY \
                $CONTAINER_ID bash -c '
                echo Starting GPU-accelerated YOLO Detector...
                python3.10 /workspaces/isaac_ros-dev/yolo_detector_standalone.py
            '
        "
    fi

    # Terminal 4: C++ MPCC stack (controller + mission manager + sign detector + odom)
    MPCC_LAUNCH_ARGS=""

    # Load preset if specified (e.g., --2025)
    if [ -n "$USE_PRESET" ]; then
        MPCC_LAUNCH_ARGS=$(load_preset "$USE_PRESET") || {
            print_error "Failed to load preset '$USE_PRESET'"
            exit 1
        }
    fi

    # Enable dashboard if requested
    if [ "$USE_DASHBOARD" = true ]; then
        MPCC_LAUNCH_ARGS="$MPCC_LAUNCH_ARGS use_dashboard:=true"
    fi

    # Note: path overlay is launched as a separate terminal (not via launch file)
    # because it needs X11 DISPLAY access for matplotlib GUI

    if [ -n "$USE_PRESET" ]; then
        print_info "Launching C++ MPCC stack (preset: $USE_PRESET)..."
    else
        print_info "Launching C++ MPCC stack (controller + mission + sign detector)..."
    fi
    # Write the launch args to a temp file that the container can read,
    # avoiding double-quote nesting issues with bash -c inside bash -c.
    local mpcc_args_file="/tmp/acc_mpcc_launch_args.txt"
    echo "$MPCC_LAUNCH_ARGS" > "$mpcc_args_file"
    # Also copy into the Docker-mounted workspace so the container can read it
    cp "$mpcc_args_file" "$DOCKER_PKG_DIR/mpcc_launch_args.txt"

    launch_terminal "MPCC-Mission" "100x20+800+400" "
        echo 'Waiting 45 seconds for Nav2 to initialize...'
        sleep 45
        docker exec -it $CONTAINER_ID bash -c '
            source /opt/ros/humble/setup.bash
            source /workspace/cartographer_ws/install/setup.bash
            source /workspaces/isaac_ros-dev/ros2/install/setup.bash
            LAUNCH_ARGS=\$(cat /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/mpcc_launch_args.txt 2>/dev/null)
            echo Starting C++ MPCC stack with args: \$LAUNCH_ARGS
            ros2 launch acc_stage1_mission mpcc_mission_launch.py \$LAUNCH_ARGS
        '
    "

    # Path Overlay Visualizer (opt-in, separate terminal for X11 display)
    if [ "$USE_OVERLAY" = true ]; then
        # Allow Docker containers to access the host X server
        xhost +local: > /dev/null 2>&1 || print_warning "xhost +local: failed — overlay may fall back to headless"
        print_info "Launching Path Overlay visualizer..."
        launch_terminal "Path-Overlay" "100x20+0+800" "
            echo 'Waiting 50 seconds for MPCC stack to publish /plan...'
            sleep 50
            docker exec -it \
                -e DISPLAY=\$DISPLAY \
                $CONTAINER_ID bash -c '
                source /opt/ros/humble/setup.bash
                source /workspace/cartographer_ws/install/setup.bash
                source /workspaces/isaac_ros-dev/ros2/install/setup.bash
                echo Starting Path Overlay...
                echo \"DISPLAY=\$DISPLAY\"
                ros2 run acc_stage1_mission path_overlay
            '
        "
    fi

    echo ""
    print_status "All terminals launched!"
    echo ""
    print_info "Stack:"
    print_info "  C++ MPCC controller  (SQP solver, Eigen + gradient projection)"
    print_info "  C++ mission manager  (road graph A* + state machine)"
    print_info "  C++ sign detector    (HSV contour-based detection)"
    print_info "  C++ state estimator  (EKF: TF + encoders)"
    print_info "  C++ obstacle tracker (Kalman + lidar)"
    if [ "$USE_GPU_YOLO" = true ]; then
        print_info "  GPU YOLO detector    (Python 3.10 + CUDA, alongside C++ HSV)"
    fi
    if [ "$USE_OVERLAY" = true ]; then
        print_info "  Path overlay         (matplotlib bird's-eye track + /plan)"
    fi
    if [ -n "$USE_PRESET" ]; then
        print_info "  Preset: $USE_PRESET"
    fi
    echo ""
    print_info "The mission will start automatically in ~45 seconds"
    echo ""

    # Log files location
    echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  SESSION LOGS                                                 ║${NC}"
    echo -e "${YELLOW}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${YELLOW}║${NC}  Session dir  : ${SESSION_DIR}${YELLOW}${NC}"
    echo -e "${YELLOW}║${NC}  Terminal logs : QCar2-Hardware.log, SLAM-Nav2.log,${YELLOW}${NC}"
    echo -e "${YELLOW}║${NC}                  MPCC-Mission.log, YOLO-Bridge.log${YELLOW}${NC}"
    echo -e "${YELLOW}║${NC}  Node logs    : behavior_*.log, coordinates_*.csv,${YELLOW}${NC}"
    echo -e "${YELLOW}║${NC}                  mpcc_*.csv${YELLOW}${NC}"
    echo -e "${YELLOW}║${NC}                                                           ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  View latest:  ./run_mission.sh --logs                    ${YELLOW}║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    print_warning "To stop all nodes and close terminals:"
    echo "    ./run_mission.sh --stop"
    echo ""
}

show_logs() {
    print_header
    LOG_DIR="$DOCKER_PKG_DIR/logs"

    # Also try to copy node logs from container /tmp/mission_logs
    CONTAINER_ID=$(get_container_id 2>/dev/null)
    if [ -n "$CONTAINER_ID" ]; then
        docker cp "$CONTAINER_ID:/tmp/mission_logs/." "$LOG_DIR/" 2>/dev/null || true
    fi

    if [ ! -d "$LOG_DIR" ]; then
        print_warning "No log directory found at: $LOG_DIR"
        return
    fi

    # Find session directory
    local session_dir=""
    if [ -n "$SHOW_LOGS_SESSION" ]; then
        # User specified a session
        if [ -d "$LOG_DIR/$SHOW_LOGS_SESSION" ]; then
            session_dir="$LOG_DIR/$SHOW_LOGS_SESSION"
        elif [ -d "$LOG_DIR/session_$SHOW_LOGS_SESSION" ]; then
            session_dir="$LOG_DIR/session_$SHOW_LOGS_SESSION"
        else
            print_error "Session not found: $SHOW_LOGS_SESSION"
            print_info "Available sessions:"
            ls -1d "$LOG_DIR"/session_* 2>/dev/null | while read d; do
                echo "    $(basename "$d")"
            done
            return
        fi
    else
        # Find latest session
        session_dir=$(ls -1dt "$LOG_DIR"/session_* 2>/dev/null | head -1)
    fi

    # List all available sessions
    local session_count
    session_count=$(ls -1d "$LOG_DIR"/session_* 2>/dev/null | wc -l)

    echo ""
    if [ -n "$session_dir" ]; then
        print_info "Session: $(basename "$session_dir")"
        if [ "$session_count" -gt 1 ]; then
            print_info "($session_count sessions available — use --logs <session_name> to view others)"
        fi
        echo ""

        # Show session info
        if [ -f "$session_dir/session_info.txt" ]; then
            echo -e "${GREEN}=== Session Info ===${NC}"
            cat "$session_dir/session_info.txt"
            echo ""
        fi

        # Show terminal log files
        local has_terminal_logs=false
        for logfile in "$session_dir"/*.log; do
            [ -f "$logfile" ] || continue
            has_terminal_logs=true
            break
        done

        if [ "$has_terminal_logs" = true ]; then
            echo -e "${GREEN}=== Terminal Logs ===${NC}"
            printf "  %-25s %8s %8s\n" "FILE" "SIZE" "LINES"
            echo "  -----------------------------------------------"
            for logfile in "$session_dir"/*.log; do
                [ -f "$logfile" ] || continue
                local fname=$(basename "$logfile")
                local fsize=$(du -h "$logfile" 2>/dev/null | cut -f1)
                local flines=$(wc -l < "$logfile" 2>/dev/null)
                printf "  %-25s %8s %8s\n" "$fname" "$fsize" "$flines"
            done
            echo ""

            # Show tail of each terminal log
            for logfile in "$session_dir"/*.log; do
                [ -f "$logfile" ] || continue
                local fname=$(basename "$logfile")
                local flines=$(wc -l < "$logfile" 2>/dev/null)
                if [ "$flines" -gt 0 ]; then
                    echo -e "${BLUE}--- ${fname} (last 15 lines) ---${NC}"
                    tail -15 "$logfile"
                    echo ""
                fi
            done
        fi
    fi

    # Show node-level logs (behavior, coordinates, mpcc CSV — written to LOG_DIR root)
    echo -e "${GREEN}=== Node Log Files ===${NC}"

    LATEST_BEHAVIOR=$(ls -t "$LOG_DIR"/behavior_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_BEHAVIOR" ]; then
        local blines=$(wc -l < "$LATEST_BEHAVIOR")
        echo -e "${BLUE}--- Behavior Log ($blines lines): $(basename "$LATEST_BEHAVIOR") ---${NC}"
        tail -20 "$LATEST_BEHAVIOR"
        echo ""
    fi

    LATEST_COORDS=$(ls -t "$LOG_DIR"/coordinates_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_COORDS" ]; then
        local clines=$(wc -l < "$LATEST_COORDS")
        echo -e "${BLUE}--- Coordinate Trace ($clines lines): $(basename "$LATEST_COORDS") ---${NC}"
        head -1 "$LATEST_COORDS"
        echo "  ..."
        tail -10 "$LATEST_COORDS"
        echo ""
    fi

    LATEST_MPCC=$(ls -t "$LOG_DIR"/mpcc_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_MPCC" ]; then
        local mlines=$(wc -l < "$LATEST_MPCC")
        echo -e "${BLUE}--- MPCC CSV ($mlines lines): $(basename "$LATEST_MPCC") ---${NC}"
        head -1 "$LATEST_MPCC"
        echo "  ..."
        tail -10 "$LATEST_MPCC"
        echo ""
    fi

    # If nothing found at all
    if [ -z "$session_dir" ] && [ -z "$LATEST_BEHAVIOR" ] && [ -z "$LATEST_COORDS" ] && [ -z "$LATEST_MPCC" ]; then
        print_warning "No log files found."
        print_info "Logs are created when the mission starts running."
        print_info "Expected location: $LOG_DIR"
    fi
}

# Parse arguments — collect all flags first, then execute actions.
# This allows flags like --report to work regardless of order (e.g. --stop --report or --report --stop).
DO_STOP=false
DO_RESET=false
DO_LOGS=false
DO_HELP=false
while [ $# -gt 0 ]; do
    case "$1" in
        --stop|-s)
            DO_STOP=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --reset|-r)
            DO_RESET=true
            shift
            ;;
        --logs|-l)
            DO_LOGS=true
            # Check if next arg is a session name (not another flag)
            if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                SHOW_LOGS_SESSION="$2"
                shift
            fi
            shift
            ;;
        --help|-h)
            DO_HELP=true
            shift
            ;;
        --no-gpu)
            USE_GPU_YOLO=false
            shift
            ;;
        --2025|--legacy)
            USE_PRESET="legacy_2025"
            shift
            ;;
        --dashboard)
            USE_DASHBOARD=true
            shift
            ;;
        --overlay)
            USE_OVERLAY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

# Execute action flags (mutually exclusive: stop > reset > logs > help > launch)
if [ "$DO_HELP" = true ]; then
    print_header
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  ./run_mission.sh             Launch full stack (auto-detects GPU)"
    echo "  ./run_mission.sh --no-gpu    Launch without GPU YOLO (C++ HSV only)"
    echo "  ./run_mission.sh --2025      Use PolyCtrl 2025 MPCC weights"
    echo "  ./run_mission.sh --dashboard Enable real-time telemetry dashboard"
    echo "  ./run_mission.sh --overlay   Enable path overlay map visualizer"
    echo ""
    echo "  ./run_mission.sh --stop      Stop + reset car to hub"
    echo "  ./run_mission.sh --stop --report  Stop + reset + generate path report PNG"
    echo "  ./run_mission.sh --reset     Reset scenario (sync + rebuild + reset car)"
    echo "  ./run_mission.sh --logs      Show latest session logs"
    echo "  ./run_mission.sh --logs <s>  Show logs for specific session"
    echo ""
    echo "C++ Stack (always launched):"
    echo "  mpcc_controller_node    C++ SQP MPCC controller"
    echo "  mission_manager_node    C++ mission state machine + road graph A*"
    echo "  sign_detector_node      C++ HSV sign/light/cone detection"
    echo "  odom_from_tf_node       C++ TF -> /odom bridge"
    echo "  state_estimator_node    C++ EKF state estimator"
    echo "  obstacle_tracker_node   C++ Kalman + lidar tracker"
    echo "  traffic_light_map_node  C++ spatial traffic light mapping"
    echo ""
    echo "GPU YOLO (auto-detected, alongside C++ HSV):"
    echo "  Requires: nvidia-smi + Python 3.10 + torch CUDA + YOLO model"
    echo "  Disable with --no-gpu if GPU detection causes issues"
    echo ""
    echo "Flags can be combined: ./run_mission.sh --2025 --dashboard"
    exit 0
fi

if [ "$DO_STOP" = true ]; then
    stop_all
    # Reset car to taxi hub spawn after stopping.
    # Use inline Python to (1) zero velocity/steering, (2) teleport to hub.
    # enableDynamics=False freezes the car so it doesn't coast after teleport.
    # Then re-enable dynamics so the next run can drive.
    print_info "Resetting car to taxi hub position..."
    if timeout 10 docker exec virtual-qcar2 python3 -c "
import sys, time
import numpy as np
from quanser.hardware import HIL
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# Zero HIL motors (steering=ch1000, throttle=ch11000)
try:
    card = HIL()
    card.open('qcar2', '0@tcpip://localhost:18960')
    ch = np.array([1000, 11000], dtype=np.uint32)
    card.write_other(ch, 2, np.array([0.0, 0.0], dtype=np.float64))
    card.close()
except: pass

# Teleport to hub, dynamics OFF, then ON
qlabs = QuanserInteractiveLabs()
try:
    qlabs.open('localhost')
except:
    sys.exit(1)
car = QLabsQCar2(qlabs)
car.actorNumber = 0
car.set_transform_and_request_state_degrees(
    location=[-1.205, -0.83, 0.005], rotation=[0, 0, -44.7],
    enableDynamics=False, headlights=False, leftTurnSignal=False,
    rightTurnSignal=False, brakeSignal=False, reverseSignal=False,
    waitForConfirmation=True)
time.sleep(0.3)

# Zero HIL again before re-enabling dynamics
try:
    card = HIL()
    card.open('qcar2', '0@tcpip://localhost:18960')
    ch = np.array([1000, 11000], dtype=np.uint32)
    card.write_other(ch, 2, np.array([0.0, 0.0], dtype=np.float64))
    card.close()
except: pass

car.set_transform_and_request_state_degrees(
    location=[-1.205, -0.83, 0.005], rotation=[0, 0, -44.7],
    enableDynamics=True, headlights=False, leftTurnSignal=False,
    rightTurnSignal=False, brakeSignal=False, reverseSignal=False,
    waitForConfirmation=True)
qlabs.close()
print('Car reset to hub, HIL motors zeroed')
" 2>&1; then
        print_status "Car reset to hub [-1.205, -0.83]"
    else
        print_warning "Car position reset failed or timed out (QLabs may not be running)"
    fi
    if [ "$GENERATE_REPORT" = true ]; then
        generate_report
    fi
    exit 0
fi

if [ "$DO_RESET" = true ]; then
    reset_car
    exit 0
fi

if [ "$DO_LOGS" = true ]; then
    show_logs
    exit 0
fi

# If no action flag was set, launch the mission
launch_mission
