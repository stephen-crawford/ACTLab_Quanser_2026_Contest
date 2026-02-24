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
#   ./run_mission.sh --stop      - Stop all nodes
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
        --exclude='logs/*.log' \
        --exclude='logs/*.csv' \
        --exclude='logs/session_*' \
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

    # Wait for container to initialize and verify it's running
    print_info "Waiting for simulation scenario to initialize..."
    sleep 3
    if ! docker ps --format '{{.Names}}' | grep -q "virtual-qcar2"; then
        print_error "Simulation container exited - ensure QLabs is running with Plane world selected"
        print_info "Check logs with: docker logs virtual-qcar2"
        exit 1
    fi

    # Wait for virtual hardware ports to be available (QLabs needs time to spawn QCar2)
    print_info "Waiting for virtual hardware ports (QCar2 HIL on port 18960)..."
    local port_wait=0
    local port_max=60
    while ! ss -tln 2>/dev/null | grep -q ":18960 " && ! nc -z localhost 18960 2>/dev/null; do
        if [ $port_wait -ge $port_max ]; then
            print_warning "Virtual hardware ports not detected after ${port_max}s - proceeding anyway"
            break
        fi
        sleep 2
        port_wait=$((port_wait + 2))
        printf "\r${BLUE}[i]${NC} Waiting for simulation hardware... %ds" "$port_wait"
    done
    echo ""
    print_status "Simulation container started - $SCENARIO_SCRIPT active"
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

        # Rebuild C++ nodes
        docker exec "$CONTAINER_ID" bash -c '
            source /opt/ros/humble/setup.bash
            cd /workspaces/isaac_ros-dev/ros2
            ln -sfn /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/cpp src/acc_mpcc_controller_cpp
            colcon build --packages-select acc_mpcc_controller_cpp \
                --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 2>&1 | tail -10
        ' && print_status "C++ nodes rebuilt" || print_warning "C++ build had warnings"
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

    # Step 1: Kill ROS2 processes inside the container
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
    else
        print_warning "Dev container not running"
    fi

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
    # IMPORTANT: Only use exact hyphenated names we assigned in launch_terminal().
    # Do NOT use partial matches (e.g. "QCar2", "SLAM") — they could close
    # Quanser Interactive Labs or other unrelated windows.
    # Uses regex anchors (^...$) for exact matching to avoid substring collisions.
    if command -v xdotool &> /dev/null; then
        for title in "QCar2-Hardware" "SLAM-Nav2" "YOLO-Bridge" "GPU-YOLO" "Obstacle-Detector" "MPCC-Mission" "Mission-Manager" "Path-Overlay"; do
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
        for title in "QCar2-Hardware" "SLAM-Nav2" "YOLO-Bridge" "GPU-YOLO" "Obstacle-Detector" "MPCC-Mission" "Mission-Manager" "Path-Overlay"; do
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

    # Reset the scenario so we start from a clean state (car at start, signs/lights reset)
    print_info "Resetting scenario for fresh run..."
    docker exec virtual-qcar2 pkill -f "python3.*Setup_" 2>/dev/null || true
    sleep 1
    docker exec -d virtual-qcar2 bash -c \
        "cd /home/qcar2_scripts/python && python3 Base_Scenarios_Python/$SCENARIO_SCRIPT"
    print_info "Waiting for scenario to initialize..."
    sleep 5
    if docker exec virtual-qcar2 pgrep -f "$SCENARIO_SCRIPT" > /dev/null 2>&1; then
        print_status "Scenario initialized: $SCENARIO_SCRIPT"
    else
        print_warning "Scenario script may have exited - check QLabs"
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
    print_info "Building C++ nodes (acc_mpcc_controller_cpp)..."
    docker exec "$CONTAINER_ID" bash -c '
        source /opt/ros/humble/setup.bash
        cd /workspaces/isaac_ros-dev/ros2
        ln -sfn /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/cpp src/acc_mpcc_controller_cpp
        colcon build --packages-select acc_mpcc_controller_cpp \
            --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 2>&1
    '
    if [ $? -eq 0 ]; then
        print_status "C++ nodes built successfully"
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

    # Terminal 1: QCar2 Hardware Nodes (with retry if sim not ready)
    print_info "Launching QCar2 hardware nodes..."
    launch_terminal "QCar2-Hardware" "100x20+0+0" "
        docker exec -it $CONTAINER_ID bash -c '
            source /opt/ros/humble/setup.bash
            source /workspace/cartographer_ws/install/setup.bash
            source /workspaces/isaac_ros-dev/ros2/install/setup.bash
            echo Starting QCar2 hardware nodes...
            for attempt in 1 2 3; do
                echo [Attempt \$attempt/3] Launching hardware nodes...
                ros2 launch qcar2_nodes qcar2_virtual_launch.py &
                HW_PID=\$!
                sleep 5
                if kill -0 \$HW_PID 2>/dev/null; then
                    echo Hardware nodes running
                    wait \$HW_PID
                    break
                else
                    echo Hardware nodes exited early - simulation may still be starting
                    if [ \$attempt -lt 3 ]; then
                        echo Waiting 10s before retry...
                        sleep 10
                    fi
                fi
            done
        '
    "
    sleep 5

    # Terminal 2: SLAM and Navigation
    # Wait for hardware nodes to establish sensor connections and start publishing
    print_info "Waiting for hardware sensor data to start flowing (10s)..."
    sleep 10
    print_info "Launching SLAM and Navigation (wait ~25s for full init)..."
    launch_terminal "SLAM-Nav2" "100x20+0+400" "
        docker exec -it $CONTAINER_ID bash -c '
            source /opt/ros/humble/setup.bash
            source /workspace/cartographer_ws/install/setup.bash
            source /workspaces/isaac_ros-dev/ros2/install/setup.bash
            echo Starting SLAM and Navigation...
            echo Please wait ~25 seconds for Nav2 to fully initialize...
            ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
        '
    "
    sleep 5

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

# Parse arguments (supports combining flags)
while [ $# -gt 0 ]; do
    case "$1" in
        --stop|-s)
            stop_all
            exit 0
            ;;
        --reset|-r)
            reset_car
            exit 0
            ;;
        --logs|-l)
            # Check if next arg is a session name (not another flag)
            if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                SHOW_LOGS_SESSION="$2"
                shift
            fi
            show_logs
            exit 0
            ;;
        --help|-h)
            print_header
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  ./run_mission.sh             Launch full stack (auto-detects GPU)"
            echo "  ./run_mission.sh --no-gpu    Launch without GPU YOLO (C++ HSV only)"
            echo "  ./run_mission.sh --2025      Use PolyCtrl 2025 MPCC weights"
            echo "  ./run_mission.sh --dashboard Enable real-time telemetry dashboard"
            echo "  ./run_mission.sh --overlay   Enable path overlay map visualizer"
            echo ""
            echo "  ./run_mission.sh --stop      Stop all mission terminals"
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

# If we didn't exit from --stop/--reset/--logs/--help, launch the mission
launch_mission
