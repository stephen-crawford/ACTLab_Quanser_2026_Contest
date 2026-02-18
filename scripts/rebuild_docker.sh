#!/bin/bash
# =============================================================================
# Docker Rebuild Script for ACC Stage 1 Mission
# =============================================================================
# This script rebuilds the Docker container and/or ROS2 packages.
#
# Usage:
#   ./rebuild_docker.sh              # Rebuild ROS2 packages only
#   ./rebuild_docker.sh --full       # Full Docker image rebuild
#   ./rebuild_docker.sh --clean      # Clean build (remove build artifacts)
#   ./rebuild_docker.sh --help       # Show help
#
# Run this script from the HOST machine (not inside Docker).
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
ROS2_SRC_DIR="$( cd "$PACKAGE_DIR/.." && pwd )"
ROS2_WS_DIR="$( cd "$ROS2_SRC_DIR/.." && pwd )"
ACC_DEV_DIR="$( cd "$ROS2_WS_DIR/../.." && pwd )"

# Docker paths
ISAAC_ROS_COMMON="$ACC_DEV_DIR/isaac_ros_common"
DOCKER_SCRIPTS="$ISAAC_ROS_COMMON/scripts"

# Container settings
PLATFORM="$(uname -m)"
CONTAINER_NAME="isaac_ros_dev-${PLATFORM}-container"
IMAGE_NAME="isaac_ros_dev-${PLATFORM}"

# Default options
FULL_REBUILD=false
CLEAN_BUILD=false
VERBOSE=false

function print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function show_help() {
    echo "Docker Rebuild Script for ACC Stage 1 Mission"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --full       Full Docker image rebuild (rebuilds all layers)"
    echo "  --clean      Clean build (remove build/install/log directories)"
    echo "  --verbose    Show verbose output"
    echo "  --help       Show this help message"
    echo ""
    echo "Paths:"
    echo "  Package:     $PACKAGE_DIR"
    echo "  ROS2 WS:     $ROS2_WS_DIR"
    echo "  ACC Dev:     $ACC_DEV_DIR"
    echo "  Docker:      $DOCKER_SCRIPTS"
    echo ""
    echo "Examples:"
    echo "  $0                 # Quick rebuild of ROS2 packages"
    echo "  $0 --clean         # Clean rebuild of ROS2 packages"
    echo "  $0 --full          # Full Docker image rebuild"
    echo "  $0 --full --clean  # Full rebuild with clean packages"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_REBUILD=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_header "ACC Stage 1 Mission - Docker Rebuild"

# Verify we're not inside Docker
if [ -f /.dockerenv ]; then
    print_error "This script should be run from the HOST, not inside Docker!"
    exit 1
fi

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Verify paths exist
if [ ! -d "$ACC_DEV_DIR" ]; then
    print_error "ACC Development directory not found: $ACC_DEV_DIR"
    exit 1
fi

if [ ! -d "$DOCKER_SCRIPTS" ]; then
    print_error "Docker scripts directory not found: $DOCKER_SCRIPTS"
    exit 1
fi

print_info "Platform: $PLATFORM"
print_info "Container: $CONTAINER_NAME"
print_info "Image: $IMAGE_NAME"
print_info "ROS2 Workspace: $ROS2_WS_DIR"

# ============================================================================
# Full Docker Image Rebuild
# ============================================================================
if [ "$FULL_REBUILD" = true ]; then
    print_header "Full Docker Image Rebuild"

    # Stop any running container
    if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_info "Stopping running container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" || true
    fi

    # Remove the container
    if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_info "Removing container: $CONTAINER_NAME"
        docker rm "$CONTAINER_NAME" || true
    fi

    # Read the config to get the image key
    CONFIG_FILE="$DOCKER_SCRIPTS/.isaac_ros_common-config"
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        IMAGE_KEY="${CONFIG_IMAGE_KEY:-ros2_humble.user}"
        print_info "Image key from config: $IMAGE_KEY"
    else
        IMAGE_KEY="ros2_humble.ros_cartographer.user.quanser.python310"
        print_warn "Config file not found, using default: $IMAGE_KEY"
    fi

    # Build the Docker image
    print_info "Building Docker image with key: $IMAGE_KEY"
    cd "$DOCKER_SCRIPTS"

    if [ "$VERBOSE" = true ]; then
        ./build_base_image.sh "$IMAGE_KEY" "$IMAGE_NAME"
    else
        ./build_base_image.sh "$IMAGE_KEY" "$IMAGE_NAME" 2>&1 | tee /tmp/docker_build.log
    fi

    print_info "Docker image rebuild complete!"
fi

# ============================================================================
# ROS2 Package Rebuild (inside container)
# ============================================================================
print_header "ROS2 Package Rebuild"

# Check if container is running
CONTAINER_RUNNING=false
if docker ps -q --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q .; then
    CONTAINER_RUNNING=true
    print_info "Using running container: $CONTAINER_NAME"
fi

# Build command to run inside container
BUILD_CMD=""

if [ "$CLEAN_BUILD" = true ]; then
    BUILD_CMD+="echo 'Cleaning build artifacts...' && "
    BUILD_CMD+="cd /workspaces/isaac_ros-dev && "
    BUILD_CMD+="rm -rf build install log && "
fi

# Fix Python version mismatch: ROS2 Humble uses Python 3.8, but container may have Python 3.10
# Install empy for whatever Python version CMake picks up
BUILD_CMD+="cd /workspaces/isaac_ros-dev && "
BUILD_CMD+="source /opt/ros/humble/setup.bash && "
BUILD_CMD+="echo 'Fixing Python dependencies...' && "
BUILD_CMD+="python3 -m pip install --user empy==3.3.4 lark catkin_pkg 2>/dev/null || true && "
BUILD_CMD+="/usr/local/bin/python3.10 -m pip install --user empy==3.3.4 lark catkin_pkg 2>/dev/null || true && "
BUILD_CMD+="echo 'Building ROS2 packages...' && "
BUILD_CMD+="colcon build --symlink-install --packages-select acc_stage1_mission qcar2_interfaces qcar2_nodes 2>&1 && "
BUILD_CMD+="echo 'Build complete!' && "
BUILD_CMD+="source install/setup.bash && "
BUILD_CMD+="echo 'Packages installed successfully'"

if [ "$CONTAINER_RUNNING" = true ]; then
    # Run build in existing container
    print_info "Running build in container..."
    docker exec -it -u admin "$CONTAINER_NAME" bash -c "$BUILD_CMD"
else
    # Start a new container for the build
    print_info "Starting new container for build..."

    # Docker run arguments
    DOCKER_ARGS=(
        "-it"
        "--rm"
        "--privileged"
        "--network" "host"
        "-v" "/tmp/.X11-unix:/tmp/.X11-unix"
        "-v" "$HOME/.Xauthority:/home/admin/.Xauthority:rw"
        "-e" "DISPLAY"
        "-e" "NVIDIA_VISIBLE_DEVICES=all"
        "-e" "NVIDIA_DRIVER_CAPABILITIES=all"
        "-v" "$ROS2_WS_DIR:/workspaces/isaac_ros-dev"
        "-v" "/etc/localtime:/etc/localtime:ro"
        "--runtime" "nvidia"
        "--user" "admin"
        "--workdir" "/workspaces/isaac_ros-dev"
    )

    docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash -c "$BUILD_CMD"
fi

print_header "Build Complete!"
print_info "To run the container interactively:"
print_info "  cd $DOCKER_SCRIPTS && ./run_dev.sh $ROS2_WS_DIR"
print_info ""
print_info "To launch the MPCC mission inside the container:"
print_info "  ros2 launch acc_stage1_mission mpcc_mission_launch.py"
