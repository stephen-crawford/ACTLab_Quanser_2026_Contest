#!/bin/bash
# Build the MPCC C++ solver shared library and optionally the ROS2 C++ node
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building MPCC C++ solver..."

# Detect Eigen include path
EIGEN_INCLUDE=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

# Compile to shared library
g++ -shared -fPIC -O2 -march=native \
    $EIGEN_INCLUDE \
    -std=c++17 \
    -o libmpcc_solver.so \
    mpcc_solver.cpp

echo "Built: $SCRIPT_DIR/libmpcc_solver.so"
echo "Size: $(du -h libmpcc_solver.so | cut -f1)"

# Quick sanity check: verify symbols exist
if nm -D libmpcc_solver.so | grep -q mpcc_create; then
    echo "Symbols verified OK"
else
    echo "ERROR: Expected symbols not found!"
    exit 1
fi

echo ""
echo "To build the C++ ROS2 nodes (MPCC controller + sign detector), run:"
echo "  cd <ros2_workspace> && colcon build --packages-select acc_mpcc_controller_cpp"
echo "Then run with:"
echo "  ros2 run acc_mpcc_controller_cpp mpcc_controller_node"
echo "  ros2 run acc_mpcc_controller_cpp sign_detector_node"
