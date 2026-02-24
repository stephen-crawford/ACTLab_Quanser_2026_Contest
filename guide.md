# ACC Stage 1 Mission - Setup & Run Guide

## Overview

This package provides the mission manager for the Quanser ACC Virtual ROS Stage I contest. The mission navigates: **Hub → Pickup → Dropoff → Hub**.

---

## Quick Start (Recommended)

### Option A: Using Helper Scripts

**From Host (Ubuntu 24.04):**
```bash
cd /home/$USER/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission/scripts

# Setup containers (opens QLabs first)
./run_stage1.sh

# Or setup + run scenario automatically
./run_stage1.sh --scenario
```

**Then in DEV container:**
```bash
cd /workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/scripts

# Mission only (no obstacle detection)
./run_stage1_dev.sh

# With traffic system detector (CPU-based)
./run_stage1_dev.sh --detect

# With YOLO detector (requires GPU)
./run_stage1_dev.sh --yolo
```

### Option B: Manual Setup (Step-by-Step)

Follow the detailed steps below.

---

## Step 1: Open QLabs

Launch QLabs and open **Plane World**.

---

## Step 2: Start ENV Container (Host Terminal)

```bash
cd /home/$USER/Documents/ACC_Development/docker/development_docker/quanser_dev_docker_files

# Start the container
sudo docker run --rm -it --network host --name virtual-qcar2 quanser/virtual-qcar2 bash
```

**Additional ENV terminals:**
```bash
docker exec -it virtual-qcar2 bash
```

---

## Step 3: Start DEV Container (New Host Terminal)

```bash
docker context use default

cd /home/$USER/Documents/ACC_Development/isaac_ros_common

./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
```

**Additional DEV terminals:**
```bash
cd /home/$USER/Documents/ACC_Development/isaac_ros_common
./scripts/run_dev.sh /home/$USER/Documents/ACC_Development/Development
```

---

## Step 4: Run Setup Scenario (ENV Container)

In the ENV container shell:
```bash
python3 /home/qcar2_scripts/python/Base_Scenarios_Python/Setup_Competition_Map.py 2>/dev/null || \
python3 /home/qcar2_scripts/python/Setup_Competition_Map.py
```

---

## Step 5: Build & Source (DEV Container)

```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

cd /workspaces/isaac_ros-dev/ros2
colcon build --packages-select acc_stage1_mission --symlink-install
source install/setup.bash
```

---

## Step 6: Launch SLAM & Navigation (DEV Container - Terminal 1)

```bash
ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
```

Wait ~20-30 seconds for Nav2 to fully initialize.

---

## Step 7: Launch Mission (DEV Container - Terminal 2)

**Basic launch:**
```bash
ros2 launch acc_stage1_mission mission_launch.py
```

**With parameters:**
```bash
ros2 launch acc_stage1_mission mission_launch.py \
    goal_timeout_s:=180.0 \
    max_retries_per_leg:=5 \
    enable_obstacle_detection:=true
```

---

## Step 8: Launch Detector (DEV Container - Terminal 3, Optional)

**YOLO detector (GPU required):**
```bash
ros2 run qcar2_autonomy yolo_detector
```

**Traffic system detector (CPU-based):**
```bash
ros2 run qcar2_autonomy traffic_system_detector
```

---

## Configuration

### Mission Waypoints

Edit `config/mission.yaml` to set waypoint coordinates:

```yaml
pickup:
  x: 0.125
  y: 4.395
  yaw: 0.0

dropoff:
  x: -0.905
  y: 0.800
  yaw: 0.0

hub:
  x: 0.0
  y: 0.0
  yaw: 0.0
```

### Finding Coordinates

1. Launch Nav2
2. Run pose logger: `ros2 run acc_stage1_mission pose_logger`
3. Use RViz2 "2D Goal Pose" to drive QCar2 to each location
4. Copy the logged coordinates to `mission.yaml`

### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_file` | mission.yaml | Path to mission config |
| `use_tf_hub` | true | Capture hub position from TF at startup |
| `hub_tf_timeout_s` | 15.0 | TF lookup timeout |
| `goal_timeout_s` | 120.0 | Navigation goal timeout |
| `max_retries_per_leg` | 3 | Retries before recovery strategy |
| `goal_tol_m` | 0.35 | Goal tolerance in meters |
| `enable_obstacle_detection` | true | Subscribe to /motion_enable |
| `obstacle_pause_timeout_s` | 30.0 | Max pause time for obstacles |
| `enable_led` | true | Control QCar2 LEDs from mission state |
| `backup_distance_m` | 0.15 | Recovery backup distance |
| `backup_speed` | 0.1 | Recovery backup speed (m/s) |

---

## Debugging & Monitoring

### Check ROS2 Status
```bash
# Reset discovery if stale
ros2 daemon stop && ros2 daemon start

# List active nodes
ros2 node list

# List topics with types
ros2 topic list -t
```

### Monitor Mission State
```bash
# Mission status
ros2 topic echo /mission_state

# Navigation feedback
ros2 topic echo /navigate_to_pose/_action/feedback

# Obstacle detection
ros2 topic echo /motion_enable
```

### Pose Logging
```bash
ros2 run acc_stage1_mission pose_logger
```

---

## File Structure

```
acc_stage1_mission/
├── acc_stage1_mission/
│   ├── mission_manager.py     # Main mission state machine
│   └── pose_logger.py         # Utility for finding coordinates
├── config/
│   └── mission.yaml           # Waypoint configuration
├── launch/
│   ├── mission_launch.py      # Standard launch file
│   └── mission_with_detection_launch.py  # Launch with detector
├── scripts/
│   ├── run_stage1.sh          # Host orchestration script
│   └── run_stage1_dev.sh      # DEV container script
└── guide.md                   # This file
```

---

## Troubleshooting

### "No transform from base_link to map"
- Wait for SLAM/Nav2 to fully initialize (~25s)
- Check that Cartographer is publishing TF

### Navigation goals timeout
- Increase `goal_timeout_s`
- Check costmap for obstacles blocking path
- Verify waypoint coordinates are reachable

### Mission stuck at pickup/dropoff
- Check `dwell_s` setting in mission.yaml
- Verify goal tolerance vs Nav2 settings

### Detector not working
- YOLO requires GPU - use traffic_system_detector on CPU
- Check camera topic: `ros2 topic echo /camera/color_image --once`
