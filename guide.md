# ACC Mission - Setup & Run Guide (Current)

## Overview

This repository runs the current C++ MPCC mission stack for the Quanser ACC virtual contest.  
Mission sequence: **Hub -> Pickup -> Dropoff -> Hub**.

Authoritative references:
- `README.md` (active run/build commands)
- `docs/DEBUG_BASELINE_LOCK.md` (locked debugging baseline)

---

## Quick Start (Recommended)

### Option A: Current One-Command Flow (Recommended)

Run from repo root:
```bash
./run_mission.sh
./run_mission.sh --no-gpu
./run_mission.sh --overlay
./run_mission.sh --dashboard
./run_mission.sh --stop
./run_mission.sh --reset
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

## Step 6: Launch SLAM + Hardware (DEV Container - Terminal 1)

```bash
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py
```

Wait ~20-30 seconds for SLAM and sensors to initialize.

---

## Step 7: Launch MPCC Mission (DEV Container - Terminal 2)

```bash
ros2 launch acc_stage1_mission mpcc_mission_launch.py
```

## Step 8: Optional Visualization (Host/DEV)

Use `./run_mission.sh --overlay` or `./run_mission.sh --dashboard` for additional diagnostics.

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

1. Launch SLAM + mission stack
2. Run pose logger: `ros2 run acc_stage1_mission pose_logger`
3. Use RViz2 "2D Goal Pose" to drive QCar2 to each location
4. Copy the logged coordinates to `mission.yaml`

### Mission Parameters

Primary MPCC runtime parameters are loaded from `config/presets/default.yaml` via `run_mission.sh`.

Current baseline (deployment-tuned):
- `reference_velocity: 0.45`
- `contour_weight: 8.0`
- `lag_weight: 15.0`
- `horizon: 10`
- `boundary_weight: 0.0`

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

### Monitor MPCC / Mission State
```bash
# MPCC status
ros2 topic echo /mpcc/status

# Traffic/sign decisions
ros2 topic echo /traffic_control_state

# Obstacle detection
ros2 topic echo /motion_enable

# Planned path
ros2 topic echo /plan
```

### Pose Logging
```bash
ros2 run acc_stage1_mission pose_logger
```

---

## File Structure (Current)

```
quanser-acc/
├── cpp/                       # C++ controller/mission nodes
├── config/presets/            # MPCC preset parameters
├── launch/mpcc_mission_launch.py
├── run_mission.sh             # Primary launch/stop/reset entrypoint
├── README.md                  # Main project documentation
└── docs/DEBUG_BASELINE_LOCK.md
```

---

## Troubleshooting

### "No transform from base_link to map"
- Wait for Cartographer to initialize (~25s)
- Check TF tree and odometry bridge nodes

### Oscillation / oversteer
- Use the triage order in `docs/DEBUG_BASELINE_LOCK.md`
- Verify active params from preset and launch logs before retuning

### Detector not working
- YOLO bridge requires GPU availability
- Check camera topic: `ros2 topic echo /camera/color_image --once`
