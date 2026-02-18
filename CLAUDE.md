# Quanser ACC Self-Driving Competition - Virtual Stage

## Overview

ROS 2 Humble autonomous driving stack for the Quanser QCar2 (1:10 scale) in
QLabs virtual environment. Built for the ACC 2025/2026 Student Competition.

The system implements a taxi service: Hub -> Pickup -> Dropoff -> Hub, obeying
traffic signs, traffic lights, lane boundaries, and avoiding obstacles.

## Architecture

```
                 +------------------+
                 | mission_manager  |  (Python) State machine: hub->pickup->dropoff->hub
                 +--------+---------+
                          |
              +-----------+-----------+
              |                       |
     +--------v--------+    +--------v---------+
     | road_graph       |    | sign_detector    |  C++ HSV+contour (preferred)
     | (path planning)  |    | or               |
     +--------+---------+    | obstacle_detector|  Python YOLO (fallback)
              |              +--------+----------+
              |                       |
     +--------v---------+   +--------v----------+
     | mpcc_controller   |   | /traffic_control  |
     | (C++ or Python)  |   | _state            |
     +--------+----------+   +-------------------+
              |
     +--------v----------+
     | nav2_qcar_command  |  (C++) Twist -> MotorCommands
     | _convert           |
     +--------+-----------+
              |
     +--------v----------+
     | qcar2_hardware    |  (C++) HIL driver
     +--------------------+
```

### Packages

| Package | Language | Purpose |
|---------|----------|---------|
| `acc_stage1_mission` | Python | Mission logic, MPCC control (Python), detection, path planning |
| `acc_mpcc_controller_cpp` | C++ | C++ MPCC controller + sign detector nodes |
| `qcar2_nodes` | C++ | Hardware interface (motors, cameras, lidar, LEDs) |
| `qcar2_interfaces` | C++ | Custom ROS 2 messages (MotorCommands, BooleanLeds) |
| `qcar2_autonomy` | Python | Quanser reference autonomy (Nav2 client) |

### Key Nodes

| Node | Language | File | Purpose |
|------|----------|------|---------|
| `mission_manager` | Python | `mission_manager.py` | State machine, Nav2/MPCC goal management |
| `mpcc_controller` | Python | `mpcc_controller.py` | MPCC path follower (CasADi or C++ solver) |
| `mpcc_controller_cpp` | C++ | `cpp/mpcc_controller_node.cpp` | C++ MPCC controller with boundary constraints |
| `sign_detector` | C++ | `cpp/sign_detector_node.cpp` | Fast HSV+contour sign/light/cone detection |
| `obstacle_detector` | Python | `obstacle_detector.py` | YOLO-based detection (fallback) |
| `odom_from_tf` | Python | `odom_from_tf.py` | TF -> /odom bridge |

### Topics

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/plan` | Path | mission_manager | mpcc_controller |
| `/cmd_vel_nav` | Twist | mpcc_controller | nav2_qcar_command_convert |
| `/motion_enable` | Bool | sign_detector / obstacle_detector | mpcc_controller, mission_manager |
| `/traffic_control_state` | String (JSON) | sign_detector / obstacle_detector | mpcc_controller |
| `/obstacle_positions` | String (JSON) | obstacle_detector | mpcc_controller_cpp |
| `/mpcc/status` | String | mpcc_controller | mission_manager |
| `/mission/hold` | Bool | mission_manager | mpcc_controller |
| `/camera/color_image` | Image | rgbd (qcar2_nodes) | sign_detector, obstacle_detector |

## Coordinate Systems

- **QLabs frame**: World coordinates used in Setup_Real_Scenario.py
- **Map frame**: Cartographer SLAM frame (origin at car's initial position)
- **Transform**: Translate by car spawn (-1.205, -0.83), rotate by 0.7177 rad

Key locations (QLabs frame):
- Hub: (-1.205, -0.83), heading -44.7 deg
- Pickup: (0.125, 4.395)
- Dropoff: (-0.905, 0.800)

## Competition Rules (Key Infractions)

| Infraction | Stars Lost |
|-----------|-----------|
| Minor lane departure (<3s, <1 car width) | -1 |
| Major lane departure (3-6s) | -2 |
| Disqualifying lane departure (>6s or >1 width) | -5 |
| Incomplete stop at stop sign | -2 |
| Stopping over stop line | -1 |
| Running red light | -2 |
| Cone collision | -2 |
| Failure to yield | -2 |

## Building

```bash
# Inside Isaac ROS container
cd /workspaces/isaac_ros-dev/ros2

# Build Python package
colcon build --packages-select acc_stage1_mission
source install/setup.bash

# Build C++ nodes (MPCC controller + sign detector)
colcon build --packages-select acc_mpcc_controller_cpp
source install/setup.bash

# Build C++ MPCC solver shared library (for Python ctypes binding)
cd src/acc_stage1_mission/cpp
bash build.sh
```

## Running

```bash
# Terminal 1: Launch hardware + SLAM + Nav2
ros2 launch qcar2_nodes virtual_sim.launch.py

# Terminal 2: Launch mission (Python MPCC + optional C++ controller/sign detector)
ros2 launch acc_stage1_mission mpcc_mission_launch.py

# With C++ controller:
ros2 launch acc_stage1_mission mpcc_mission_launch.py use_cpp_controller:=true

# With C++ sign detector:
ros2 launch acc_stage1_mission mpcc_mission_launch.py use_cpp_sign_detector:=true

# Terminal 3: Launch Python YOLO obstacle detector (if not using C++ sign detector)
ros2 run acc_stage1_mission obstacle_detector

# Terminal 4: QLabs scenario setup
python3 Setup_Real_Scenario_Interleaved.py
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/mission.yaml` | Mission waypoints, transforms, timing |
| `config/road_boundaries.yaml` | Road segments, traffic controls, obstacle zones |

## MPCC Tuning

Critical parameters (same defaults in Python `MPCCConfig` and C++ `mpcc::Config`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `contour_weight` | 25.0 | Lateral deviation penalty (**PRIMARY** - must be > lag_weight) |
| `lag_weight` | 5.0 | Progress tracking (secondary to lane keeping) |
| `reference_velocity` | 0.35 | Target speed (m/s) - reduced for safe cornering |
| `boundary_weight` | 30.0 | Road boundary soft constraint penalty |
| `max_velocity` | 0.40 | Hard speed limit (m/s) |
| `steering_rate_weight` | 4.0 | Smooth steering (prevents oscillation) |

Curvature-adaptive speed: `v_ref = reference_velocity * exp(-1.2 * |curvature|)`

## Root Cause Analysis (Feb 2025)

### Lane Violations
- **Cause**: contour_weight (8) < lag_weight (15) prioritized progress over lane keeping
- **Fix**: Inverted ratio to contour=25, lag=5; reduced velocity; increased boundary_weight

### Sign Detection Failures
- **Cause**: Detection persistence threshold too high (5 frames); global 15s stop sign cooldown; traffic light state machine fragile
- **Fix**: Persistence=2 frames; spatial per-sign cooldown (5s); traffic light timeout (15s) + no-red expiry (2s)

## Known Issues & Debug Tips

- Check `/mpcc/status` for controller state
- Check `/traffic_control_state` for sign/light detection
- Check `/motion_enable` for obstacle detector output
- Mission logs written to `logs/` directory
- If vehicle oscillates, reduce `reference_velocity` and increase `steering_rate_weight`
- If vehicle cuts corners, increase `contour_weight` relative to `lag_weight`
- C++ sign detector uses HSV color thresholds tuned for QLabs - may need adjustment for real hardware
