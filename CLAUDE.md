# Quanser ACC Self-Driving Competition - Virtual Stage

## Overview

ROS 2 Humble autonomous driving stack for the Quanser QCar2 (1:10 scale) in
QLabs virtual environment. Built for the ACC 2025/2026 Student Competition.

The system implements a taxi service: Hub -> Pickup -> Dropoff -> Hub, obeying
traffic signs, traffic lights, lane boundaries, and avoiding obstacles.

## Architecture

```
                 +-------------------+
                 | mission_manager   |  C++ (default) or Python fallback
                 +--------+----------+
                          |
              +-----------+-----------+
              |                       |
     +--------v--------+    +--------v---------+
     | road_graph       |    | sign_detector    |  C++ HSV+contour (default)
     | (path planning)  |    | or               |
     +--------+---------+    | obstacle_detector|  Python modular backends (fallback)
              |              +--------+----------+
              |                       |
     +--------v---------+   +--------v----------+
     | mpcc_controller   |   | /traffic_control  |
     | C++ SQP (default)|   | _state            |
     +--------+----------+   +-------------------+
              |
              | MotorCommands (direct: steering_angle + motor_throttle)
              | (bypasses nav2_qcar_command_convert)
              |
     +--------v----------+
     | qcar2_hardware    |  (C++) HIL driver + built-in PID
     +--------------------+
```

Module configuration: `config/modules.yaml` (detection, path planning, controller backends)

### Packages

| Package | Language | Purpose |
|---------|----------|---------|
| `acc_stage1_mission` | Python | Mission logic, MPCC control (fallback), detection (modular), path planning |
| `acc_mpcc_controller_cpp` | C++ | MPCC controller, sign detector, mission manager, odom_from_tf (default stack) |
| `qcar2_nodes` | C++ | Hardware interface (motors, cameras, lidar, LEDs) |
| `qcar2_interfaces` | C++ | Custom ROS 2 messages (MotorCommands, BooleanLeds) |
| `qcar2_autonomy` | Python | Quanser reference autonomy (Nav2 client) |

### Key Nodes

| Node | Language | File | Purpose |
|------|----------|------|---------|
| `mission_manager` | C++/Python | `cpp/mission_manager_node.cpp` / `mission_manager.py` | State machine, path planning (C++ default) |
| `mpcc_controller` | C++/Python | `cpp/mpcc_controller_node.cpp` / `mpcc_controller.py` | MPCC path follower (C++ SQP default) |
| `sign_detector` | C++ | `cpp/sign_detector_node.cpp` | Fast HSV+contour sign/light/cone detection (default) |
| `obstacle_detector` | Python | `obstacle_detector.py` | Modular detection backends (fallback when C++ sign detector disabled) |
| `odom_from_tf` | C++/Python | `cpp/odom_from_tf_node.cpp` / `odom_from_tf.py` | TF -> /odom bridge (C++ default) |
| `path_overlay` | Python | `path_overlay.py` | Bird's-eye path + vehicle visualizer (opt-in: `--overlay`) |

### Topics

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/plan` | Path | mission_manager | mpcc_controller |
| `/qcar2_motor_speed_cmd` | MotorCommands | mpcc_controller (direct mode) | qcar2_hardware |
| `/cmd_vel_nav` | Twist | mpcc_controller (legacy/debug) | nav2_qcar_command_convert |
| `/qcar2_joint` | JointState | qcar2_hardware | mpcc_controller (encoder velocity) |
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
# Recommended: use run_mission.sh (handles all terminals, builds, and Docker)
./run_mission.sh             # Full C++ stack (auto-detects GPU for YOLO)
./run_mission.sh --no-gpu    # C++ HSV detection only (skip GPU YOLO)
./run_mission.sh --2025      # Use PolyCtrl 2025 MPCC weights for comparison
./run_mission.sh --overlay   # Show bird's-eye path overlay (planned path + vehicle)
./run_mission.sh --stop      # Stop all nodes
./run_mission.sh --reset     # Sync code, rebuild, reset car

# Manual launch (inside Isaac ROS container):
# Terminal 1: Launch hardware + SLAM + Nav2
ros2 launch qcar2_nodes virtual_sim.launch.py

# Terminal 2: Launch MPCC mission (all C++ nodes)
ros2 launch acc_stage1_mission mpcc_mission_launch.py

# Terminal 3: QLabs scenario setup
python3 Setup_Real_Scenario_Interleaved.py
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/modules.yaml` | Module backend selection (detection, path planning, controller) |
| `config/mission.yaml` | Mission waypoints, transforms, timing |
| `config/road_boundaries.yaml` | Road segments, traffic controls, obstacle zones |

## MPCC Tuning

Critical parameters (same defaults in Python `MPCCConfig` and C++ `mpcc::Config`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `contour_weight` | 8.0 | Lateral deviation penalty (lane keeping) |
| `lag_weight` | 12.0 | Progress tracking (**PRIMARY** - must be > contour_weight, ref: 7.0) |
| `velocity_weight` | 15.0 | Velocity reference tracking (ref: R_ref=17.0) |
| `reference_velocity` | 0.65 | Target speed (m/s) - ref uses 0.70 |
| `boundary_weight` | 20.0 | Road boundary soft constraint penalty |
| `max_velocity` | 1.2 | Hard speed limit (m/s) - ref uses 2.0; qcar2_hardware PD controls actual speed |
| `use_direct_motor` | true | Bypass nav2_qcar_command_convert, publish MotorCommands directly |
| `steering_rate_weight` | 3.0 | Smooth steering (prevents oscillation) |

Curvature-adaptive speed: `v_ref = reference_velocity * exp(-0.4 * |curvature|)` (matched to reference)

## Root Cause Analysis (Feb 2025-2026)

### Vehicle Not Following Path (Feb 2026)
- **Cause**: contour_weight (25) >> lag_weight (5) + low velocity_weight (2.0) made solver find "staying still in lane" optimal; curvature decay -1.2 was 3x too aggressive vs reference -0.4
- **Fix**: Aligned with reference (PolyCtrl 2025): contour=8, lag=12, velocity_weight=15, curvature decay -0.4, max_velocity 1.2, reference_velocity 0.65

### Lane Violations (Feb 2025)
- **Cause**: contour_weight (8) < lag_weight (15) prioritized progress over lane keeping
- **Fix**: Balanced ratio with contour=8, lag=12 (progress-first but with lane keeping)

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
