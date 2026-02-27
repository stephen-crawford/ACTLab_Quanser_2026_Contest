# ACC Self-Driving Stack — Quanser QCar2

Autonomous driving stack for the [ACC 2025/2026 Student Self-Driving Car Competition](https://quanser.github.io/student-competitions/), running on a 1:10 scale Quanser QCar2 in the QLabs virtual environment.

The vehicle operates as a taxi: **Hub &rarr; Pickup &rarr; Dropoff &rarr; Hub**, obeying stop signs, yield signs, traffic lights, staying in-lane, and avoiding cones and pedestrians.

## Why a Modular Stack

The code features a modular design with interchangeable components to allow for testing and changes based on the scenario. Each subsystem: detection, path planning, control, state estimation, and obstacle tracking, can be swapped across a variety of options via YAML configuration. Using this design, a variety of tests were run to determine the best complete stack. 


The result is a C++ real-time stack (7 nodes; controller runs at `dt=0.1s` / 10 Hz) which shares interfaces, configuration, and mission logic.

## Architecture

```
Camera ─────► sign_detector (C++ HSV)  ──► /motion_enable
              or                            /traffic_control_state
              

Lidar ──────► obstacle_tracker (C++ Kalman) ──► /obstacle_positions

TF (SLAM) ──► odom_from_tf (C++) ──► /odom
              state_estimator (C++ EKF)

Road Graph ─► mission_manager (C++) ──► /plan (Path)
                                         │
                                         ▼
                                    mpcc_controller (C++ SQP)
                                         │
                                         ▼
                                    /qcar2_motor_speed_cmd (direct)
                                         │
                                         ▼
                                    qcar2_hardware (HIL driver)
```

All 7 C++ nodes launch from a single `ros2 launch` command. 

### Packages

| Package | Language | Role |
|---------|----------|------|
| `acc_mpcc_controller_cpp` | C++ | Primary stack: MPCC controller, mission manager, sign detector, state estimator, obstacle tracker, traffic light mapper, odom bridge |
| `acc_stage1_mission` | Python | Interfaces, YOLO bridge, dashboard |
| `qcar2_nodes` | C++ | Hardware interface (motors, cameras, lidar, LEDs) |
| `qcar2_interfaces` | C++ | Custom ROS 2 messages (`MotorCommands`, `BooleanLeds`) |

## Module Configuration

All backend selection is driven by `config/modules.yaml`:

```yaml
detection:
  backend: auto          # auto | hsv | yolo_coco | custom | hybrid | hough_hsv

path_planning:
  backend: experience_astar  # astar | dijkstra | weighted_astar | experience_astar

controller:
  backend: cpp           # auto | casadi | cpp | pure_pursuit

pedestrian_tracking:
  backend: kalman        # kalman | simple

state_estimation:
  backend: ekf           # ekf | raw

obstacle_tracking:
  backend: kalman        # kalman | simple
  use_lidar: true
```

Every choice is documented with benchmark results from the comparison scripts. ROS parameters override YAML at launch time.

### Presets

Preset YAML files in `config/presets/` bundle MPCC tuning parameters for quick comparison:

| Preset | `reference_velocity` | `contour_weight` | `lag_weight` | `horizon` | Notes |
|--------|---------------------|-------------------|--------------|-----------|-------|
| `default` | 0.45 m/s | 8.0 | 15.0 | 10 | Deployment-tuned baseline (Feb 2026) |

```bash
./run_mission.sh           # Uses default preset
```

## Comparison Tests

Two comparison scripts benchmark all backend alternatives on the same test inputs and produce quantitative metrics plus visualizations. Results are committed under `scripts/detection_results/` and `scripts/pathfinding_results/`.

### Detection Comparison

**Script:** `scripts/compare_detectors.py`

Generates 24 synthetic QLabs-like test images (5 object classes x 3 distances + multi-object scenes + edge cases) and benchmarks each detection backend.

| Backend | F1 | Precision | Recall | Latency (mean) | Notes |
|---------|----|-----------|--------|-----------------|-------|
| HSV+Contour | 0.429 | 0.429 | 0.429 | 1.63 ms | Pareto-optimal: fastest with competitive accuracy |
| Hough+HSV | 0.444 | 0.462 | 0.429 | 3.17 ms | Slightly better precision via Hough circle detection |
| Hybrid HSV+YOLO | 0.429 | 0.429 | 0.429 | 1.43 ms | HSV pre-filter + YOLO verification |
| Custom YOLOv8n | — | — | — | GPU-dependent | Best on real QLabs frames (not benchmarked offline) |

Per-class breakdown shows HSV excels at stop signs (F1=0.53) and cones (F1=0.67) — the two classes most critical for infraction avoidance. Traffic lights are handled by the state machine's temporal persistence rather than single-frame accuracy.

**Why HSV is the default:** At 1.6 ms per frame it runs at >600 Hz, leaving the CPU budget entirely for the MPCC solver. It requires zero ML dependencies, making the stack self-contained. When a GPU is available, the YOLO bridge relays custom-model detections for pedestrians and edge cases that HSV cannot handle, but it never overrides the HSV node's traffic control decisions.

**Output figures** (in `scripts/detection_results/`):
1. Detection grid — ground truth vs. predictions across scenes and detectors
2. Per-class precision/recall bar chart
3. Inference speed box plot
4. F1 vs. latency Pareto frontier
5. Distance estimation accuracy scatter
6. Confusion matrices

### Path Planning Comparison

**Script:** `scripts/compare_pathfinding.py`

Benchmarks 8 planning algorithms on the SDCS road graph for all 3 mission legs (Hub&rarr;Pickup, Pickup&rarr;Dropoff, Dropoff&rarr;Hub):

| Algorithm | Path Length | Nodes Expanded | Time | Notes |
|-----------|-----------|----------------|------|-------|
| A\* | 9.891 m (optimal) | 14 | 0.14 ms | Baseline optimal |
| Experience A\* | 9.891 m (optimal) | 14 (first), 0 (cached) | 0.008 ms cached | **13x faster on repeated queries** |
| Weighted A\* (e=1.5) | 9.891 m | fewer | ~0.10 ms | Bounded suboptimal, same path on this graph |
| Dijkstra | 9.891 m (optimal) | more | ~0.20 ms | No heuristic, more expansions |
| Bidirectional A\* | 9.891 m | ~7 each direction | ~0.12 ms | Two frontiers meet in the middle |
| D\* (traffic weights) | varies | varies | varies | +20 penalty on traffic/crosswalk edges |
| RRT\* | near-optimal | N/A | ~50 ms | Continuous space, road-biased sampling |
| CHOMP | near-optimal | N/A | ~100 ms | Trajectory optimization, gradient descent |

**Why Experience A\* is the default:** The taxi mission has exactly 3 fixed legs. After the first query, Experience A\* validates cached paths in O(n) and returns identical optimal paths in 0.008 ms — fast enough for real-time replanning if the road graph changes (e.g., cone-blocked edges). Graph-based planners dominate sampling-based ones (RRT\*, CHOMP) on this structured road network.

**Output figures** (in `scripts/pathfinding_results/`):
1. Road network visualization
2. Hub-to-pickup path comparison
3. Path length comparison (bar chart)
4. All-legs overlay
5. Weighted A\* epsilon tradeoff curve
6. Experience A\* speedup over repeated queries
7. Continuous vs. graph planning comparison
8. RRT\* tree visualization
9. CHOMP convergence plot
10. Comprehensive dashboard

## Detection Model

A custom YOLOv8n model (`models/best.pt`, 6.3 MB) is trained on QLabs screenshots for 8 classes:

| ID | Class | Used for |
|----|-------|----------|
| 0 | cone | Obstacle avoidance |
| 1 | green | Traffic light — go |
| 2 | person | Pedestrian stop |
| 3 | red | Traffic light — stop |
| 4 | round | Roundabout (unused in Stage 1) |
| 5 | stop | Stop sign — 3s pause |
| 6 | yellow | Traffic light — caution |
| 7 | yield | Yield sign — 2s pause |

Training pipeline: `training/capture_data.py` collects annotated frames, `training/train.py` fine-tunes YOLOv8n, and `training/data.yaml` defines the class mapping. The model runs on the host GPU via `yolo_detector_standalone.py` (Python 3.10 + CUDA), communicating with the ROS 2 stack through the YOLO bridge over local TCP/UDP sockets.

## MPCC Controller

Model Predictive Contouring Control (MPCC) formulates path following as a constrained optimization: minimize lateral deviation from the reference path (contouring error) while maximizing progress along it (lag), subject to kinematic constraints and road boundaries.

## Obstacle avoidance

Linearized halfspace constraints are used for real-time obstacle avoidance behavior. This is applied to all stationary obstacles in the road allowing for evasive replanning into the left lane, as well as pedestrians who are not in crosswalks. Pedestrians in crosswalks are treated as stop signs until they clear the road.

### C++ SQP Implementation (Default)

The primary controller (`cpp/mpcc_controller_node.cpp`) uses an Eigen-based Sequential Quadratic Programming solver with gradient projection for boundary constraints. Key design choices:

- **Direct motor commands**: Publishes `MotorCommands` (steering angle + throttle) directly to `qcar2_hardware`, bypassing the Nav2 velocity converter for lower latency
- **Curvature-adaptive speed**: `v_ref = reference_velocity * exp(-0.4 * |curvature|)` — slows into turns, accelerates on straights
- **Encoder feedback**: Reads `/qcar2_joint` wheel encoder velocity for closed-loop speed estimation
- **Road boundary soft constraints**: Loaded from `config/road_boundaries.yaml`, penalized in the cost function rather than hard-constrained (avoids infeasibility on noisy localization)

## Running

```bash
# Recommended: automated multi-terminal launch
./run_mission.sh             # Full C++ stack (auto-detects GPU for YOLO)
./run_mission.sh --no-gpu    # C++ HSV detection only
./run_mission.sh --dashboard # Enable real-time telemetry plots
./run_mission.sh --stop      # Stop all nodes (preserves QLabs window)
./run_mission.sh --reset     # Sync code + rebuild + reset car position
./run_mission.sh --logs      # Show latest session logs

# Manual launch (inside Isaac ROS container):
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py # Terminal 1: hardware + SLAM
ros2 launch acc_stage1_mission mpcc_mission_launch.py # Terminal 2: MPCC + mission
python3 Setup_Real_Scenario_Interleaved.py            # Terminal 3: QLabs scenario
```

### Session Logging

Every `run_mission.sh` invocation creates a session directory under `logs/session_YYYYMMDD_HHMMSS/` with:

- Per-terminal log files (`QCar2-Hardware.log`, `SLAM-Nav2.log`, `MPCC-Mission.log`, `YOLO-Bridge.log`, `GPU-YOLO.log`)
- `session_info.txt` with timestamp, git commit, GPU status, and preset
- MPCC per-cycle CSV (`mpcc_*.csv`) with position, velocity, errors, solve times
- Behavior event log (`behavior_*.log`) with state transitions and timing

All logs persist after terminal windows close, enabling post-mortem analysis of any mission failure.

## Building

```bash
# Inside the Isaac ROS dev container
cd /workspaces/isaac_ros-dev/ros2

# Build C++ stack (primary)
colcon build --packages-select acc_mpcc_controller_cpp
source install/setup.bash

# Build Python package (fallback + config + launch files)
colcon build --packages-select acc_stage1_mission
source install/setup.bash
```

## Repository Structure

```
quanser-acc/
├── cpp/                          # C++ node sources
│   ├── mpcc_controller_node.cpp  #   MPCC SQP controller
│   ├── mission_manager_node.cpp  #   Mission state machine + road graph
│   ├── sign_detector_node.cpp    #   HSV sign/light/cone detector
│   ├── state_estimator_node.cpp  #   EKF state fusion
│   ├── obstacle_tracker_node.cpp #   Multi-class Kalman tracker
│   ├── traffic_light_map_node.cpp#   Known light position mapper
│   ├── odom_from_tf_node.cpp     #   TF -> /odom bridge
│   ├── mpcc_solver.h             #   Eigen SQP solver
│   ├── road_graph.{h,cpp}        #   Road network + A* planner
│   ├── road_boundaries.{h,cpp}   #   Spline boundary loader
│   └── CMakeLists.txt
├── acc_stage1_mission/           # Python package
│   ├── yolo_bridge.py            #   GPU YOLO <-> ROS2 relay
│   ├── detection_interface.py    #   Detection backend abstraction
│   ├── planner_interface.py      #   Path planning backend abstraction
│   ├── module_config.py          #   YAML config loader + validation
│   ├── pedestrian_tracker.py     #   Kalman pedestrian tracker
│   ├── road_graph.py             #   Python road graph (fallback)
│   ├── dashboard.py              #   Real-time telemetry plot
├── config/
│   ├── modules.yaml              #   Backend selection (benchmarked defaults)
│   ├── mission.yaml              #   Waypoints + coordinate transforms
│   ├── road_boundaries.yaml      #   Road segments for MPCC boundaries
│   └── presets/                  #   MPCC tuning presets
├── launch/
│   └── mpcc_mission_launch.py    #   ROS 2 launch (7 C++ nodes)
├── models/
│   └── best.pt                   #   Custom YOLOv8n (8 classes, QLabs)
├── training/
│   ├── capture_data.py           #   Training data collection
│   ├── train.py                  #   YOLOv8 fine-tuning
│   └── data.yaml                 #   Class definitions
├── scripts/
│   ├── compare_detectors.py      #   Detection backend benchmark
│   ├── compare_pathfinding.py    #   Path planning benchmark
│   ├── detection_results/        #   Benchmark outputs (figures + metrics.json)
│   └── pathfinding_results/      #   Benchmark outputs (10 figures)
├── run_mission.sh                #   One-command launch + stop + logs
├── guide.md                      #   Setup and operation guide
└── CLAUDE.md                     #   Development reference
```

## Competition Infractions and Mitigations

| Infraction | Stars | Our Mitigation |
|-----------|-------|----------------|
| Lane departure (minor) | -1 | Progress-first MPCC tuning (`lag_weight > contour_weight`) + curvature-adaptive speed |
| Lane departure (major) | -2 to -5 | Reference-matched dynamics/timing (`dt=0.1`, 10 Hz loop) + boundary-aware cost (`boundary_weight=0` baseline) |
| Incomplete stop at stop sign | -2 | 3-second timed stop; spatial per-sign cooldown prevents double-stopping |
| Running red light | -2 | Traffic light state machine with 8s timeout + post-action suppression |
| Cone collision | -2 | Lidar-fused obstacle tracker; MPCC path avoidance |
| Failure to yield | -2 | Yield sign detection (HSV yellow triangle); 2s pause + 8s suppression |
| Stopping over stop line | -1 | Distance estimation from bbox width; early braking |

## Debugging

```bash
# Check controller state
ros2 topic echo /mpcc/status

# Check what signs/lights are detected
ros2 topic echo /traffic_control_state

# Check if motion is enabled
ros2 topic echo /motion_enable

# Check obstacle positions
ros2 topic echo /obstacle_positions

# View latest session logs
./run_mission.sh --logs
```

If the vehicle oscillates, reduce `reference_velocity` and increase `steering_rate_weight`. If it cuts corners, increase `contour_weight` relative to `lag_weight`.

For authoritative debugging context and verified baseline values, use `docs/DEBUG_BASELINE_LOCK.md`.

## Acknowledgements

This code was created with help from the PolyCtrl team submission from 2025 (collaborators with Brown). Further, code from https://github.com/stephen-crawford/PyMPC and github.com/tud-amr/mpc_planner was used as a reference throughout. All rights to the original code belong to the original authors, as appropriate.
