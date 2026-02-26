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
| `acc_stage1_mission` | Python | Visualization tools (path_overlay, dashboard), GPU YOLO bridge, road_graph (for reports) |
| `acc_mpcc_controller_cpp` | C++ | All core nodes: MPCC controller, sign detector, mission manager, odom_from_tf, state estimator |
| `qcar2_nodes` | C++ | Hardware interface (motors, cameras, lidar, LEDs) |
| `qcar2_interfaces` | C++ | Custom ROS 2 messages (MotorCommands, BooleanLeds) |
| `qcar2_autonomy` | Python | Quanser reference autonomy (Nav2 client) |

### Key Nodes

| Node | Language | File | Purpose |
|------|----------|------|---------|
| `mission_manager` | C++ | `cpp/mission_manager_node.cpp` | State machine, path planning |
| `mpcc_controller` | C++ | `cpp/mpcc_controller_node.cpp` | MPCC path follower (SQP solver) |
| `sign_detector` | C++ | `cpp/sign_detector_node.cpp` | Fast HSV+contour sign/light/cone detection |
| `odom_from_tf` | C++ | `cpp/odom_from_tf_node.cpp` | TF -> /odom bridge |
| `state_estimator` | C++ | `cpp/state_estimator_node.cpp` | Encoder velocity + TF fusion |
| `obstacle_tracker` | C++ | `cpp/obstacle_tracker_node.cpp` | Kalman multi-object tracker |
| `traffic_light_map` | C++ | `cpp/traffic_light_map_node.cpp` | Spatial traffic light mapping |
| `path_overlay` | Python | `path_overlay.py` | Bird's-eye path + vehicle visualizer (opt-in: `--overlay`) |
| `dashboard` | Python | `dashboard.py` | Real-time telemetry plots (opt-in: `--dashboard`) |
| `yolo_bridge` | Python | `yolo_bridge.py` | GPU YOLO detection bridge (opt-in, requires GPU) |

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
# Terminal 1: Launch Cartographer SLAM + hardware (lidar, cameras, qcar2_hardware)
# NOTE: Use qcar2_cartographer_virtual_launch.py, NOT qcar2_slam_and_nav_bringup_virtual_launch.py
# The bringup launch adds AMCL + static_odom_tf + Nav2 which conflict with Cartographer
ros2 launch qcar2_nodes qcar2_cartographer_virtual_launch.py

# Terminal 2: Launch MPCC mission (all C++ nodes)
ros2 launch acc_stage1_mission mpcc_mission_launch.py

# Terminal 3: QLabs scenario setup (must be running before SLAM starts)
python3 Setup_Real_Scenario_Interleaved.py
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/modules.yaml` | Module backend selection (detection, path planning, controller) |
| `config/mission.yaml` | Mission waypoints, transforms, timing |
| `config/road_boundaries.yaml` | Road segments, traffic controls, obstacle zones |

## MPCC Tuning

Critical parameters (C++ `mpcc::Config` defaults, tuned via full-mission simulation):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `horizon` | 25 | Prediction horizon steps (2.5s at dt=0.1). Ref uses 10; our QP solver needs longer lookahead for curve anticipation. This is the MOST impactful parameter for CTE. |
| `contour_weight` | 8.0 | Lateral deviation penalty (lane keeping) — higher for tighter tracking |
| `lag_weight` | 10.0 | Progress tracking (ref: 7.0) |
| `velocity_weight` | 15.0 | Velocity reference tracking (ref: R_ref=17.0) |
| `reference_velocity` | 0.50 | Target speed (m/s) - lower for tighter tracking (ref uses 0.70) |
| `boundary_weight` | 0.0 | Road boundary soft constraint penalty (ref has 0; disabled — fights contour cost on curves) |
| `max_steering` | 0.45 | Hardware servo limit in radians (25.8°). Ref uses π/6=30° but hardware clips at 0.45. |
| `max_velocity` | 1.2 | Hard speed limit (m/s) - ref uses 2.0; qcar2_hardware PD controls actual speed |
| `use_direct_motor` | true | Bypass nav2_qcar_command_convert, publish MotorCommands directly |
| `heading_weight` | 2.0 | Heading alignment (not in ref; iLQR needs explicit heading cost unlike IPOPT). Too high (≥5) causes swerving oscillation. |
| `steering_rate_weight` | 0.5 | Smooth steering (ref R_u[1]=1.1; lower for agile steering in curves) |

Curvature-adaptive speed: `v_ref = reference_velocity * exp(-0.4 * |curvature|)` (matched to reference)

## Root Cause Analysis (Feb 2025-2026)

### Vehicle Not Following Path (Feb 2026)
- **Cause**: contour_weight (25) >> lag_weight (5) + low velocity_weight (2.0) made solver find "staying still in lane" optimal; curvature decay -1.2 was 3x too aggressive vs reference -0.4
- **Fix**: Aligned with reference (PolyCtrl 2025): contour=8, lag=12, velocity_weight=15, curvature decay -0.4, max_velocity 1.2, reference_velocity 0.65

### C++ Coordinate Transform Rotation (Feb 2026)
- **Cause**: C++ `coordinate_transform.h` used R(-θ) for qlabs_to_map instead of R(+θ). All paths were rotated ~82° wrong in map frame. Pickup point placed 7.1m from correct position.
- **Fix**: Swapped sin_t signs in all 5 transform functions to match Python R(+θ)

### Cross-Track Error 0.25m (Feb 2026)
- **Cause**: Three factors — (1) progress tracking could backtrack when vehicle veered laterally (reference uses monotonic-only), (2) contour:lag ratio 8:12=0.67 was 2.5x higher than reference 1.8:7=0.26 causing aggressive lateral oscillation, (3) fixed v_ref lookahead didn't adapt to actual vehicle speed
- **Fix**: Strictly monotonic progress (no backward), contour=4/lag=15 (ratio=0.27 matching reference), lookahead uses max(actual_v, v_ref)

### Lane Violations (Feb 2025)
- **Cause**: contour_weight (8) < lag_weight (15) prioritized progress over lane keeping
- **Fix**: Balanced ratio with contour=4, lag=15 (progress-first, reference-matched ratio)

### Hub Heading & Path Generation Mismatch (Feb 2026)
- **Cause**: C++ `road_graph.cpp` treated `-44.7` as degrees → heading 5.503 rad. Reference treats it as raw radians → heading 5.5655 rad. Difference of 3.58° changed SCS arc geometry for hub edges, causing path displacement ~0.31m at 5m distance. Python `road_graph.py` also had: (1) all hub edge radii=0 instead of reference 0.866326, (2) no B-spline resampling (rough non-uniform path), (3) no scale factor [1.01,1.0], (4) wrong default transform angle (0.7802 vs 0.7177).
- **Fix**: Hub heading = `fmod(-44.7, 2π)` = 5.5655 rad in both C++ and Python. Hub→1 edge radius=0.866326. 10→hub radius=0 (straight, since SCS infeasible at this heading — reference also silently fails on this edge). Added B-spline resample + scale factor to Python.

### Vehicle Swerving & Lane Violations — High CTE 0.746m (Feb 2026)
- **Cause**: Control smoothness weights far too conservative vs reference — steering_rate_weight=3.0 (ref 1.1, 2.7x too high), acceleration_weight=1.0 (ref 0.005, 200x too high), jerk_weight=0.3 (ref has none), max_steering_rate=0.8 (ref unlimited), max_acceleration=0.8 (ref unlimited). Also boundary_weight=20.0 fighting path following (ref has 0). Combined effect: vehicle couldn't steer or decelerate fast enough for curves → overshooting → oscillation → 0.746m max CTE.
- **Fix**: steering_rate_weight=1.2, acceleration_weight=0.01, jerk_weight=0.0, max_steering_rate=1.5, max_acceleration=1.5, boundary_weight=8.0, steering_weight=0.05

### Obstacle Collisions & 0.15 m/s Avg Speed (Feb 2026)
- **Cause**: Three compounding failures — (1) `sign_detector_node.cpp` created `obstacle_pub_` but NEVER published to `/obstacle_positions`, so MPCC solver's halfspace obstacle constraints (weight=200) never activated; (2) `/motion_enable` was binary stop/go — when cone detected, vehicle fully stopped instead of steering around; (3) `mission_manager_node.cpp` had path-level obstacle avoidance disabled (`enable_path_avoidance_=false`). Result: vehicle entered stop-resume cycles (avg 0.15 m/s) and collided with obstacles it couldn't avoid.
- **Fix**: (1) Added TF-based obstacle position publishing to sign_detector — computes map-frame (x,y) from camera bearing + distance + TF lookup; (2) Changed motion_enable from binary stop to obstacle-aware speed limit (0.20 m/s creep while solver steers around); (3) ~~Enabled path-level avoidance in mission_manager~~ (later re-disabled, see below)

### Path-Level Avoidance Causing Vehicle Stuck & Path Ballooning (Feb 2026)
- **Cause**: `enable_path_avoidance_=true` in `mission_manager_node.cpp` combined with obstacle tracker false positives from lidar. The obstacle tracker confirms tracks with only 2 hits and persists them for 30 seconds. Lidar clusters from yield signs, walls, and other fixed objects get misclassified as "person" obstacles. The `PathModifier::generate_avoidance_path()` laterally offsets path waypoints by ~0.35m around each phantom obstacle, and re-fires every 2 seconds. Result: path "balloons" outward, MPCC can't follow it, vehicle gets stuck at ~18% progress with steering saturated at 30°.
- **Fix**: Disabled `enable_path_avoidance_=false`. MPCC solver's built-in halfspace obstacle constraints handle avoidance at the control level (same approach as reference implementation). The path-level avoidance was an addition not in the reference that does more harm than good with the noisy lidar-only obstacle tracker.

### Route Leg Slicing — Pickup Index Wrong Pass (Feb 2026)
- **Cause**: The loop path `[24, 20, 9, 10]` passes near Pickup TWICE (outbound via node 20 at idx ~10513, returning via inner track at idx ~24083). `find_closest_idx` (global argmin) picked the second pass (0.0000004m closer due to B-spline fitting). This put `pickup_idx > dropoff_idx`, causing `pickup_to_dropoff` leg to be silently skipped and `hub_to_pickup` to cover 24083 waypoints (most of the loop).
- **Fix**: Added `find_first_local_min()` in both C++ (`road_graph.cpp`) and Python (`road_graph.py`) — scans forward to find the first contiguous region within 0.5m of target, returns argmin within that region. Dropoff search starts after pickup index to enforce ordering.

### Vehicle Stuck at 18% Progress — 35° Heading Mismatch (Feb 2026)
- **Cause**: Cartographer config `tracking_frame = "base_scan"` (2026 Quanser default) differs from reference `tracking_frame = "base_link"` (2025 config). The lidar driver publishes ranges in reverse order (CW→CCW convention), which mirrors left/right in the scan. With `tracking_frame = "base_link"`, Cartographer applies `base_link→base_scan` TF before processing scan data, causing the vehicle to initialize at heading ~41° in map frame — matching the path heading (35°). With `tracking_frame = "base_scan"`, no TF correction is applied and the vehicle starts at heading 0°, causing a 35° mismatch with the path. Steering saturates at max (30°) trying to correct, and the vehicle gets stuck circling.
- **Fix**: Changed `tracking_frame` to `"base_link"` in `qcar2_2d.lua` (matching reference). Applied via sed in SLAM helper script inside `run_mission.sh`.

### Oversteering on Curves — Wrong Yaw Rate Formula (Feb 2026)
- **Cause**: Solver dynamics used `dψ = (v/L)·sin(β)` instead of the reference formula `dψ = (v/L)·tan(δ)·cos(β)`. Since `sin(β) = tan(δ)·cos(β)/2`, the solver's yaw rate was exactly **half** the correct value. The solver compensated by commanding 2× the needed steering angle to achieve the desired turn radius, but the vehicle (using its own internal model) turned at the correct rate — causing overshoot on every curve.
- **Fix**: Changed dynamics to `dψ = (v/L)·tan(δ)·cos(β)` matching reference MPCC.py line 56. Updated Jacobian linearization accordingly. Added 7 reference-matching dynamics validation tests.

### Oversteering on Roundabout — Timer/Solver dt Mismatch (Feb 2026)
- **Cause**: Control loop ran at 20Hz (50ms timer) but solver dt=0.1s (10Hz). The warm-start shifted trajectory by one solver step (0.1s) every 0.05s call, making predicted trajectory out of sync with reality by 2x. Reference (PolyCtrl 2025) uses controllerUpdateRate=10 with dt=0.1. Also, path reference arrays were horizon-sized (N refs) but solver needs N+1 refs (k=0..N); terminal cost at k=N reused ref[N-1].
- **Fix**: Changed control timer to match solver dt (`int(dt*1000)ms`). Fixed `get_spline_path_refs` and `get_path_refs` to return `horizon+1` refs. Fixed stuck detection timer to use `config_.dt`.

### Oversteering at Path Start — 0.467m CTE (Feb 2026)
- **Cause**: CubicSplinePath amplified curvature at SCS straight→arc junctions. The first ~86mm of each leg path had κ=2.25 (vehicle's physical limit, clamped from higher). Curvature then dropped from 2.25 to 0.00 over ~40mm — the vehicle couldn't unwind max steering fast enough, overshooting by 0.4m+. Three factors: (1) raw SCS waypoints were smooth but cubic spline fitting created curvature overshoot at junction transitions, (2) solver used pre-computed path references that couldn't adapt when vehicle deviated, (3) u_prev used solver's previous command instead of measured velocity, creating phantom smoothness cost
- **Fix**: (1) Gaussian path smoothing in `cubic_spline_path.h` (σ=150 waypoints ≈ 150mm) spreads curvature transitions over ~300mm; (2) Adaptive path re-projection via `PathLookup` in controller node — solver re-projects predicted positions onto path each SQP iteration (matches reference θ-as-decision-variable); (3) u_prev updated from measured state (matching reference MPC_node.py line 553). Max CTE: 0.467m → 0.176m

### Sign Detection Failures
- **Cause**: Detection persistence threshold too high (5 frames); global 15s stop sign cooldown; traffic light state machine fragile
- **Fix**: Persistence=2 frames; spatial per-sign cooldown (5s); traffic light timeout (15s) + no-red expiry (2s)

## Reference Codebase (PolyCtrl 2025)

**Location**: `/home/stephen/ACC2025_Quanser_Student_Competition`

This is the working reference implementation from the ACC 2025 competition (team PolyCtrl).
Our codebase was built to match its path planning and controller behavior.

### Key Reference Files

| File | Purpose |
|------|---------|
| `polyctrl/polyctrl/MPC_node.py` | Main control loop: road map init, coordinate transform, MPCC solve, mission state machine |
| `polyctrl/polyctrl/MPCC.py` | CasADi MPCC solver (IPOPT backend) — contour/lag error, bicycle dynamics |
| `polyctrl/utils/path_planning.py` | RoadMap, RoadMapNode, RoadMapEdge, SCSPath, generate_path, D* search |
| `polyctrl/utils/mats.py` | SDCSRoadMap: 24-node road network with pixel→QLabs coordinate conversion |
| `polyctrl/polyctrl/Detection_node.py` | YOLOv8 detection (cone, stop, traffic lights, etc.) |
| `Setup_Real_Scenario_Interleaved.py` | QLabs scenario setup (waypoints, traffic elements, obstacles) |

### Reference Path Planning Approach

1. **Single loop path**: `nodeSequence = [24, 20, 9, 10]` (hub → near pickup → far end → near hub)
2. **SCS waypoint generation** between each node pair via graph edges
3. **B-spline resample** via `splprep/splev` at `spacing=0.001` (1mm uniform)
4. **Scale factor** `[1.01, 1.0]` (slight x-axis expansion for lane centering)
5. **Coordinate transform**: QLabs → map frame: translate by `(-1.205, -0.83)`, rotate by `θ=0.7177 rad`
6. **CubicSpline** arc-length parameterization for MPCC reference spline
7. **Curvature** computed via `np.gradient` on spline-evaluated points

### Reference MPCC Parameters

| Parameter | Reference Value | Our Value |
|-----------|----------------|-----------|
| `q_c` (contour) | 1.8 | 4.0 (ratio 0.27 matches) |
| `q_l` (lag) | 7.0 | 15.0 (ratio 0.27 matches) |
| `gamma` (progress) | 1.0 | (implicit in lag_weight) |
| `R_u` (control smooth) | [0.005, 1.1] | acceleration_weight=0.01, steering_rate_weight=1.2 |
| `R_ref` (vel tracking) | [17.0, 0.05] | velocity_weight=15.0 |
| `v_max` | 2.0 | 1.2 |
| `u_ref_max` | 0.70 | 0.65 |
| Horizon K | 10 @ 10Hz | 10 @ 20Hz |
| Solver | CasADi + IPOPT | Condensed QP + ADMM (ported from PyMPC) |

## Reference Codebase (PyMPC cpp_mpc)

**Location**: `/home/stephen/PyMPC/cpp_mpc`

This is a C++ scenario-based MPC implementation with a condensed QP + ADMM solver.
Our MPCC solver is based on its solver architecture.

### Key Reference Files

| File | Purpose |
|------|---------|
| `src/mpc_controller.cpp` | Core algorithm: SQP loop, condensed QP construction, sensitivity matrices, line search |
| `src/qp_solver.cpp` | ADMM QP solver (Eigen-only, no external deps): handles inequality + box constraints |
| `src/dynamics.cpp` | Unicycle dynamics with RK4 integration + Jacobian computation |
| `src/collision_constraints.cpp` | Linearized halfspace constraints for obstacle avoidance |
| `src/contouring_mpc.cpp` | Contouring MPC wrapper: goal-based tracking, obstacle modes |
| `include/config.hpp` | Configuration parameters (horizon, weights, solver settings) |
| `include/types.hpp` | Core data structures (EgoState, EgoInput, Scenario, etc.) |

### PyMPC Solver Architecture

1. **Condensed QP formulation**: Decision variables are `Δu` only (2N variables for N steps).
   Position sensitivities `P[k][j] = ∂position[k]/∂u[j]` built from Jacobian products.
   Hessian `H = Σ w_goal * P[k]^T * P[k] + w_vel * V[k]^T * V[k] + diag(w_accel, w_steer)`.
2. **ADMM QP solver**: Solves `min 0.5*x^T*H*x + g^T*x s.t. C*x >= d, lb <= x <= ub`.
   Uses Cholesky factorization of KKT matrix, adaptive rho scaling, warm-start support.
3. **SQP outer loop**: Re-linearizes dynamics and constraints, line search (α = 1, 0.5, 0.25).
4. **Nonlinear RK4 rollout**: After QP solve, propagate with full nonlinear dynamics.

### PyMPC vs Our Implementation

| Aspect | PyMPC | Our MPCC |
|--------|-------|----------|
| Dynamics | Unicycle (4D: x,y,θ,v) | Ackermann bicycle (5D: x,y,θ,v,δ) |
| Controls | [a, ω] direct | [a, δ_dot] rate-based |
| Cost | Goal tracking + velocity | Contouring e_c + lag e_l + heading + velocity |
| QP Solver | ADMM (Eigen-only) | ADMM (ported from PyMPC) |
| Constraints | Hard halfspace (via QP) | Hard halfspace (via QP) + soft boundary penalties |
| Horizon | N=20, dt=0.1s | N=10, dt=0.1s |

### Hub Node Heading

The reference uses `-44.7 % (2*pi)` where `-44.7` is treated as **raw radians** (not degrees).
This gives `5.5655 rad ≈ 318.9°`. The transform angle `0.7177 = 2π - 5.5655`.
Our code matches: `std::fmod(-44.7, 2π)` in C++, `(-44.7) % (2*np.pi)` in Python.

## What Is Based on the 2025 Reference Implementation

The reference codebase (PolyCtrl 2025) is written entirely in Python using CasADi. Our
stack re-implements the core algorithms in C++ for performance, and adds features the
reference lacks (C++ sign detection, road boundary constraints, obstacle tracking, recovery
strategies). Below is a component-by-component breakdown.

### Directly Ported from Reference (algorithm and data)

| Our File(s) | Reference File | What Was Ported |
|-------------|----------------|-----------------|
| `cpp/road_graph.cpp`, `road_graph.py` | `utils/mats.py` | SDCSRoadMap 24-node graph: all node pixel coordinates, heading angles, edge connectivity, turn radii (inner/outer/circle/oneway/kink). Ported to C++ structs. |
| `cpp/road_graph.cpp`, `road_graph.py` | `utils/path_planning.py` | SCSPath (Straight-Curve-Straight) algorithm: direction detection via signed angle, circle center solve (parallel/anti-parallel/general 2×2 system), arc discretization. Ported to C++ with identical math. |
| `cpp/road_graph.cpp`, `road_graph.py` | `utils/path_planning.py` | A* graph search with +20 edge penalties on traffic-controlled edges (reference uses D* with same penalties; produces identical routes). Path generation via node sequences. Ported to C++. |
| `cpp/road_graph.cpp`, `road_graph.py` | `MPC_node.py:125-136` | Hub spawn node setup: position `(-1.205, -0.83)`, heading `-44.7 % 2π`, edge radii `(0.0, 1.48202, 0.866326)`, loop sequence `[24, 20, 9, 10]`. Ported identically. |
| `cpp/road_graph.cpp`, `road_graph.py` | `MPC_node.py:136` | Path resampling to uniform 0.001m spacing with scale factor `[1.01, 1.0]`. C++ uses natural cubic spline (Thomas algorithm, equivalent to reference scipy B-spline); Python uses scipy `splprep/splev` to match exactly. |
| `cpp/coordinate_transform.h`, `road_graph.py` | `MPC_node.py:139-153` | QLabs↔map coordinate transform: translate by `(-1.205, -0.83)`, rotate by `R(+θ)` with `θ=0.7177 rad`. Ported to C++ inline functions. |
| `cpp/mpcc_solver.h` | `MPCC.py` | MPCC contouring/lag error formulation: `e_c = sin(φ)(X−x_ref) − cos(φ)(Y−y_ref)`, `e_l = −cos(φ)(X−x_ref) − sin(φ)(Y−y_ref)`. Bicycle model with slip angle `β = atan(tan(δ)/2)`. Contour:lag weight ratio 0.27 matching reference 1.8:7. |
| `cpp/mpcc_solver.h` | `MPCC.py:52-57` | Vehicle dynamics model: `dX = v·cos(ψ+β)`, `dY = v·sin(ψ+β)`, `dψ = (v/L)·tan(δ)·cos(β)` with L=0.256m wheelbase. Identical formulation. |
| `cpp/mpcc_controller_node.cpp` | `MPC_node.py:601-637` | Curvature-adaptive speed: `v_ref = reference_velocity × exp(−0.4 × |κ|)`. Same decay constant −0.4 as reference. |
| `cpp/cubic_spline_path.h` | `MPC_node.py:165-180` | CubicSpline arc-length parameterization for controller reference path. C++ uses Thomas algorithm (O(n) tridiagonal solver); reference uses scipy CubicSpline. Functionally equivalent. |
| `config/presets/default.yaml` | `MPC_node.py:601-637` | MPCC weight ratios calibrated to match reference: contour:lag=0.27, velocity_weight=15, steering_rate_weight=1.2, reference_velocity=0.65. |

### Inspired by Reference (same concept, different implementation)

| Our Component | Reference Equivalent | Differences |
|---------------|---------------------|-------------|
| Condensed QP + ADMM solver (`mpcc_solver.h`) | CasADi + IPOPT (`MPCC.py`) + PyMPC ADMM (`qp_solver.cpp`) | Solver architecture ported from PyMPC: condensed QP formulation (2N decision variables), ADMM QP solver, SQP outer loop with nonlinear RK4 rollout. Cost function from PolyCtrl 2025 (contouring/lag). No CasADi dependency. |
| C++ mission state machine (`mission_manager_node.cpp`) | Python state machine (`MPC_node.py:531-722`) | Reference embeds mission logic in control loop; we use a separate node with Nav2 action clients, recovery strategies, and LED control. Same mission sequence (hub→pickup→dropoff→hub). |
| C++ sign detector (`sign_detector_node.cpp`) | YOLOv8 (`Detection_node.py`) | Reference uses neural network (YOLOv8 + CUDA); we use fast HSV thresholding + contour analysis (no GPU required). Detects same classes: stop, traffic light colors, cones. |
| Road boundary constraints (`road_boundaries.h`) | Not in reference | Added soft quadratic penalty for lane boundaries. Reference relies solely on contour weight for lane keeping. |
| Obstacle tracker (`obstacle_tracker_node.cpp`) | Inline in `MPC_node.py` | Reference handles cone avoidance inline in MPCC cost function; we use a separate node with Kalman filtering that publishes obstacle positions + velocities for the controller. |
| Obstacle avoidance (`mpcc_solver.h`) | Soft circular penalty in `MPCC.py` | Reference uses simple distance penalty; we use linearized halfspace constraints re-linearized per SQP iteration. Supports dynamic obstacles via velocity prediction over the horizon. |
| Hermite path blending (`mission_manager_node.cpp:588-668`) | Not in reference | Added Hermite spline transition at path start to align with vehicle heading. Reference starts MPCC from current pose without blending. |
| Recovery strategies (`mission_manager_node.cpp`) | Not in reference | Added retry, backup, costmap clear, skip, restart strategies. Reference has no recovery — if MPCC fails, it just stops. |

### Not from Reference (original work)

| Component | Description |
|-----------|-------------|
| `cpp/state_estimator_node.cpp` | Encoder-based velocity estimation + TF fusion. Reference reads velocity from JointState directly. |
| `cpp/traffic_light_map_node.cpp` | Spatial traffic light mapping (associates detections with known intersection positions). Not in reference. |
| `cpp/sign_detector_node.cpp` | Full HSV pipeline with morphological ops, contour area filtering, spatial cooldowns. Reference uses YOLO only. |
| `path_overlay.py` | Bird's-eye visualization of planned path + vehicle position. Debug tool not in reference. |
| `config/road_boundaries.yaml` | Road segment definitions with boundary polygons for soft constraints. Not in reference. |
| `run_mission.sh` | Automated launch script handling Docker, builds, terminal management. Not in reference. |

## Known Issues & Debug Tips

- Check `/mpcc/status` for controller state
- Check `/traffic_control_state` for sign/light detection
- Check `/motion_enable` for obstacle detector output
- Mission logs written to `logs/` directory
- If vehicle oscillates, reduce `reference_velocity` and slightly increase `steering_rate_weight` (current 1.2, ref 1.1)
- If vehicle cuts corners, increase `contour_weight` relative to `lag_weight`
- C++ sign detector uses HSV color thresholds tuned for QLabs - may need adjustment for real hardware
