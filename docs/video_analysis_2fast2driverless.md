# Replication Guide for the "2 Fast 2 Driverless – Quanser Self-Driving Competition 2025" Video

## What this document is (and isn't)

This is a practical reverse-engineering guide to help you reproduce the kind of system and demo shown in the team's submission video for the Quanser ACC Self-Driving Car Student Competition 2025.

Because the source video itself was not directly accessible for frame-by-frame inspection in this environment, this guide is based on:

* the public video title/context,
* competition descriptions,
* publicly described task requirements,
* common Quanser/QCar competition pipelines,
* and best practices for student autonomous taxi demos.

It is designed to help you reproduce the **technical stack, architecture, engineering style, and demonstration strategy** likely used in a strong competition submission.

---

## 1) Likely system objective shown in the video

The 2025 Quanser ACC challenge was framed as an **autonomous taxi service** on a 1:10-scale city map. A strong submission video typically demonstrates:

1. **Autonomous route planning** between pickup and drop-off points
2. **Traffic rule compliance** (traffic lights, signs)
3. **Obstacle handling** (static and dynamic)
4. **Reliable lane following / intersection behavior**
5. **Real-time onboard decision making**
6. **Repeatability and system robustness**
7. **Simulation-to-real transfer (digital twin → physical car)**

If you want to duplicate the result, think of the project as a full stack autonomy pipeline, not just "lane following."

---

## 2) Techniques likely used (reverse-engineered stack)

### A. Perception stack (core techniques)

A competitive submission almost certainly uses a modular perception pipeline with some combination of:

#### 1. Lane / road understanding

* Classical CV first (faster, deterministic):

  * perspective transform / ROI masking
  * thresholding (color/intensity)
  * edge detection (Canny)
  * line fitting / centerline estimation
* Optional learned component for robustness in variable lighting

**Why likely:** competition settings demand reliability + low latency + explainability.

#### 2. Traffic light and sign detection

Two common approaches:

* **Classical CV** (HSV color masks, blob detection, geometric constraints)
* **Small neural detector/classifier** (YOLO-nano / MobileNet / custom CNN)

Likely strategy in a high-performing team:

* Keep detection lightweight
* Use confidence thresholds + temporal smoothing
* Add rule-based filtering (e.g., sign only valid near intersections)

#### 3. Obstacle detection and tracking

Potential inputs on QCar platform include camera(s), depth, LiDAR, etc. A robust approach usually combines:

* nearest obstacle extraction from depth/LiDAR
* occupancy grid or local obstacle map
* motion gating / tracking for dynamic obstacles

#### 4. Sensor fusion (lightweight)

Even if not full EKF-SLAM, teams often fuse:

* odometry / wheel encoders
* IMU heading
* camera detections
* depth/LiDAR obstacle cues

**Practical competition pattern:** "good enough fusion" with clear fail-safes beats overly complex state estimation.

---

### B. Planning and decision-making

#### 1. Mission / task planner (taxi logic)

Likely implemented as a finite state machine (FSM):

* Idle / waiting for request
* Route planning to pickup
* Pickup stop / dwell
* Route planning to drop-off
* Delivery completion
* Repeat

#### 2. Route planner on road graph

Expected for taxi challenge:

* represent city map as graph (nodes = intersections/stops, edges = lanes/segments)
* shortest path (Dijkstra/A*)
* edge costs may include blockages / dynamic obstacles / traffic delays

#### 3. Behavior planner (rules)

A separate layer from path planning, handling:

* stop at signs / red lights
* yield rules
* safe merge / turn handling
* reroute triggers

**Strong strategy:** split planning into **global route** + **local behavior** + **tracking control**.

---

### C. Control stack

#### 1. Lateral control

Likely one of:

* Pure Pursuit
* Stanley controller
* PID on lane-center error + heading error

#### 2. Longitudinal control

* PID speed controller
* Speed profile by context:

  * slower near intersections
  * slower during perception ambiguity
  * stop/go logic from traffic rules

#### 3. Safety supervisor (critical)

High-performing competition teams usually add a top-level safety layer that can:

* cap speed
* force stop on low confidence / obstacle proximity
* reset state after rule violations

---

### D. Simulation-to-real strategy (digital twin)

Public descriptions mention use of a **digital twin** before deployment. This is a major clue.

Likely workflow:

1. Develop and debug in simulator/QLabs
2. Log failures and corner cases
3. Tune thresholds/controller gains in sim
4. Validate on physical track
5. Keep platform-dependent interfaces isolated (same logic, different drivers)

**Replication principle:** separate `core autonomy logic` from `sim/hardware adapters`.

---

## 3) Likely engineering strategies that made the video strong

### Strategy 1: Rule-based reliability over flashy models

In competition settings, teams often prioritize deterministic behavior for:

* traffic compliance
* intersection handling
* fail-safe reactions

Use ML only where it adds clear value (e.g., sign/light detection), not everywhere.

### Strategy 2: Hierarchical autonomy

A good submission almost always layers the stack:

* Perception
* State estimation / world model
* Mission planner
* Behavior planner
* Controller
* Safety supervisor

This makes debugging and demo narration much easier.

### Strategy 3: Time-budgeted computation

On embedded hardware (e.g., Jetson), teams succeed by assigning update rates:

* camera processing: e.g., 10–30 Hz
* control loop: e.g., 20–100 Hz
* route replanning: lower rate or event-driven
* heavy vision models: reduced resolution + throttled inference

### Strategy 4: Temporal smoothing and debouncing

Competition-winning behavior often depends on *not reacting instantly to every noisy detection*:

* require N consecutive frames before accepting sign/light
* hold state for minimum duration
* hysteresis on obstacle and lane confidence

### Strategy 5: Fail-soft behavior

Instead of "perfect or crash," use fallbacks:

* if lane uncertain → slow down
* if sign confidence drops → maintain last valid state briefly
* if path blocked → stop and replan

---

## 4) Code style likely used (and what you should copy)

Even without the exact code, the most reproducible style for this kind of project is:

### A. Modular package layout

```text
project/
  config/
    sim.yaml
    real.yaml
    controller.yaml
    perception.yaml
  src/
    main.py
    runtime/
      node_manager.py
      scheduler.py
    io/
      camera_interface.py
      lidar_interface.py
      imu_interface.py
      motor_interface.py
      sim_adapter.py
      hardware_adapter.py
    perception/
      lane_detector.py
      traffic_light_detector.py
      sign_detector.py
      obstacle_detector.py
      tracking.py
    localization/
      state_estimator.py
      map_projection.py
    planning/
      road_graph.py
      route_planner.py
      behavior_fsm.py
      local_planner.py
    control/
      lateral_controller.py
      longitudinal_controller.py
      safety_supervisor.py
    utils/
      logger.py
      timing.py
      geometry.py
      visualization.py
  tests/
  scripts/
    run_sim.sh
    run_real.sh
    replay_logs.py
  logs/
  videos/
```

### B. "Thin main, fat modules"

`main.py` should only:

* load config
* initialize modules
* run loop
* coordinate shutdown

Avoid giant scripts with all logic in one file.

### C. Config-driven behavior

Use YAML/TOML/JSON for:

* thresholds
* PID gains
* camera calibration
* map graph definitions
* speed limits

This makes sim-to-real tuning much faster.

### D. Strong logging and diagnostics

Essential for replication:

* timestamped logs
* per-module latency
* detection confidences
* FSM state transitions
* control outputs
* reason for emergency stop

### E. Deterministic interfaces

Define small dataclasses / message structs, e.g.:

* `PerceptionFrame`
* `VehicleState`
* `RoutePlan`
* `BehaviorCommand`
* `ControlCommand`

This reduces coupling and makes debugging much easier.

---

## 5) Video production techniques likely used in the submission (and how to reproduce)

Strong competition submissions often look better not because of editing tricks, but because they clearly communicate the system.

### A. Structure the demo like a proof

Replicate this format:

1. **Problem statement** (autonomous taxi challenge)
2. **System overview diagram** (pipeline)
3. **Perception demo** (what the car sees, detects)
4. **Planning demo** (route visualization, decision points)
5. **Full mission run** (start to finish, real-time or 2x speed)
6. **Statistics** (completion time, infractions, reliability)

### B. Use overlays for clarity

* Bird's-eye path overlay showing planned vs actual trajectory
* Detection bounding boxes on camera feed
* State machine status indicator
* Speed/heading display

### C. Show failure handling

Demonstrating graceful recovery from edge cases (missed detection, brief obstacle) is more impressive than a perfect run — it shows robustness.
