# Debug Baseline Lock (Authoritative)

Last verified: 2026-02-27
Scope: oversteering diagnosis + simulation prerequisites.

## Lock Policy
- Treat this file as the single baseline for debugging decisions.
- Do not change entries unless new runtime evidence contradicts them.
- If evidence conflicts, append a "Candidate Update" section first; do not overwrite baseline facts.

## Oversteering: Determined Primary Cause
- **Current codebase does NOT show the old severe oversteer bugs** (yaw-rate formula and timer-dt mismatch are fixed).
- **Most likely active cause in practice is configuration drift and stale reference usage**, not core dynamics math:
  - `README.md` still contains outdated controller/tuning claims (e.g., old weights and old curvature decay text).
  - Actual controller code and current preset use different values.
  - If launch/runtime parameters are taken from stale assumptions, steering behavior can be interpreted as "oversteer" even when solver math is correct.

## Verified Controller Facts (Do Not Re-litigate Unless Code Changes)
- Yaw rate formula is correct in `cpp/mpcc_types.h`: `dpsi = (v/L) * tan(delta) * cos(beta)`.
- Controller timer is tied to solver dt in `cpp/mpcc_controller_node.cpp` (100 ms for `dt=0.1`).
- `heading_weight` default is `0.0` in `cpp/mpcc_types.h`.
- `steering_rate_weight` set in controller init is `1.1` (reference matched).
- Curvature speed scheduling in controller uses `exp(-0.4 * |curvature|)`.
- Preset `config/presets/default.yaml` currently uses:
  - `reference_velocity: 0.45`
  - `contour_weight: 8.0`
  - `lag_weight: 15.0`
  - `horizon: 10`
  - `boundary_weight: 0.0`

## Image-Based Observations (2026-02-27 screenshots)
- Planned-vs-executed plot shows **moderate tracking quality**, not catastrophic divergence:
  - Max CTE around 0.23 m, average CTE around 0.05 m, average speed around 0.28 m/s.
  - Largest deviation appears on long right-side return segment.
- QLabs scene image does not indicate immediate hard-spin instability; vehicle appears near lower-right loop area.
- Conclusion: behavior is more consistent with tuning/context mismatch than a fundamental model bug.

## Required Simulation Components (Verified from student resources + QLabs docs)
- Host requirements:
  - Ubuntu 24.04 + NVIDIA GPU.
  - Docker engine + NVIDIA Container Toolkit.
- Quanser runtime dependencies:
  - `qlabs-unreal`, `python3-quanser-apis`, `quarc-runtime`.
- Containers:
  - Virtual env container: `quanser/virtual-qcar2` with `--network host`.
  - Dev container: Isaac ROS via `run_dev.sh`.
- Launch order:
  - Open QLabs Plane/SDCS world.
  - Start virtual container.
  - Run map spawn script (`Setup_Competition_Map.py` or interleaved variant).
  - Start development container.
  - Build/source ROS workspace.
  - Launch `qcar2_nodes` virtual launch.
- QLabs communication reference (official docs):
  - QCar2 HIL `18960`
  - Video2D `18940-18943`
  - Video3D `18965`
  - LiDAR `18966`
  - GPS `18967`
  - LED strip `18969`

## Networking Context (User-provided, keep as active assumption)
- Local adapters noted for Quanser application:
  - `10.39.59.233`
  - `172.17.0.1`
- Keep these IPs in scope during connectivity/debug checks.

## Fast Triage Order (Always Use)
1. Confirm runtime parameters actually loaded (not README assumptions).
2. Confirm QLabs + container communication health (ports and adapter pathing).
3. Confirm frame/setup order (scenario spawn before node launch sequence).
4. Only then retune contour/lag/speed; avoid changing model equations first.

## Candidate Update (2026-02-27)
- Added steering output slew limiter in `cpp/mpcc_controller_node.cpp`:
  - New parameter: `steering_slew_rate` (default `1.0` rad/s).
  - Applies per-cycle clamp to steering command deltas before publish.
  - Goal: suppress abrupt steering jumps that produce practical oversteer.
- Propagated parameter through launch/preset paths:
  - `launch/mpcc_mission_launch.py`
  - `cpp/launch/cpp_mission_launch.py`
  - `config/presets/default.yaml`
- Also aligned stale legacy launch defaults (`cpp/launch/cpp_mission_launch.py`) with active baseline:
  - `reference_velocity=0.45`, `lag_weight=15.0`, `horizon=10`
