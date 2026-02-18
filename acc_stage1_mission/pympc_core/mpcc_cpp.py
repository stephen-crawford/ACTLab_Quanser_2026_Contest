"""
Python ctypes wrapper for the C++ MPCC solver.

Loads libmpcc_solver.so and provides a Pythonic interface compatible
with the MPCCSolver API used by mpcc_controller.py.

The C++ solver uses:
- Ackermann dynamics with RK4 simulation + Euler Jacobians
- SQP with iterative gradient projection QP
- Warm-starting from previous solution
- ~100us solve time (vs ~50ms for CasADi)
"""

import ctypes
import ctypes.util
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .spline_path import CubicSplinePath


# ============================================================================
# C struct definitions matching mpcc_solver.h
# ============================================================================

class MPCCParams(ctypes.Structure):
    _fields_ = [
        ("horizon", ctypes.c_int),
        ("dt", ctypes.c_double),
        ("wheelbase", ctypes.c_double),
        ("max_velocity", ctypes.c_double),
        ("min_velocity", ctypes.c_double),
        ("max_steering", ctypes.c_double),
        ("max_acceleration", ctypes.c_double),
        ("max_steering_rate", ctypes.c_double),
        ("reference_velocity", ctypes.c_double),
        ("contour_weight", ctypes.c_double),
        ("lag_weight", ctypes.c_double),
        ("velocity_weight", ctypes.c_double),
        ("steering_weight", ctypes.c_double),
        ("acceleration_weight", ctypes.c_double),
        ("steering_rate_weight", ctypes.c_double),
        ("jerk_weight", ctypes.c_double),
        ("robot_radius", ctypes.c_double),
        ("safety_margin", ctypes.c_double),
        ("obstacle_weight", ctypes.c_double),
        ("boundary_weight", ctypes.c_double),
        ("max_sqp_iterations", ctypes.c_int),
        ("max_qp_iterations", ctypes.c_int),
        ("qp_tolerance", ctypes.c_double),
    ]


class MPCCPathPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("cos_theta", ctypes.c_double),
        ("sin_theta", ctypes.c_double),
        ("curvature", ctypes.c_double),
    ]


class MPCCObstacle(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("radius", ctypes.c_double),
    ]


class MPCCBoundary(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_double),
        ("ny", ctypes.c_double),
        ("b_left", ctypes.c_double),
        ("b_right", ctypes.c_double),
    ]


class MPCCResultC(ctypes.Structure):
    _fields_ = [
        ("v_cmd", ctypes.c_double),
        ("delta_cmd", ctypes.c_double),
        ("omega_cmd", ctypes.c_double),
        ("solve_time_us", ctypes.c_double),
        ("success", ctypes.c_int),
        ("predicted_x", ctypes.c_double * 50),
        ("predicted_y", ctypes.c_double * 50),
        ("predicted_theta", ctypes.c_double * 50),
        ("predicted_len", ctypes.c_int),
    ]


# ============================================================================
# Library loader
# ============================================================================

def _find_library() -> Optional[str]:
    """Find the libmpcc_solver.so shared library."""
    # Search locations in priority order
    search_dirs = [
        # Same package cpp/ directory
        os.path.join(os.path.dirname(__file__), '..', '..', 'cpp'),
        # Installed location (colcon)
        os.path.join(os.path.dirname(__file__), '..', '..', 'lib'),
        # Build directory
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'build',
                     'acc_stage1_mission', 'lib'),
        # Absolute path as last resort
        '/home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission/cpp',
    ]

    for d in search_dirs:
        path = os.path.join(os.path.abspath(d), 'libmpcc_solver.so')
        if os.path.exists(path):
            return path

    return None


def _load_library():
    """Load the C++ MPCC solver library and configure function signatures."""
    lib_path = _find_library()
    if lib_path is None:
        return None

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return None

    # Configure function signatures
    lib.mpcc_create.argtypes = [ctypes.POINTER(MPCCParams)]
    lib.mpcc_create.restype = ctypes.c_void_p

    lib.mpcc_destroy.argtypes = [ctypes.c_void_p]
    lib.mpcc_destroy.restype = None

    lib.mpcc_reset.argtypes = [ctypes.c_void_p]
    lib.mpcc_reset.restype = None

    lib.mpcc_solve.argtypes = [
        ctypes.c_void_p,                      # solver
        ctypes.POINTER(ctypes.c_double),       # state[5]
        ctypes.POINTER(MPCCPathPoint),         # path points
        ctypes.c_int,                          # n_path
        ctypes.POINTER(MPCCObstacle),          # obstacles
        ctypes.c_int,                          # n_obstacles
        ctypes.POINTER(MPCCBoundary),          # boundaries
        ctypes.c_int,                          # n_boundaries
        ctypes.c_double,                       # current_progress
        ctypes.c_double,                       # path_total_length
        ctypes.POINTER(MPCCResultC),           # result
    ]
    lib.mpcc_solve.restype = ctypes.c_int

    return lib


# Global library handle (loaded once)
_LIB = _load_library()


def is_available() -> bool:
    """Check if the C++ MPCC solver is available."""
    return _LIB is not None


# ============================================================================
# Python-friendly result class
# ============================================================================

@dataclass
class MPCCResult:
    """Result from C++ MPCC solver, matching pympc_core.solver.MPCCResult."""
    v_cmd: float = 0.0
    delta_cmd: float = 0.0
    omega_cmd: float = 0.0
    predicted_trajectory: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    solve_time: float = 0.0
    success: bool = False
    cost: float = float('inf')


# ============================================================================
# C++ MPCC Solver wrapper
# ============================================================================

class CppMPCCSolver:
    """
    Python wrapper for the C++ MPCC solver via ctypes.

    Drop-in replacement for pympc_core.MPCCSolver with the same solve() API.
    Uses the high-performance C++ implementation for ~500x speedup over CasADi.
    """

    def __init__(self, config):
        """
        Initialize the C++ solver.

        Args:
            config: MPCCConfig from pympc_core.solver (or compatible object)
        """
        if _LIB is None:
            raise RuntimeError("C++ MPCC solver library not available")

        self._config = config

        # Create C params struct
        params = MPCCParams()
        params.horizon = getattr(config, 'horizon', 15)
        params.dt = getattr(config, 'dt', 0.1)
        params.wheelbase = getattr(config, 'wheelbase', 0.256)
        params.max_velocity = getattr(config, 'max_velocity', 0.5)
        params.min_velocity = max(0.0, getattr(config, 'min_velocity', 0.0))
        params.max_steering = getattr(config, 'max_steering', 0.45)
        params.max_acceleration = getattr(config, 'max_acceleration', 0.8)
        params.max_steering_rate = getattr(config, 'max_steering_rate', 0.8)
        params.reference_velocity = getattr(config, 'reference_velocity', 0.45)
        params.contour_weight = getattr(config, 'contour_weight', 25.0)
        params.lag_weight = getattr(config, 'lag_weight', 5.0)
        params.velocity_weight = getattr(config, 'velocity_weight', 2.0)
        params.steering_weight = getattr(config, 'steering_weight', 2.0)
        params.acceleration_weight = getattr(config, 'acceleration_weight', 1.0)
        params.steering_rate_weight = getattr(config, 'steering_rate_weight', 3.0)
        params.jerk_weight = getattr(config, 'jerk_weight', 0.5)
        params.robot_radius = getattr(config, 'robot_radius', 0.15)
        params.safety_margin = getattr(config, 'safety_margin', 0.12)
        params.obstacle_weight = getattr(config, 'obstacle_weight', 200.0)
        params.boundary_weight = getattr(config, 'boundary_weight', 20.0)
        params.max_sqp_iterations = getattr(config, 'max_sqp_iterations', 3)
        params.max_qp_iterations = getattr(config, 'max_qp_iterations', 50)
        params.qp_tolerance = getattr(config, 'qp_tolerance', 1e-4)

        self._solver = _LIB.mpcc_create(ctypes.byref(params))
        if not self._solver:
            raise RuntimeError("Failed to create C++ MPCC solver")

        self._horizon = params.horizon

    def __del__(self):
        if hasattr(self, '_solver') and self._solver and _LIB is not None:
            _LIB.mpcc_destroy(self._solver)
            self._solver = None

    def reset(self):
        """Reset warm-start state (call when path changes)."""
        if self._solver:
            _LIB.mpcc_reset(self._solver)

    def solve(
        self,
        x0: np.ndarray,
        path: CubicSplinePath,
        current_progress: float,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        boundary_constraints: Optional[List[Tuple[np.ndarray, float, float]]] = None,
    ) -> MPCCResult:
        """
        Solve the MPCC problem using the C++ solver.

        API matches pympc_core.MPCCSolver.solve().

        Args:
            x0: Initial state [x, y, theta, v, delta]
            path: Reference path (CubicSplinePath)
            current_progress: Current arc-length progress on path
            obstacles: List of (x, y, radius) tuples
            boundary_constraints: List of (normal_vec, b_left, b_right) per horizon step

        Returns:
            MPCCResult with optimal commands and predicted trajectory
        """
        if obstacles is None:
            obstacles = []
        if boundary_constraints is None:
            boundary_constraints = []

        cfg = self._config
        N = self._horizon

        # Prepare state array
        state = (ctypes.c_double * 5)(*x0[:5])

        # Prepare path reference points for the horizon
        n_path = N + 1
        path_arr = (MPCCPathPoint * n_path)()
        ref_v = getattr(cfg, 'reference_velocity', 0.45)
        for k in range(n_path):
            s_k = current_progress + k * ref_v * cfg.dt
            s_k = min(s_k, path.total_length - 0.01)
            ref_x, ref_y, cos_t, sin_t = path.get_path_reference(s_k)
            curvature = path.get_curvature(s_k)
            path_arr[k].x = ref_x
            path_arr[k].y = ref_y
            path_arr[k].cos_theta = cos_t
            path_arr[k].sin_theta = sin_t
            path_arr[k].curvature = curvature

        # Prepare obstacles
        n_obs = min(len(obstacles), 10)
        obs_arr = (MPCCObstacle * max(n_obs, 1))()
        for i in range(n_obs):
            obs_arr[i].x = obstacles[i][0]
            obs_arr[i].y = obstacles[i][1]
            obs_arr[i].radius = obstacles[i][2]

        # Prepare boundaries
        n_bd = min(len(boundary_constraints), N)
        bd_arr = (MPCCBoundary * max(n_bd, 1))()
        for i in range(n_bd):
            normal, b_left, b_right = boundary_constraints[i]
            bd_arr[i].nx = float(normal[0]) if hasattr(normal, '__len__') else 0.0
            bd_arr[i].ny = float(normal[1]) if hasattr(normal, '__len__') else 1.0
            bd_arr[i].b_left = float(b_left)
            bd_arr[i].b_right = float(b_right)

        # Call C++ solver
        result_c = MPCCResultC()
        ret = _LIB.mpcc_solve(
            self._solver,
            state,
            path_arr, n_path,
            obs_arr, n_obs,
            bd_arr, n_bd,
            ctypes.c_double(current_progress),
            ctypes.c_double(path.total_length),
            ctypes.byref(result_c),
        )

        # Convert to Python result
        n_pred = result_c.predicted_len
        predicted = np.zeros((n_pred, 3))
        for i in range(n_pred):
            predicted[i, 0] = result_c.predicted_x[i]
            predicted[i, 1] = result_c.predicted_y[i]
            predicted[i, 2] = result_c.predicted_theta[i]

        return MPCCResult(
            v_cmd=result_c.v_cmd,
            delta_cmd=result_c.delta_cmd,
            omega_cmd=result_c.omega_cmd,
            predicted_trajectory=predicted,
            solve_time=result_c.solve_time_us / 1e6,  # Convert us to seconds
            success=(result_c.success != 0),
        )
