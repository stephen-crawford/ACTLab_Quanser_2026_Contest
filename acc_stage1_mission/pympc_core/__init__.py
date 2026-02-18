"""
PyMPC Core - Essential MPC algorithms extracted from PyMPC framework.

This module provides the core MPCC (Model Predictive Contouring Control)
solver with proper CasADi constraint handling, adapted for the QCar2
Ackermann vehicle model.

Key components:
- CubicSplinePath: Smooth cubic spline path representation
- AckermannDynamics: QCar2 vehicle dynamics with RK4 integration
- MPCCSolver: CasADi-based MPCC solver (fresh Opti per solve)
- CppMPCCSolver: High-performance C++ solver via ctypes (~500x faster)
"""

from .dynamics import AckermannDynamics
from .spline_path import CubicSplinePath
from .solver import MPCCSolver, MPCCConfig, MPCCResult

# Try to import C++ solver
try:
    from .mpcc_cpp import CppMPCCSolver, is_available as cpp_solver_available
    HAS_CPP_SOLVER = cpp_solver_available()
except ImportError:
    HAS_CPP_SOLVER = False
    CppMPCCSolver = None

__all__ = [
    'AckermannDynamics',
    'CubicSplinePath',
    'MPCCSolver',
    'CppMPCCSolver',
    'MPCCConfig',
    'MPCCResult',
    'HAS_CPP_SOLVER',
]
