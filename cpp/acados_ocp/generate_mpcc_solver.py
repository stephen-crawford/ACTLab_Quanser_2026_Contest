#!/usr/bin/env python3
"""
Generate acados MPCC solver C code for QCar2.

Defines the MPCC OCP using CasADi symbolics matching the PolyCtrl 2025 reference:
- State (nx=4): [X, Y, psi, theta_A] — position, heading, arc-length progress
- Controls (nu=3): [V, delta, V_theta] — speed, steering, progress speed
- Dynamics: bicycle model + theta_A integration (MPCC.py lines 52-57)
- Cost: EXTERNAL type with GAUSS_NEWTON Hessian approximation using custom Hessian
- Constraints: box on controls, nonlinear obstacle avoidance

The generated C code goes to c_generated_code/ and is compiled into a static library.

Usage:
    export ACADOS_SOURCE_DIR=/home/stephen/acados
    cd /home/stephen/quanser-acc/cpp/acados_ocp
    python3 generate_mpcc_solver.py
"""

import os
import sys
import numpy as np

# Set acados path before imports
ACADOS_DIR = os.environ.get('ACADOS_SOURCE_DIR', '/home/stephen/acados')
os.environ['ACADOS_SOURCE_DIR'] = ACADOS_DIR

from casadi import SX, vertcat, atan, tan, cos, sin, atan2, sqrt, fmod, DM
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# ============================================================================
# Vehicle parameters (matching mpcc_types.h Config defaults)
# ============================================================================
L = 0.256        # wheelbase [m]
N = 10           # horizon steps
DT = 0.1         # time step [s]

# Control limits
V_MIN = 0.0
V_MAX = 1.2
DELTA_MAX = 0.45  # ±25.8° hardware servo limit
VTHETA_MIN = 0.0  # arc-length speed >= 0 (no backward progress)
VTHETA_MAX = 2.0  # max arc-length speed

# State limits
X_MIN = -50.0
X_MAX = 50.0
PSI_MIN = -1e4    # unconstrained heading
PSI_MAX = 1e4
THETA_MIN = 0.0
THETA_MAX = 50.0  # max path length (will be updated at runtime)

# ============================================================================
# Parameter indices (per-stage parameters passed at runtime)
# ============================================================================
# Spline references: x_ref, y_ref, dx_ref, dy_ref (4 doubles)
# Weights: q_c, q_l, gamma, R_v_accel, R_v_steer, R_vtheta, R_ref_vel, R_ref_steer (8 doubles)
# References: v_ref, delta_ref (2 doubles)
# Previous controls: v_prev, delta_prev, vtheta_prev (3 doubles)
# Obstacle: obs_x, obs_y, obs_r (3 doubles)
# Total: 20 doubles per stage
N_PARAMS = 20

# Parameter slicing
IDX_XREF = 0
IDX_YREF = 1
IDX_DXREF = 2
IDX_DYREF = 3
IDX_QC = 4
IDX_QL = 5
IDX_GAMMA = 6
IDX_RV_ACCEL = 7
IDX_RV_STEER = 8
IDX_RV_THETA = 9
IDX_RREF_VEL = 10
IDX_RREF_STEER = 11
IDX_VREF = 12
IDX_DELTAREF = 13
IDX_VPREV = 14
IDX_DELTAPREV = 15
IDX_VTHETAPREV = 16
IDX_OBSX = 17
IDX_OBSY = 18
IDX_OBSR = 19


def create_mpcc_model():
    """Create the MPCC bicycle model with arc-length progress state."""
    model = AcadosModel()
    model.name = 'mpcc_qcar2'

    # States: [X, Y, psi, theta_A]
    X_pos = SX.sym('X_pos')
    Y_pos = SX.sym('Y_pos')
    psi = SX.sym('psi')
    theta_A = SX.sym('theta_A')  # arc-length progress along path
    x = vertcat(X_pos, Y_pos, psi, theta_A)

    # Controls: [V, delta, V_theta]
    V = SX.sym('V')
    delta = SX.sym('delta')
    V_theta = SX.sym('V_theta')  # arc-length progress speed
    u = vertcat(V, delta, V_theta)

    # State derivatives (explicit ODE)
    # Bicycle dynamics (MPCC.py lines 52-57)
    beta = atan(tan(delta) / 2.0)
    dX = V * cos(psi + beta)
    dY = V * sin(psi + beta)
    dpsi = V / L * tan(delta) * cos(beta)
    dtheta = V_theta  # arc-length integrator

    xdot = SX.sym('xdot', 4)
    f_expl = vertcat(dX, dY, dpsi, dtheta)

    model.x = x
    model.u = u
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    # Parameters (per-stage)
    p = SX.sym('p', N_PARAMS)
    model.p = p

    return model


def create_mpcc_ocp():
    """Create the MPCC OCP matching the PolyCtrl 2025 reference formulation.

    Uses EXTERNAL cost type with EXACT Hessian and higher Levenberg-Marquardt
    regularization for numerical stability.
    """
    ocp = AcadosOcp()

    # Model
    model = create_mpcc_model()
    ocp.model = model

    # Dimensions
    nx = 4
    nu = 3
    ocp.dims.N = N

    # Extract symbolic variables
    x = model.x
    u = model.u
    p = model.p

    X_pos = x[0]
    Y_pos = x[1]
    psi = x[2]
    theta_A = x[3]
    V = u[0]
    delta = u[1]
    V_theta = u[2]

    # Extract parameters
    x_ref = p[IDX_XREF]
    y_ref = p[IDX_YREF]
    dx_ref = p[IDX_DXREF]
    dy_ref = p[IDX_DYREF]
    q_c = p[IDX_QC]
    q_l = p[IDX_QL]
    gamma = p[IDX_GAMMA]
    R_v_accel = p[IDX_RV_ACCEL]
    R_v_steer = p[IDX_RV_STEER]
    R_v_theta = p[IDX_RV_THETA]
    R_ref_vel = p[IDX_RREF_VEL]
    R_ref_steer = p[IDX_RREF_STEER]
    v_ref = p[IDX_VREF]
    delta_ref = p[IDX_DELTAREF]
    v_prev = p[IDX_VPREV]
    delta_prev = p[IDX_DELTAPREV]
    vtheta_prev = p[IDX_VTHETAPREV]

    # ---- Cost function (EXTERNAL type) ----
    # Contouring/lag errors (MPCC.py lines 46-50)
    phi = atan2(dy_ref, dx_ref)
    e_c = sin(phi) * (X_pos - x_ref) - cos(phi) * (Y_pos - y_ref)
    e_l = -cos(phi) * (X_pos - x_ref) - sin(phi) * (Y_pos - y_ref)

    # Stage cost (MPCC.py lines 86-98)
    cost_contouring = q_c * e_c**2 + q_l * e_l**2

    # Progress reward: -gamma * V_theta * DT
    # To make this work with exact Hessian, we use the form:
    # gamma * DT * (VTHETA_MAX - V_theta)^2 / (2 * VTHETA_MAX) - gamma * DT * VTHETA_MAX / 2
    # This is equivalent to a quadratic penalty that pushes V_theta toward VTHETA_MAX.
    # The constant term doesn't affect optimization.
    # Simpler approach: just use -gamma * V_theta * DT directly. The Hessian w.r.t. V_theta
    # is zero for this linear term, but with Levenberg-Marquardt regularization this works.
    cost_progress = -gamma * V_theta * DT

    # Control smoothness: (u[k] - u[k-1])^2 via parameters
    cost_smooth_v = R_v_accel * (V - v_prev)**2
    cost_smooth_delta = R_v_steer * (delta - delta_prev)**2
    cost_smooth_vtheta = R_v_theta * (V_theta - vtheta_prev)**2

    # Velocity/steering tracking (reference R_ref, applied conditionally via weights)
    cost_ref_vel = R_ref_vel * (V - v_ref)**2
    cost_ref_steer = R_ref_steer * (delta - delta_ref)**2

    stage_cost = (cost_contouring + cost_progress +
                  cost_smooth_v + cost_smooth_delta + cost_smooth_vtheta +
                  cost_ref_vel + cost_ref_steer)

    # Terminal cost: 2x contouring/lag weight at final stage
    terminal_cost = 2.0 * (q_c * e_c**2 + q_l * e_l**2)

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # ---- Constraints ----

    # Initial state constraint (set at runtime)
    ocp.constraints.x0 = np.zeros(nx)

    # Box constraints on controls
    ocp.constraints.lbu = np.array([V_MIN, -DELTA_MAX, VTHETA_MIN])
    ocp.constraints.ubu = np.array([V_MAX, DELTA_MAX, VTHETA_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # Box constraints on states (loose, mainly theta_A bounds)
    ocp.constraints.lbx = np.array([X_MIN, X_MIN, PSI_MIN, THETA_MIN])
    ocp.constraints.ubx = np.array([X_MAX, X_MAX, PSI_MAX, THETA_MAX])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    # Terminal state box constraints
    ocp.constraints.lbx_e = np.array([X_MIN, X_MIN, PSI_MIN, THETA_MIN])
    ocp.constraints.ubx_e = np.array([X_MAX, X_MAX, PSI_MAX, THETA_MAX])
    ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

    # Nonlinear obstacle constraint: (X-ox)^2 + (Y-oy)^2 >= (or + robot_r)^2
    # Formulated as h(x,u,p) >= 0:  (X-ox)^2 + (Y-oy)^2 - (or + 0.13)^2 >= 0
    obs_x = p[IDX_OBSX]
    obs_y = p[IDX_OBSY]
    obs_r = p[IDX_OBSR]
    robot_r = 0.13  # robot_radius from Config

    h_obs = (X_pos - obs_x)**2 + (Y_pos - obs_y)**2 - (obs_r + robot_r)**2

    # Path constraints (nonlinear)
    ocp.model.con_h_expr = h_obs
    ocp.constraints.lh = np.array([0.0])      # >= 0
    ocp.constraints.uh = np.array([1e6])       # no upper bound

    # Terminal path constraint
    ocp.model.con_h_expr_e = h_obs
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([1e6])

    # Slack variables for soft obstacle constraint
    # L1 + L2 penalties for constraint violation
    ocp.constraints.idxsh = np.array([0])
    Zl = np.array([100.0])   # L2 penalty lower
    Zu = np.array([0.0])     # L2 penalty upper (not needed for >= constraint)
    zl = np.array([50.0])    # L1 penalty lower
    zu = np.array([0.0])     # L1 penalty upper
    ocp.cost.Zl = Zl
    ocp.cost.Zu = Zu
    ocp.cost.zl = zl
    ocp.cost.zu = zu

    ocp.constraints.idxsh_e = np.array([0])
    ocp.cost.Zl_e = Zl
    ocp.cost.Zu_e = Zu
    ocp.cost.zl_e = zl
    ocp.cost.zu_e = zu

    # ---- Solver options ----
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 20
    ocp.solver_options.qp_solver_iter_max = 500
    ocp.solver_options.integrator_type = 'ERK'     # Explicit Runge-Kutta
    ocp.solver_options.sim_method_num_stages = 4   # RK4
    ocp.solver_options.sim_method_num_steps = 1    # 1 step per interval
    ocp.solver_options.tf = N * DT                 # total horizon time
    ocp.solver_options.hessian_approx = 'EXACT'    # Exact Hessian for EXTERNAL cost
    ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.levenberg_marquardt = 1e-1   # Strong regularization for tight curves
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4
    ocp.solver_options.nlp_solver_tol_comp = 1e-3

    # Parameter values (will be overwritten at runtime, but needed for dimensions)
    ocp.parameter_values = np.zeros(N_PARAMS)
    # Set obstacle far away by default
    ocp.parameter_values[IDX_OBSX] = 1000.0
    ocp.parameter_values[IDX_OBSY] = 1000.0
    ocp.parameter_values[IDX_OBSR] = 0.1

    # Code generation directory
    ocp.code_export_directory = 'c_generated_code'

    return ocp


def main():
    print("Generating MPCC solver C code for QCar2...")
    print(f"  Horizon N={N}, dt={DT}s, wheelbase L={L}m")
    print(f"  State: [X, Y, psi, theta_A] (nx=4)")
    print(f"  Controls: [V, delta, V_theta] (nu=3)")
    print(f"  Parameters per stage: {N_PARAMS}")
    print(f"  Cost type: EXTERNAL with EXACT Hessian + MIRROR regularization")

    ocp = create_mpcc_ocp()

    # Generate solver
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp_mpcc.json')

    print(f"\nGenerated C code in: {ocp.code_export_directory}/")
    print("Solver created successfully.")

    # Quick validation: solve with zero initial state on straight path
    print("\nRunning validation solve...")
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    solver.set(0, 'x', x0)
    solver.constraints_set(0, 'lbx', x0)
    solver.constraints_set(0, 'ubx', x0)

    # Set parameters for straight path along x-axis
    for k in range(N + 1):
        p_k = np.zeros(N_PARAMS)
        # Reference: straight path along x-axis
        s = k * 0.045  # ~0.45 m/s * 0.1s spacing
        p_k[IDX_XREF] = s
        p_k[IDX_YREF] = 0.0
        p_k[IDX_DXREF] = 1.0    # dx/ds = 1 (along x)
        p_k[IDX_DYREF] = 0.0    # dy/ds = 0
        # Weights
        p_k[IDX_QC] = 15.0      # contour
        p_k[IDX_QL] = 10.0      # lag
        p_k[IDX_GAMMA] = 1.0    # progress
        p_k[IDX_RV_ACCEL] = 0.01
        p_k[IDX_RV_STEER] = 1.5
        p_k[IDX_RV_THETA] = 0.1
        p_k[IDX_RREF_VEL] = 15.0 if k == 0 else 0.0  # R_ref at k=0 only
        p_k[IDX_RREF_STEER] = 0.05
        p_k[IDX_VREF] = 0.45
        p_k[IDX_DELTAREF] = 0.0
        # Previous controls (for k=0)
        p_k[IDX_VPREV] = 0.2
        p_k[IDX_DELTAPREV] = 0.0
        p_k[IDX_VTHETAPREV] = 0.0
        # Obstacle far away
        p_k[IDX_OBSX] = 1000.0
        p_k[IDX_OBSY] = 1000.0
        p_k[IDX_OBSR] = 0.1

        if k < N:
            solver.set(k, 'p', p_k)
            # Initialize controls
            solver.set(k, 'u', np.array([0.3, 0.0, 0.3]))
            # Initialize states
            solver.set(k, 'x', np.array([s, 0.0, 0.0, s]))
        else:
            solver.set(k, 'p', p_k)
            solver.set(k, 'x', np.array([s, 0.0, 0.0, s]))

    status = solver.solve()
    print(f"  Solver status: {status} ({'success' if status == 0 else 'warning/fail'})")

    u0 = solver.get(0, 'u')
    x1 = solver.get(1, 'x')
    print(f"  u[0] = [V={u0[0]:.3f}, delta={u0[1]:.4f}, V_theta={u0[2]:.3f}]")
    print(f"  x[1] = [X={x1[0]:.3f}, Y={x1[1]:.4f}, psi={x1[2]:.4f}, theta={x1[3]:.3f}]")

    # Print solve time
    t_solve = solver.get_stats('time_tot')
    print(f"  Solve time: {t_solve*1e6:.0f} us")

    # Print SQP iteration stats
    sqp_iters = solver.get_stats('sqp_iter')
    print(f"  SQP iterations: {sqp_iters}")

    # Print full trajectory
    print("\n  Predicted trajectory:")
    for k in range(N+1):
        xk = solver.get(k, 'x')
        if k < N:
            uk = solver.get(k, 'u')
            print(f"    k={k:2d}: x=[{xk[0]:7.3f} {xk[1]:7.3f} {xk[2]:6.3f} {xk[3]:6.3f}]  u=[{uk[0]:5.3f} {uk[1]:6.3f} {uk[2]:5.3f}]")
        else:
            print(f"    k={k:2d}: x=[{xk[0]:7.3f} {xk[1]:7.3f} {xk[2]:6.3f} {xk[3]:6.3f}]")

    print("\nDone! C code is in c_generated_code/")
    print("Next: build with the C++ wrapper (acados_mpcc_solver.h)")


if __name__ == '__main__':
    main()
