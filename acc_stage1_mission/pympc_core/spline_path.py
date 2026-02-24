"""
Cubic spline path representation for MPCC.

Adapted from PyMPC's TKSpline/Spline2D (utils/math_tools.py).
Provides smooth path interpolation for contouring error computation,
which is critical for stable MPCC performance.

Key improvements over piecewise-linear ReferencePath:
- Continuous curvature (C2 continuity)
- Smooth tangent angles (no jumps at waypoints)
- Accurate contouring/lag error decomposition
"""

import numpy as np
from typing import List, Tuple, Optional


class CubicSplinePath:
    """
    2D cubic spline path with arc-length parameterization.

    The path is parameterized by cumulative arc length s.
    Provides position, tangent, and curvature at any s.
    """

    def __init__(self, waypoints: np.ndarray, smooth: bool = True):
        """
        Build cubic spline through waypoints.

        Args:
            waypoints: Nx2 array of [x, y] waypoints
            smooth: If True, use cubic spline. If False, piecewise linear.
        """
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.n_points = len(self.waypoints)

        if self.n_points < 2:
            raise ValueError("Need at least 2 waypoints")

        # Compute cumulative arc length
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        self.s_values = np.zeros(self.n_points)
        self.s_values[1:] = np.cumsum(seg_lengths)
        self.total_length = self.s_values[-1]

        if self.total_length < 1e-6:
            raise ValueError("Path has zero length")

        # Build spline coefficients
        if smooth and self.n_points >= 4:
            self._build_cubic_spline()
        else:
            self._use_linear = True

    def _build_cubic_spline(self):
        """Build natural cubic spline coefficients for x(s) and y(s)."""
        self._use_linear = False
        self._cx = self._compute_spline_coeffs(self.s_values, self.waypoints[:, 0])
        self._cy = self._compute_spline_coeffs(self.s_values, self.waypoints[:, 1])

    @staticmethod
    def _compute_spline_coeffs(t: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Compute natural cubic spline coefficients.

        For each segment i, the spline is:
            S_i(s) = a_i + b_i*(s-t_i) + c_i*(s-t_i)^2 + d_i*(s-t_i)^3

        Returns list of (a, b, c, d) for each segment.
        """
        n = len(t) - 1
        h = np.diff(t)

        # Set up tridiagonal system for natural spline
        A = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)

        A[0, 0] = 1.0
        A[n, n] = 1.0

        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        c = np.linalg.solve(A, rhs)

        coeffs = []
        for i in range(n):
            a = y[i]
            b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3.0
            d = (c[i + 1] - c[i]) / (3.0 * h[i])
            coeffs.append((a, b, c[i], d))

        return coeffs

    def _find_segment(self, s: float) -> Tuple[int, float]:
        """Find which segment s falls in, and the local parameter."""
        s = np.clip(s, 0.0, self.total_length)
        idx = np.searchsorted(self.s_values, s) - 1
        idx = np.clip(idx, 0, self.n_points - 2)
        ds = s - self.s_values[idx]
        return int(idx), ds

    def get_position(self, s: float) -> Tuple[float, float]:
        """Get (x, y) position at arc length s."""
        if self._use_linear:
            return self._linear_position(s)

        idx, ds = self._find_segment(s)
        ax, bx, cx, dx = self._cx[idx]
        ay, by, cy, dy = self._cy[idx]
        x = ax + bx * ds + cx * ds**2 + dx * ds**3
        y = ay + by * ds + cy * ds**2 + dy * ds**3
        return float(x), float(y)

    def get_tangent(self, s: float) -> float:
        """Get tangent angle at arc length s."""
        if self._use_linear:
            return self._linear_tangent(s)

        idx, ds = self._find_segment(s)
        _, bx, cx, dx = self._cx[idx]
        _, by, cy, dy = self._cy[idx]
        dx_ds = bx + 2 * cx * ds + 3 * dx * ds**2
        dy_ds = by + 2 * cy * ds + 3 * dy * ds**2
        return float(np.arctan2(dy_ds, dx_ds))

    def get_curvature(self, s: float) -> float:
        """Get curvature at arc length s."""
        if self._use_linear:
            return 0.0

        idx, ds = self._find_segment(s)
        _, bx, cx, dx = self._cx[idx]
        _, by, cy, dy = self._cy[idx]

        dx_ds = bx + 2 * cx * ds + 3 * dx * ds**2
        dy_ds = by + 2 * cy * ds + 3 * dy * ds**2
        d2x_ds2 = 2 * cx + 6 * dx * ds
        d2y_ds2 = 2 * cy + 6 * dy * ds

        denom = (dx_ds**2 + dy_ds**2)**1.5
        if abs(denom) < 1e-10:
            return 0.0
        kappa = float((dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denom)
        # Clamp curvature to prevent spikes at sharp corners between straight segments
        max_kappa = 5.0  # min turn radius ~0.2m
        return max(-max_kappa, min(max_kappa, kappa))

    def _linear_position(self, s: float) -> Tuple[float, float]:
        """Piecewise linear position fallback."""
        idx, ds = self._find_segment(s)
        seg_len = self.s_values[idx + 1] - self.s_values[idx]
        if seg_len < 1e-10:
            return float(self.waypoints[idx, 0]), float(self.waypoints[idx, 1])
        alpha = ds / seg_len
        x = self.waypoints[idx, 0] + alpha * (self.waypoints[idx + 1, 0] - self.waypoints[idx, 0])
        y = self.waypoints[idx, 1] + alpha * (self.waypoints[idx + 1, 1] - self.waypoints[idx, 1])
        return float(x), float(y)

    def _linear_tangent(self, s: float) -> float:
        """Piecewise linear tangent fallback."""
        idx, _ = self._find_segment(s)
        dx = self.waypoints[idx + 1, 0] - self.waypoints[idx, 0]
        dy = self.waypoints[idx + 1, 1] - self.waypoints[idx, 1]
        return float(np.arctan2(dy, dx))

    def find_closest_progress(self, x: float, y: float,
                               s_hint: Optional[float] = None) -> float:
        """
        Find arc-length progress closest to point (x, y).

        Args:
            x, y: Query point
            s_hint: Optional hint for search start (warm start)

        Returns:
            Arc-length s of closest point
        """
        point = np.array([x, y])

        # First pass: find closest waypoint
        dists = np.sqrt(np.sum((self.waypoints - point)**2, axis=1))
        closest_idx = np.argmin(dists)
        best_s = self.s_values[closest_idx]
        best_dist = dists[closest_idx]

        # Refine by checking adjacent segments
        for idx in range(max(0, closest_idx - 1), min(self.n_points - 1, closest_idx + 2)):
            if idx >= self.n_points - 1:
                continue
            p1 = self.waypoints[idx]
            p2 = self.waypoints[idx + 1]
            v = p2 - p1
            seg_len_sq = np.dot(v, v)
            if seg_len_sq < 1e-10:
                continue
            t = np.clip(np.dot(point - p1, v) / seg_len_sq, 0.0, 1.0)
            proj = p1 + t * v
            dist = np.linalg.norm(point - proj)
            if dist < best_dist:
                best_dist = dist
                seg_len = np.sqrt(seg_len_sq)
                best_s = self.s_values[idx] + t * seg_len

        return best_s

    def compute_contouring_errors(self, x: float, y: float,
                                   s: float) -> Tuple[float, float]:
        """
        Compute contouring (lateral) and lag (longitudinal) errors.

        Args:
            x, y: Vehicle position
            s: Current path progress

        Returns:
            (e_c, e_l): contouring error, lag error
        """
        ref_x, ref_y = self.get_position(s)
        theta_ref = self.get_tangent(s)

        dx = x - ref_x
        dy = y - ref_y

        cos_t = np.cos(theta_ref)
        sin_t = np.sin(theta_ref)

        e_l = cos_t * dx + sin_t * dy
        e_c = -sin_t * dx + cos_t * dy

        return e_c, e_l

    def get_path_reference(self, s: float) -> Tuple[float, float, float, float]:
        """
        Get path reference at progress s.

        Returns:
            (ref_x, ref_y, cos_tangent, sin_tangent)
        """
        ref_x, ref_y = self.get_position(s)
        theta = self.get_tangent(s)
        return ref_x, ref_y, np.cos(theta), np.sin(theta)

    def sample_points(self, n_points: int = 100) -> np.ndarray:
        """Sample n_points uniformly along the path. Returns [n_points, 2]."""
        s_vals = np.linspace(0, self.total_length, n_points)
        points = np.array([self.get_position(s) for s in s_vals])
        return points
