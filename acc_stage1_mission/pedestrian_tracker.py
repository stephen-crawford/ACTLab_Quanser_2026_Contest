from __future__ import annotations

"""
Kalman Filter Pedestrian Tracker

Tracks pedestrians in map frame using a constant-velocity Kalman filter.
Each tracked pedestrian has:
- Smoothed (x, y) position in map frame
- Estimated (vx, vy) velocity
- On-road status based on road boundary checks

Usage:
    tracker = PedestrianKalmanTracker(road_boundaries)
    tracker.predict(dt)
    tracker.update(measurements)  # list of (x_map, y_map)
    should_stop = tracker.any_on_road()
"""

import math
import time
import numpy as np


class KalmanTrack:
    """Single pedestrian track with constant-velocity Kalman filter.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, x: float, y: float, track_id: int):
        self.track_id = track_id
        self.created_at = time.time()
        self.last_update = time.time()
        self.hits = 1          # Number of measurement updates
        self.misses = 0        # Consecutive frames without measurement

        # State vector [x, y, vx, vy]
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)

        # State covariance — start with moderate uncertainty
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])

        # Process noise — tuned for pedestrian walking (~1 m/s max)
        # Position uncertainty grows with time; velocity can change
        self.q_pos = 0.01    # Position process noise (m^2/s)
        self.q_vel = 0.5     # Velocity process noise (m^2/s^3)

        # Measurement noise — depth camera + TF uncertainty
        self.R = np.diag([0.15, 0.15])  # ~15cm measurement noise

        # Measurement matrix H: observe [x, y] from state [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def vx(self) -> float:
        return float(self.state[2])

    @property
    def vy(self) -> float:
        return float(self.state[3])

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2)

    def predict(self, dt: float):
        """Predict step: advance state by dt seconds."""
        if dt <= 0:
            return

        # State transition matrix (constant velocity)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Process noise matrix
        Q = np.array([
            [self.q_pos * dt + self.q_vel * dt**3 / 3, 0, self.q_vel * dt**2 / 2, 0],
            [0, self.q_pos * dt + self.q_vel * dt**3 / 3, 0, self.q_vel * dt**2 / 2],
            [self.q_vel * dt**2 / 2, 0, self.q_vel * dt, 0],
            [0, self.q_vel * dt**2 / 2, 0, self.q_vel * dt],
        ], dtype=np.float64)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, z_x: float, z_y: float):
        """Update step: incorporate measurement (x, y) in map frame."""
        z = np.array([z_x, z_y], dtype=np.float64)

        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        self.last_update = time.time()
        self.hits += 1
        self.misses = 0

    def age_since_update(self) -> float:
        """Seconds since last measurement update."""
        return time.time() - self.last_update


class PedestrianKalmanTracker:
    """Multi-pedestrian tracker using Kalman filters.

    Manages multiple KalmanTrack instances, handles data association,
    track creation/deletion, and road-boundary checking.
    """

    def __init__(self, road_boundaries=None):
        self.road_boundaries = road_boundaries
        self.tracks: list[KalmanTrack] = []
        self._next_id = 0
        self._last_predict_time = time.time()

        # Association parameters
        self.max_association_dist = 1.0   # Max distance (m) to associate measurement to track
        self.max_coast_time = 2.0         # Delete track after this many seconds without update
        self.min_hits_to_confirm = 2      # Require N updates before track is "confirmed"

    def predict(self, dt: float = None):
        """Predict all tracks forward by dt seconds."""
        now = time.time()
        if dt is None:
            dt = now - self._last_predict_time
        self._last_predict_time = now

        for track in self.tracks:
            track.predict(dt)

        # Prune stale tracks
        self.tracks = [t for t in self.tracks if t.age_since_update() < self.max_coast_time]

    def update(self, measurements: list[tuple[float, float]]):
        """Update tracks with new measurements.

        Args:
            measurements: List of (x_map, y_map) pedestrian positions in map frame
        """
        if not measurements:
            # Increment miss count for all tracks
            for track in self.tracks:
                track.misses += 1
            return

        # Build cost matrix (Euclidean distance between tracks and measurements)
        n_tracks = len(self.tracks)
        n_meas = len(measurements)

        if n_tracks == 0:
            # No existing tracks — create new ones for all measurements
            for mx, my in measurements:
                self._create_track(mx, my)
            return

        # Simple greedy nearest-neighbor association
        # (Hungarian algorithm would be better for many tracks, but overkill here)
        used_tracks = set()
        used_meas = set()

        # Compute all distances
        dists = []
        for ti, track in enumerate(self.tracks):
            for mi, (mx, my) in enumerate(measurements):
                d = math.sqrt((track.x - mx)**2 + (track.y - my)**2)
                dists.append((d, ti, mi))

        # Sort by distance, assign greedily
        dists.sort(key=lambda x: x[0])
        for d, ti, mi in dists:
            if ti in used_tracks or mi in used_meas:
                continue
            if d > self.max_association_dist:
                break  # All remaining are too far
            # Associate
            mx, my = measurements[mi]
            self.tracks[ti].update(mx, my)
            used_tracks.add(ti)
            used_meas.add(mi)

        # Increment miss for unmatched tracks
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.misses += 1

        # Create new tracks for unmatched measurements
        for mi, (mx, my) in enumerate(measurements):
            if mi not in used_meas:
                self._create_track(mx, my)

    def _create_track(self, x: float, y: float):
        """Create a new track."""
        track = KalmanTrack(x, y, self._next_id)
        self._next_id += 1
        self.tracks.append(track)

    def is_on_road(self, track: KalmanTrack, margin: float = 0.15) -> bool:
        """Check if a track's smoothed position is on the road."""
        if self.road_boundaries is None:
            return True  # Can't check — assume on road (safe)

        segment = self.road_boundaries.get_active_segment(track.x, track.y)
        if segment is not None:
            return True

        for seg in self.road_boundaries.segments:
            if seg.contains_point(track.x, track.y, margin=margin):
                return True

        return False

    def any_on_road(self) -> tuple[bool, str]:
        """Check if any confirmed pedestrian is on the road.

        Returns:
            (should_stop, reason) tuple
        """
        for track in self.tracks:
            # Only consider confirmed tracks (enough hits, recent update)
            if track.hits < self.min_hits_to_confirm:
                continue
            if track.age_since_update() > 0.5:
                continue  # Stale track, don't stop for it

            if self.is_on_road(track):
                speed_str = f", speed={track.speed:.2f}m/s" if track.speed > 0.05 else ""
                return True, (f"Pedestrian on road [track {track.track_id}] "
                              f"at ({track.x:.2f}, {track.y:.2f}){speed_str}")

        return False, ""

    def get_confirmed_tracks(self) -> list[KalmanTrack]:
        """Get all confirmed (enough hits) tracks."""
        return [t for t in self.tracks
                if t.hits >= self.min_hits_to_confirm
                and t.age_since_update() < self.max_coast_time]
