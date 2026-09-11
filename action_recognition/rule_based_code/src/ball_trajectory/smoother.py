"""RTS (Rauch-Tung-Striebel) offline Kalman smoothing with gravity input.

State  x = [x, y, z, vx, vy, vz]^T, dt = 1/fps.

Transition (exact for free flight):
    x_k = F x_{k-1} + u
    F = [[I3, dt*I3], [0, I3]]
    u = [0, 0, -0.5*g*dt^2, 0, 0, -g*dt]^T     (deterministic gravity input)

Measurements are the raw triangulations only (ballistic gap fills are
predictions, not observations). Process noise is piecewise constant:
lower sigma during flight/dribble ballistic segments, higher elsewhere, and
inflated one step at bounce frames so the vertical velocity may flip.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81

def _process_noise_per_step(
    sigma_base: float,
    is_bounce: bool,
    bounce_q_multiplier: float,
    dt: float,
) -> np.ndarray:
    """Piecewise-constant white acceleration Q for one step (6x6)."""
    sigma = float(sigma_base)
    if is_bounce:
        sigma *= float(bounce_q_multiplier)
    q = sigma * sigma
    dt2 = dt * dt
    return q * np.array(
        [
            [dt2 * dt2 / 4, 0, 0, dt2 * dt / 2, 0, 0],
            [0, dt2 * dt2 / 4, 0, 0, dt2 * dt / 2, 0],
            [0, 0, dt2 * dt2 / 4, 0, 0, dt2 * dt / 2],
            [dt2 * dt / 2, 0, 0, dt2, 0, 0],
            [0, dt2 * dt / 2, 0, 0, dt2, 0],
            [0, 0, dt2 * dt / 2, 0, 0, dt2],
        ],
        dtype=np.float64,
    )


def rts_smooth(
    frames: np.ndarray,
    positions: np.ndarray,
    observed: np.ndarray,
    views: np.ndarray,
    seg_kinds: np.ndarray,
    bounce_frames: set[int],
    fps: float,
    measurement_noise_m: float = 0.03,
    process_noise_flight_m_s2: float = 1.5,
    process_noise_other_m_s2: float = 6.0,
    bounce_q_multiplier: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother over a constant-gravity process model.

    Forward Kalman pass with per-frame process noise (flight vs other
    states, bounce amplification), then the backward RTS recursion.
    """
    dt = 1.0 / fps
    n = len(frames)
    x = np.zeros((n, 6))
    P = np.zeros((n, 6, 6))
    # state: [x, y, z, vx, vy, vz]
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    B = np.zeros(6)
    B[2] = -0.5 * 9.81 * dt * dt
    B[5] = -9.81 * dt
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * (measurement_noise_m ** 2)
    Q = np.eye(6)
    Q[3, 3] = Q[4, 4] = Q[5, 5] = (process_noise_other_m_s2 * dt) ** 2
    Q[:3, :3] *= (dt * dt)
    # initial
    first = None
    for i in range(n):
        if observed[i] and np.isfinite(positions[i]).all():
            first = i
            break
    if first is None:
        return np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    x[first] = np.concatenate([positions[first], np.zeros(3)])
    P[first] = np.eye(6) * 0.1
    # forward pass
    for i in range(first, n - 1):
        q = Q.copy()
        if seg_kinds[i] == 1:  # flight
            q[5, 5] = (process_noise_flight_m_s2 * dt) ** 2
        if i in bounce_frames:
            q = q * bounce_q_multiplier
        x_pred = F @ x[i] + B
        P_pred = F @ P[i] @ F.T + q
        if observed[i + 1] and np.isfinite(positions[i + 1]).all():
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = positions[i + 1] - H @ x_pred
            x[i + 1] = x_pred + K @ innov
            P[i + 1] = P_pred - K @ H @ P_pred
        else:
            x[i + 1] = x_pred
            P[i + 1] = P_pred
    # backward pass
    smoothed = x.copy()
    for i in range(n - 2, first - 1, -1):
        C = P[i] @ F.T @ np.linalg.inv(F @ P[i] @ F.T + Q)
        smoothed[i] = x[i] + C @ (smoothed[i + 1] - (F @ x[i] + B))
    return smoothed[:, 0:3], smoothed[:, 3:6]

    """Forward-backward Kalman smoothing over the contiguous frame grid.

    Args:
        frames: (N,) grid frame indices.
        positions: (N, 3) with NaN where unmeasured.
        observed: (N,) bool — measurement available at this frame.
        views: (N,) int — triangulation view count (0 where unmeasured).
        seg_kinds: (N,) object array — "flight"/"dribble"/"" per grid frame.
        bounce_frames: set of grid-frame indices with bounces.

    Returns:
        (smoothed_positions (N,3), smoothed_velocities (N,3)); NaN before the
        first measurement.
    """
    n = len(frames)
    dt = 1.0 / float(fps)
    g = GRAVITY

    F = np.eye(6)
    F[0:3, 3:6] = dt * np.eye(3)
    H = np.zeros((3, 6))
    H[0:3, 0:3] = np.eye(3)
    u = np.zeros(6)
    u[2] = -0.5 * g * dt * dt
    u[5] = -g * dt

    sigma_meas = float(measurement_noise_m)
    sigma_flight = float(process_noise_flight_m_s2)
    sigma_other = float(process_noise_other_m_s2)

    x_filt = np.zeros((n, 6))
    P_filt = np.zeros((n, 6, 6))
    x_pred = np.zeros((n, 6))
    P_pred = np.zeros((n, 6, 6))
    has_support = np.zeros(n, dtype=bool)

    first_meas = int(np.argmax(observed)) if observed.any() else None
    if first_meas is None:
        empty = np.full((n, 3), np.nan)
        return empty, empty.copy()

    # ---- forward pass ----
    y0 = positions[first_meas]
    v0 = np.zeros(3)
    nxt = first_meas + 1
    while nxt < n and not observed[nxt]:
        nxt += 1
    if nxt < n and nxt - first_meas <= 12:
        v0 = (positions[nxt] - y0) / (float(frames[nxt] - frames[first_meas]) / fps)
    x = np.concatenate([y0, v0])
    P = np.zeros((6, 6))
    P[0:3, 0:3] = sigma_meas ** 2 * np.eye(3)
    P[3:6, 3:6] = (2.0 * sigma_meas / dt) ** 2 * np.eye(3)
    x_filt[first_meas] = x
    P_filt[first_meas] = P
    has_support[first_meas] = True

    for i in range(first_meas + 1, n):
        sigma_a = sigma_flight if seg_kinds[i] in ("flight", "dribble") else sigma_other
        Q = _process_noise_per_step(sigma_a, int(frames[i]) in bounce_frames, bounce_q_multiplier, dt)
        x_pred[i] = F @ x + u
        P_pred[i] = F @ P @ F.T + Q
        if observed[i]:
            y = positions[i]
            nviews = max(1, int(views[i]))
            R = (sigma_meas / np.sqrt(nviews)) ** 2 * np.eye(3)
            S = H @ P_pred[i] @ H.T + R
            K = P_pred[i] @ H.T @ np.linalg.inv(S)
            innovation = y - H @ x_pred[i]
            x = x_pred[i] + K @ innovation
            P = (np.eye(6) - K @ H) @ P_pred[i]
            has_support[i] = True
        else:
            x = x_pred[i].copy()
            P = P_pred[i].copy()
        x_filt[i] = x
        P_filt[i] = P

    # ---- backward pass ----
    x_smooth = np.zeros((n, 6))
    P_smooth = np.zeros((n, 6, 6))
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for i in range(n - 2, -1, -1):
        J = P_filt[i] @ F.T @ np.linalg.inv(P_pred[i + 1])
        x_smooth[i] = x_filt[i] + J @ (x_smooth[i + 1] - x_pred[i + 1])
        P_smooth[i] = P_filt[i] + J @ (P_smooth[i + 1] - P_pred[i + 1]) @ J.T

    smoothed_pos = np.full((n, 3), np.nan)
    smoothed_vel = np.full((n, 3), np.nan)
    # Only frames at/after the first measurement have support.
    for i in range(first_meas, n):
        smoothed_pos[i] = x_smooth[i, 0:3]
        smoothed_vel[i] = x_smooth[i, 3:6]
    return smoothed_pos, smoothed_vel
