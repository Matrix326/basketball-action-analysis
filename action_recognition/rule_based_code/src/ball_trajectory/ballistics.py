"""Ballistic (constant-gravity) trajectory fitting and simulation.

Model:  p(t) = p0 + v0 * tau + 0.5 * g * tau^2,   tau = (t - t0) / fps
        g = (0, 0, -G) in the calibrated world frame (z up, metres).

With g known and fixed, the three axes decouple after subtracting the gravity
offset from z, so each axis is a plain linear regression in tau — closed form,
no iteration. An optional linear-drag refinement is available but disabled by
default (basketball shots at 30 fps over <1 s are well inside the regime where
drag changes the arc by a few cm).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Gravity constant (m/s^2), world frame z up.
GRAVITY = 9.81


def fit_ballistic(
    frames: np.ndarray,
    pos: np.ndarray,
    t0: float,
    fps: float,
    g: float = GRAVITY,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit p(t) = p0 + v0*tau + 0.5*g*tau^2 by per-axis least squares.

    Args:
        frames: (M,) observation frame indices (floats ok).
        pos:    (M, 3) observed world positions.
        t0:     reference frame index of p0/v0.
        fps:    frames per second.
        g:      gravity magnitude (m/s^2).

    Returns:
        (p0 (3,), v0 (3,), rms_residual_m) — rms over all 3M components.
    """
    frames = np.asarray(frames, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    if frames.ndim != 1 or pos.shape != (len(frames), 3) or len(frames) < 2:
        raise ValueError("fit_ballistic needs >=2 (frame, 3D position) samples")

    tau = (frames - t0) / float(fps)
    gravity = np.zeros(3, dtype=np.float64)
    gravity[2] = -g
    gravity_offset = 0.5 * gravity * (tau[:, None] ** 2)

    y = pos - gravity_offset  # y = p0 + v0 * tau per axis
    n = len(frames)
    sum_tau = tau.sum()
    sum_tau2 = (tau * tau).sum()
    denominator = n * sum_tau2 - sum_tau * sum_tau
    if abs(denominator) < 1e-12:
        raise ValueError("fit_ballistic: degenerate time samples")

    v0 = (n * (tau[:, None] * y).sum(axis=0) - sum_tau * y.sum(axis=0)) / denominator
    p0 = (y.sum(axis=0) - v0 * sum_tau) / n

    model = p0[None, :] + v0[None, :] * tau[:, None] + gravity_offset
    residual = model - pos
    rms = float(np.sqrt(np.mean(residual**2)))
    return p0.astype(np.float64), v0.astype(np.float64), rms


def fit_flat(
    frames: np.ndarray,
    pos: np.ndarray,
    t0: float,
    fps: float,
) -> float:
    """Constant-velocity (g=0) fit; returns RMS residual.

    Used as a discriminator: a free-falling ball fits the gravity model far
    better than the flat model, while a carried ball fits both poorly.
    """
    _, _, rms_gravity = fit_ballistic(frames, pos, t0, fps, g=GRAVITY)
    _, _, rms_flat = fit_ballistic(frames, pos, t0, fps, g=0.0)
    return rms_flat


def simulate_ballistic(
    frames: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    t0: float,
    fps: float,
    g: float = GRAVITY,
) -> np.ndarray:
    """Evaluate the ballistic model at the given frame indices -> (M, 3)."""
    frames = np.asarray(frames, dtype=np.float64)
    tau = (frames - float(t0)) / float(fps)
    gravity = np.zeros(3, dtype=np.float64)
    gravity[2] = -g
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    return (
        p0[None, :]
        + v0[None, :] * tau[:, None]
        + 0.5 * gravity[None, :] * (tau[:, None] ** 2)
    )


def fit_with_drag(
    frames: np.ndarray,
    pos: np.ndarray,
    t0: float,
    fps: float,
    g: float = GRAVITY,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], float, float]:
    """Optional nonlinear refinement adding linear drag coefficient k.

    Model with drag:  p(t) = p0 + (v0 - g/k) * (1 - exp(-k*tau))/k + g*tau/k.
    Only accepted if k lands in a physically plausible range for a basketball
    (k = g / v_terminal, v_terminal ~ 30-40 m/s -> k ~ 0.25-0.33 s^-1) and the
    RMS improves by > 20% over the drag-free fit.

    Returns (p0, v0, k, rms, rms_no_drag); p0/v0/k are None if refinement failed.
    """
    from scipy.optimize import least_squares

    frames = np.asarray(frames, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    tau = (frames - float(t0)) / float(fps)
    gravity = np.zeros(3, dtype=np.float64)
    gravity[2] = -g

    p0_no_drag, v0_no_drag, rms_no_drag = fit_ballistic(frames, pos, t0, fps, g=g)

    def model(params: np.ndarray) -> np.ndarray:
        p0, v0, k = params[:3], params[3:6], params[6]
        # (1 - exp(-k*tau))/k is the time integral of exp(-k*tau).
        zeta = (1.0 - np.exp(-k * tau)) / k
        return (
            p0[None, :]
            + (v0[None, :] - gravity[None, :] / k) * zeta[:, None]
            + gravity[None, :] * tau[:, None] / k
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        return (model(params) - pos).ravel()

    initial = np.concatenate([p0_no_drag, v0_no_drag, [0.05]])
    try:
        result = least_squares(residuals, initial, method="trf", max_nfev=200)
        if not result.success:
            return None, None, None, rms_no_drag, rms_no_drag
        p0, v0, k = result.x[:3], result.x[3:6], result.x[6]
        if not (0.0 <= k <= 0.5):
            return None, None, None, rms_no_drag, rms_no_drag
        rms = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
        if rms > 0.8 * rms_no_drag:
            return None, None, None, rms_no_drag, rms_no_drag
        return p0, v0, float(k), rms, rms_no_drag
    except Exception:
        return None, None, None, rms_no_drag, rms_no_drag


def fit_free_g(
    frames: np.ndarray,
    pos: np.ndarray,
    t0: float,
    fps: float,
    g_init: float = GRAVITY,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit p(t) = p0 + v0*tau + 0.5*[0,0,a_z]*tau^2 with a_z FREE.

    Returns (p0, v0, a_z, rms). The fitted vertical acceleration is the
    "is this free fall?" test: close to -g for a flying ball, near 0 for a
    carried ball. Robust to uneven frame sampling (the fit uses true
    timestamps), unlike finite-difference acceleration estimates.
    """
    from scipy.optimize import least_squares

    frames = np.asarray(frames, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    tau = (frames - float(t0)) / float(fps)

    p0_init, v0_init, _ = fit_ballistic(frames, pos, t0, fps, g=g_init)

    def model(params: np.ndarray) -> np.ndarray:
        p0, v0, a_z = params[:3], params[3:6], params[6]
        accel = np.zeros(3)
        accel[2] = a_z
        return (
            p0[None, :]
            + v0[None, :] * tau[:, None]
            + 0.5 * accel[None, :] * (tau[:, None] ** 2)
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        return (model(params) - pos).ravel()

    initial = np.concatenate([p0_init, v0_init, [-g_init]])
    result = least_squares(residuals, initial, method="trf", max_nfev=300)
    if not result.success:
        return (
            p0_init,
            v0_init,
            -g_init,
            float(np.sqrt(np.mean(residuals(initial) ** 2))),
        )
    rms = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
    return result.x[:3], result.x[3:6], float(result.x[6]), rms
