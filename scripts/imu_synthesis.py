#!/usr/bin/env python3
"""
Synthesise body-frame IMU measurements from a ground-truth pose trajectory.

Shared by convert_tum_rgbd.py, convert_eth3d.py, convert_icl_nuim.py and
regenerate_bag_imu.py.  It replaces the per-converter synthesise_imu(),
which had two defects:

  1. Gyro: a one-step rotation increment R_{k+1} R_k^{-1} was divided by
     2*dt, so every interior angular-rate sample was half the true rate.
  2. Accel: double-differentiating cubic-interpolated mocap positions at
     200 Hz amplified mocap noise to ~3 m/s^2 RMS, about 20x the nominal
     BMI055 white noise that the synthetic IMU is meant to model.

Method: the trajectory is resampled to a uniform grid at the IMU rate
(cubic spline for position, Slerp for orientation) and zero-phase low-pass
filtered before differentiation.  The body rate uses a true central
difference, Log(R_{k-1}^T R_{k+1}) / (2 dt).  White Gaussian noise with the
BMI055-like densities below is then added from a seeded generator.

Conventions (unchanged from the original converters): GT rows are
[t, tx, ty, tz, qx, qy, qz, qw] giving the body pose in a z-up world frame;
the accelerometer measures R^T (a_world + [0, 0, 9.81]).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

IMU_RATE = 200.0                       # Hz
GRAVITY = np.array([0.0, 0.0, 9.81])   # m/s^2, +Z up in world frame
ACC_NOISE_DENSITY = 0.012              # m/s^2/sqrt(Hz) — BMI055 spec
GYR_NOISE_DENSITY = 0.004              # rad/s/sqrt(Hz)
CUTOFF_HZ = 10.0                       # low-pass cutoff applied before differentiating


def synthesise_imu(gt, imu_rate=IMU_RATE, noise_scale=1.0, seed=0,
                   cutoff_hz=CUTOFF_HZ, imu_t=None):
    """
    Args:
        gt: (N,8) array [t, tx, ty, tz, qx, qy, qz, qw]
        imu_rate: rate of the uniform working grid and of the default output (Hz)
        noise_scale: multiplier on the white-noise stddev (1.0 = nominal BMI055,
                     0.0 = noise-free)
        seed: seed for the noise generator
        cutoff_hz: zero-phase Butterworth low-pass cutoff (Hz); None disables it
        imu_t: optional output timestamps within the GT span; defaults to
               np.arange(t_first, t_last, 1/imu_rate) as before

    Returns:
        imu_data: (M,7) array [t, wx, wy, wz, ax, ay, az]
                  angular velocity in body frame (rad/s)
                  specific force in body frame (m/s^2), with gravity
    """
    gt = gt[np.argsort(gt[:, 0])]
    gt = gt[np.r_[True, np.diff(gt[:, 0]) > 0]]   # drop duplicate stamps
    gt_pos = gt[:, 1:4]
    rots_gt = Rotation.from_quat(gt[:, 4:8])      # scipy quat: (x,y,z,w)

    dt = 1.0 / imu_rate
    if imu_t is None:
        imu_t = np.arange(gt[0, 0], gt[-1, 0], dt)
    imu_t = np.asarray(imu_t, dtype=float)

    # Interpolate in time relative to the first GT sample: absolute stamps
    # (~1e9 s) resolve only ~2e-7 s in float64, which differentiation amplifies.
    t_ref = gt[0, 0]
    gt_t = gt[:, 0] - t_ref
    t_out = imu_t - t_ref
    t_grid = dt * np.arange(int(np.floor(gt_t[-1] / dt)) + 1)

    pos = CubicSpline(gt_t, gt_pos, axis=0)(t_grid)
    rots = Slerp(gt_t, rots_gt)(t_grid)

    # ── Angular velocity (body frame), true central difference ──────────
    omega = np.empty((len(t_grid), 3))
    omega[1:-1] = (rots[:-2].inv() * rots[2:]).as_rotvec() / (2 * dt)
    omega[0] = (rots[0].inv() * rots[1]).as_rotvec() / dt
    omega[-1] = (rots[-2].inv() * rots[-1]).as_rotvec() / dt

    if cutoff_hz is not None:
        sos = butter(4, cutoff_hz, fs=imu_rate, output='sos')
        pos = sosfiltfilt(sos, pos, axis=0)
        omega = sosfiltfilt(sos, omega, axis=0)

    # ── Specific force (body frame): R^T (a_world + g_up) ───────────────
    acc_world = CubicSpline(t_grid, pos, axis=0)(t_grid, 2)
    acc_body = rots.inv().apply(acc_world + GRAVITY)

    # ── Evaluate at the requested timestamps ────────────────────────────
    omega = CubicSpline(t_grid, omega, axis=0)(t_out)
    acc_body = CubicSpline(t_grid, acc_body, axis=0)(t_out)

    # ── Add noise ───────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    omega += rng.standard_normal(omega.shape) * (
        GYR_NOISE_DENSITY * np.sqrt(imu_rate) * noise_scale)
    acc_body += rng.standard_normal(acc_body.shape) * (
        ACC_NOISE_DENSITY * np.sqrt(imu_rate) * noise_scale)

    return np.column_stack([imu_t, omega, acc_body])
