/*
Inertial propagation and initialisation for the staged IMU mode
(imu/init_mode: staged).  Pure Eigen: no ROS or FAST-LIVO2 types, so the
maths can be unit-tested in isolation (test/test_inertial.cpp).

Replaces three behaviours of FAST-LIVO2's ImuProcess that fail for RGB-D
sequences:
  * gravity is averaged from the first IMU samples and then frozen, which is
    wrong when the sensor is moving at start-up;
  * every accelerometer sample is rescaled by 9.81 / |mean start-up accel|;
  * process noise is added as cov * dt^2 with a per-sample variance.
Here noise is specified as continuous-time densities (Kalibr convention),
gravity keeps a random walk, and gravity / gyro bias / velocity are
initialised either from a stationary window or by visual-inertial alignment
against depth-ICP poses (VINS-Mono style, with the scale fixed to 1 because
depth is metric).

Error-state layout matches FAST-LIVO2's StatesGroup:
  [rot 0..2 | pos 3..5 | inv-exposure 6 | vel 7..9 | bg 10..12 | ba 13..15 | g 16..18]
with rotation perturbed on the right, R <- R Exp(dtheta).
*/
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace inertial {

constexpr int kDim = 19;
using Mat19 = Eigen::Matrix<double, kDim, kDim>;

struct ImuSample
{
  double t;
  Eigen::Vector3d acc;  // specific force, m/s^2
  Eigen::Vector3d gyr;  // angular rate, rad/s
};

// Nominal state plus covariance in the layout above (exposure is carried in
// the covariance only; its mean lives in the caller's state).
struct NavState
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // body (IMU) -> world
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d g = Eigen::Vector3d::Zero();     // gravity vector in world
  Mat19 P = Mat19::Identity() * 1e-2;
};

// Continuous-time noise densities (Kalibr naming).  gap_* inflate the noise
// of any interval between consecutive samples longer than gap_thresh.
struct NoiseParams
{
  double gyr_nd = 2e-4;     // rad/s/sqrt(Hz)
  double acc_nd = 2e-3;     // m/s^2/sqrt(Hz)
  double gyr_rw = 2e-5;     // rad/s^2/sqrt(Hz)
  double acc_rw = 5e-4;     // m/s^3/sqrt(Hz)
  double grav_rw = 1e-3;    // m/s^2/sqrt(s), gravity-vector random walk
  // Ablation: legacy (FAST-LIVO2-style) discrete scaling, variance = cov * h^2 per interval, no gravity walk
  bool discrete = false;
  double gyr_cov = 0.1, acc_cov = 0.1, bgyr_cov = 1e-4, bacc_cov = 1e-4;
  double gap_thresh = 0.025;
  double gap_factor = 100.0;
};

// One constant-input interval [t, t + dt] built from consecutive samples.
struct Segment
{
  double dt;
  Eigen::Vector3d acc, gyr;
  bool gap;  // the samples bounding this interval are > gap_thresh apart
};

// Midpoint segments covering [t0, t1]; the first/last sample is held when
// the samples do not span the whole interval.  `samples` must be sorted.
std::vector<Segment> segments(const std::vector<ImuSample> &samples, double t0, double t1,
                              double gap_thresh);

// Strapdown propagation with first-order covariance propagation.
void propagate(NavState &x, const std::vector<Segment> &segs, const NoiseParams &n,
               bool est_bias, bool est_gravity);

// Gyro-only propagation for the warm-up before initialisation: attitude from
// the gyro, position from a constant-velocity model whose velocity follows a
// random walk of density vel_rw (m/s/sqrt(s)).  Accelerometers are unused.
void propagateWarmup(NavState &x, const std::vector<Segment> &segs, double gyr_nd,
                     double vel_rw);

// Forster et al. preintegration with first-order bias Jacobians.
struct Preintegration
{
  double dt = 0.0;
  Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
  Eigen::Vector3d dV = Eigen::Vector3d::Zero(), dP = Eigen::Vector3d::Zero();
  Eigen::Matrix3d JR_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d JV_bg = Eigen::Matrix3d::Zero(), JV_ba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d JP_bg = Eigen::Matrix3d::Zero(), JP_ba = Eigen::Matrix3d::Zero();

  Preintegration() = default;
  Preintegration(const std::vector<Segment> &segs, const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &ba);
};

// Stationary initialisation from a window of samples: gravity in the IMU
// frame (norm gravity_norm) and gyro bias from the means.
struct StaticResult
{
  bool ok = false;
  Eigen::Vector3d g_imu = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  double gyr_mean_norm = 0.0, acc_std_max = 0.0;
};
StaticResult initStatic(const std::vector<ImuSample> &window, double gravity_norm,
                        double max_gyr_norm, double max_acc_std);

// Visual-inertial alignment over consecutive frames with known (metric)
// body poses: gyro bias, per-frame world velocities and the gravity vector.
struct Frame
{
  double t;
  Eigen::Matrix3d R;  // body -> world
  Eigen::Vector3d p;
};
struct DynamicResult
{
  bool ok = false;
  std::string why;
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d g_w = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> v;  // one per frame
  double excitation = 0.0;         // std of dV/dt over the preintegrations
  double g_norm_free = 0.0;        // |g| before enforcing gravity_norm
};
DynamicResult initDynamic(const std::vector<Frame> &frames,
                          const std::vector<ImuSample> &samples, double gravity_norm,
                          double min_excitation, double max_g_norm_error);

// SO(3) helpers (exposed for tests).
Eigen::Matrix3d skew(const Eigen::Vector3d &w);
Eigen::Matrix3d expSO3(const Eigen::Vector3d &w);
Eigen::Vector3d logSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &w);

}  // namespace inertial
