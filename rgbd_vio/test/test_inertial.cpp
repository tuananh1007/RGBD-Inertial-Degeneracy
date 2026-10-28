// Unit tests for inertial_init (analytic trajectories, no ROS).
// Build: catkin target test_inertial; run: rosrun rgbd_vio test_inertial
#include "inertial_init.h"

#include <cmath>
#include <cstdio>
#include <functional>

using namespace inertial;
using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {
int failures = 0;
void check(bool ok, const char *name, double value, double tol)
{
  std::printf("%-46s %s  (value %.3g, tol %.3g)\n", name, ok ? "ok  " : "FAIL", value, tol);
  if (!ok) ++failures;
}

const Vector3d W(0.3, -0.7, 1.1);  // constant body rate, rad/s
const Matrix3d R0 = expSO3(Vector3d(0.2, -0.4, 1.0));
const Vector3d G_W = expSO3(Vector3d(0.05, -0.03, 0.0)) * Vector3d(0, 0, -9.81);  // tilted world
const double PI = 3.14159265358979;

Matrix3d rot(double t) { return R0 * expSO3(W * t); }
Vector3d pos(double t)
{
  return Vector3d(0.5 * std::sin(2 * PI * 0.5 * t), 0.3 * std::cos(2 * PI * 0.8 * t),
                  0.2 * std::sin(2 * PI * 0.3 * t));
}
Vector3d vel(double t)
{
  return Vector3d(0.5 * 2 * PI * 0.5 * std::cos(2 * PI * 0.5 * t),
                  -0.3 * 2 * PI * 0.8 * std::sin(2 * PI * 0.8 * t),
                  0.2 * 2 * PI * 0.3 * std::cos(2 * PI * 0.3 * t));
}
Vector3d accw(double t)
{
  return Vector3d(-0.5 * std::pow(2 * PI * 0.5, 2) * std::sin(2 * PI * 0.5 * t),
                  -0.3 * std::pow(2 * PI * 0.8, 2) * std::cos(2 * PI * 0.8 * t),
                  -0.2 * std::pow(2 * PI * 0.3, 2) * std::sin(2 * PI * 0.3 * t));
}

std::vector<ImuSample> imu(double t0, double t1, double rate, const Vector3d &bg,
                           const Vector3d &ba, bool moving = true)
{
  std::vector<ImuSample> s;
  for (double t = t0; t <= t1 + 1e-9; t += 1.0 / rate)
  {
    const Matrix3d R = rot(t);
    const Vector3d a = moving ? accw(t) : Vector3d::Zero();
    s.push_back({t, R.transpose() * (a - G_W) + ba, W + bg});
  }
  return s;
}
}  // namespace

int main()
{
  const Vector3d zero = Vector3d::Zero();

  // segments(): coverage and gap flagging
  {
    std::vector<ImuSample> s = imu(0.0, 1.0, 100.0, zero, zero);
    s.erase(s.begin() + 40, s.begin() + 50);  // 0.1 s hole
    double cover = 0.0;
    int gaps = 0;
    for (const Segment &g : segments(s, 0.123, 0.987, 0.025)) { cover += g.dt; gaps += g.gap; }
    check(std::abs(cover - (0.987 - 0.123)) < 1e-12, "segments cover [t0,t1]", cover, 1e-12);
    check(gaps == 1, "segments flag the 0.1 s hole", gaps, 0);
  }

  // propagate(): noise-free strapdown over 2 s reproduces the trajectory
  {
    const double t0 = 1.0, t1 = 3.0;
    NavState x;
    x.R = rot(t0); x.p = pos(t0); x.v = vel(t0); x.g = G_W;
    propagate(x, segments(imu(t0 - 0.1, t1 + 0.1, 200.0, zero, zero), t0, t1, 0.025),
              NoiseParams(), true, true);
    check(logSO3(rot(t1).transpose() * x.R).norm() < 1e-6, "propagate: attitude", logSO3(rot(t1).transpose() * x.R).norm(), 1e-6);
    check((x.p - pos(t1)).norm() < 2e-3, "propagate: position over 2 s", (x.p - pos(t1)).norm(), 2e-3);
    check((x.v - vel(t1)).norm() < 2e-3, "propagate: velocity over 2 s", (x.v - vel(t1)).norm(), 2e-3);
    const Eigen::SelfAdjointEigenSolver<Mat19> es(0.5 * (x.P + x.P.transpose()));
    check(es.eigenvalues().minCoeff() > -1e-12, "propagate: covariance stays PSD", es.eigenvalues().minCoeff(), 1e-12);
  }

  // Preintegration matches the analytic relative motion, and its bias
  // Jacobians predict a re-integration with a perturbed gyro bias.
  {
    const double ti = 0.5, tj = 0.9;
    const auto segs = segments(imu(0.0, 1.5, 200.0, zero, zero), ti, tj, 0.025);
    const Preintegration p(segs, zero, zero);
    const double h = tj - ti;
    const Matrix3d Ri = rot(ti);
    const Vector3d dV = Ri.transpose() * (vel(tj) - vel(ti) - G_W * h);
    const Vector3d dP = Ri.transpose() * (pos(tj) - pos(ti) - vel(ti) * h - 0.5 * G_W * h * h);
    check(logSO3(p.dR.transpose() * Ri.transpose() * rot(tj)).norm() < 1e-9, "preint: dR", logSO3(p.dR.transpose() * Ri.transpose() * rot(tj)).norm(), 1e-9);
    check((p.dV - dV).norm() < 1e-3, "preint: dV", (p.dV - dV).norm(), 1e-3);
    check((p.dP - dP).norm() < 1e-4, "preint: dP", (p.dP - dP).norm(), 1e-4);
    const Vector3d dbg(2e-3, -1e-3, 1.5e-3);
    const Preintegration q(segs, dbg, zero);
    const double eR = logSO3((p.dR * expSO3(p.JR_bg * dbg)).transpose() * q.dR).norm();
    const double eV = (p.dV + p.JV_bg * dbg - q.dV).norm(), eP = (p.dP + p.JP_bg * dbg - q.dP).norm();
    check(eR < 1e-6 && eV < 1e-6 && eP < 1e-6, "preint: first-order gyro-bias Jacobians", std::max({eR, eV, eP}), 1e-6);
  }

  // initStatic(): tilted, stationary sensor with a gyro bias
  {
    const Vector3d bg(0.004, -0.002, 0.003);
    std::vector<ImuSample> s;
    const Matrix3d R = rot(0.3);
    for (int k = 0; k < 200; ++k) s.push_back({k * 0.005, R.transpose() * (-G_W), bg});
    const StaticResult r = initStatic(s, 9.81, 0.03, 0.25);
    const double ang = std::acos(std::min(1.0, (R * r.g_imu).normalized().dot(G_W.normalized())));
    check(r.ok, "initStatic: accepts a stationary window", r.ok, 0);
    check(ang < 1e-6, "initStatic: gravity direction (rad)", ang, 1e-6);
    check((r.bg - bg).norm() < 1e-12, "initStatic: gyro bias", (r.bg - bg).norm(), 1e-12);
    std::vector<ImuSample> m = imu(0.0, 1.0, 200.0, zero, zero);
    check(!initStatic(m, 9.81, 0.03, 0.25).ok, "initStatic: rejects a moving window", 0, 0);
  }

  // initDynamic(): 30 Hz poses + 200 Hz IMU with a gyro bias, tilted gravity
  {
    const Vector3d bg(0.01, -0.02, 0.015);
    const auto s = imu(0.0, 2.0, 200.0, bg, zero);
    std::vector<Frame> f;
    for (double t = 0.2; t <= 1.7 + 1e-9; t += 1.0 / 30.0) f.push_back({t, rot(t), pos(t)});
    const DynamicResult r = initDynamic(f, s, 9.81, 0.25, 0.5);
    check(r.ok, r.ok ? "initDynamic: succeeds" : r.why.c_str(), r.excitation, 0.25);
    if (r.ok)
    {
      const double ang = std::acos(std::min(1.0, r.g_w.normalized().dot(G_W.normalized())));
      check((r.bg - bg).norm() < 1e-3, "initDynamic: gyro bias", (r.bg - bg).norm(), 1e-3);
      check(ang < 2e-3, "initDynamic: gravity direction (rad)", ang, 2e-3);
      check((r.v.back() - vel(f.back().t)).norm() < 0.02, "initDynamic: last-frame velocity", (r.v.back() - vel(f.back().t)).norm(), 0.02);
    }
    std::vector<ImuSample> still;
    std::vector<Frame> fs;
    const Matrix3d R = rot(0.0);
    for (int k = 0; k <= 400; ++k) still.push_back({k * 0.005, R.transpose() * (-G_W), zero});
    for (double t = 0.2; t <= 1.7; t += 1.0 / 30.0) fs.push_back({t, R, Vector3d::Zero()});
    const DynamicResult rs = initDynamic(fs, still, 9.81, 0.25, 0.5);
    check(!rs.ok && rs.why == "insufficient excitation", "initDynamic: refuses a static window", rs.excitation, 0.25);
  }

  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
