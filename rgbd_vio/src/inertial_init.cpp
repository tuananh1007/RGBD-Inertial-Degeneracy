/*
Inertial propagation and initialisation for the staged IMU mode.
See include/inertial_init.h for the conventions.
*/
#include "inertial_init.h"

#include <algorithm>
#include <cmath>

namespace inertial {

namespace {
// Orthonormal basis of the plane perpendicular to unit vector n.
Eigen::Matrix<double, 3, 2> tangentBasis(const Eigen::Vector3d &n)
{
  Eigen::Vector3d a = std::abs(n.z()) < 0.9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d b1 = (a - n.dot(a) * n).normalized();
  Eigen::Matrix<double, 3, 2> B;
  B << b1, n.cross(b1);
  return B;
}
}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d &w)
{
  Eigen::Matrix3d S;
  S << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
  return S;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d &w)
{
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();  // no namespace-scope Eigen objects:
                                                            // callers may run during static init
  const double th = w.norm();
  const Eigen::Matrix3d W = skew(w);
  if (th < 1e-8) return I3 + W + 0.5 * W * W;
  return I3 + std::sin(th) / th * W + (1.0 - std::cos(th)) / (th * th) * W * W;
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d &R)
{
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &w)
{
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();  // no namespace-scope Eigen objects:
                                                            // callers may run during static init
  const double th = w.norm();
  const Eigen::Matrix3d W = skew(w);
  if (th < 1e-8) return I3 - 0.5 * W;
  return I3 - (1.0 - std::cos(th)) / (th * th) * W + (th - std::sin(th)) / (th * th * th) * W * W;
}

std::vector<Segment> segments(const std::vector<ImuSample> &s, double t0, double t1,
                              double gap_thresh)
{
  std::vector<Segment> out;
  if (s.empty() || t1 <= t0) return out;
  if (t0 < s.front().t)
  {
    const double e = std::min(t1, s.front().t);
    out.push_back({e - t0, s.front().acc, s.front().gyr, e - t0 > gap_thresh});
  }
  for (size_t k = 0; k + 1 < s.size(); ++k)
  {
    const double a = std::max(t0, s[k].t), b = std::min(t1, s[k + 1].t);
    if (b <= a) continue;
    out.push_back({b - a, 0.5 * (s[k].acc + s[k + 1].acc), 0.5 * (s[k].gyr + s[k + 1].gyr),
                   s[k + 1].t - s[k].t > gap_thresh});
  }
  if (t1 > s.back().t)
  {
    const double a = std::max(t0, s.back().t);
    out.push_back({t1 - a, s.back().acc, s.back().gyr, t1 - a > gap_thresh});
  }
  return out;
}

void propagate(NavState &x, const std::vector<Segment> &segs, const NoiseParams &n,
               bool est_bias, bool est_gravity)
{
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();  // no namespace-scope Eigen objects:
                                                            // callers may run during static init
  for (const Segment &sg : segs)
  {
    const double h = sg.dt;
    if (h <= 0.0) continue;
    const Eigen::Vector3d w = sg.gyr - x.bg, a = sg.acc - x.ba;
    const Eigen::Matrix3d R0 = x.R, dR = expSO3(w * h);

    Mat19 F = Mat19::Identity();
    F.block<3, 3>(0, 0) = dR.transpose();
    if (est_bias) F.block<3, 3>(0, 10) = -rightJacobian(w * h) * h;
    F.block<3, 3>(3, 7) = I3 * h;
    F.block<3, 3>(7, 0) = -R0 * skew(a) * h;
    if (est_bias) F.block<3, 3>(7, 13) = -R0 * h;
    if (est_gravity) F.block<3, 3>(7, 16) = I3 * h;

    const double k = sg.gap ? n.gap_factor : 1.0;
    Mat19 Q = Mat19::Zero();
    if (n.discrete)
    {   // FAST-LIVO2-style: per-interval variance cov * h^2 (rate dependent), no gravity walk
      Q.block<3, 3>(0, 0) = I3 * (n.gyr_cov * h * h * k);
      Q.block<3, 3>(7, 7) = I3 * (n.acc_cov * h * h * k);
      if (est_bias)
      {
        Q.block<3, 3>(10, 10) = I3 * (n.bgyr_cov * h * h);
        Q.block<3, 3>(13, 13) = I3 * (n.bacc_cov * h * h);
      }
    }
    else
    {
      Q.block<3, 3>(0, 0) = I3 * (n.gyr_nd * n.gyr_nd * h * k);
      Q.block<3, 3>(7, 7) = I3 * (n.acc_nd * n.acc_nd * h * k);
      if (est_bias)
      {
        Q.block<3, 3>(10, 10) = I3 * (n.gyr_rw * n.gyr_rw * h);
        Q.block<3, 3>(13, 13) = I3 * (n.acc_rw * n.acc_rw * h);
      }
      if (est_gravity) Q.block<3, 3>(16, 16) = I3 * (n.grav_rw * n.grav_rw * h);
    }
    x.P = F * x.P * F.transpose() + Q;

    // The interval's mean specific force is rotated by the mid-interval
    // attitude (FAST-LIVO2 uses the end attitude, a first-order error).
    const Eigen::Vector3d acc_w = R0 * expSO3(w * (0.5 * h)) * a + x.g;
    x.R = R0 * dR;
    x.p += x.v * h + 0.5 * acc_w * h * h;
    x.v += acc_w * h;
  }
  const Eigen::Quaterniond q(x.R);
  x.R = q.normalized().toRotationMatrix();
}

void propagateWarmup(NavState &x, const std::vector<Segment> &segs, double gyr_nd,
                     double vel_rw)
{
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();  // no namespace-scope Eigen objects:
                                                            // callers may run during static init
  for (const Segment &sg : segs)
  {
    const double h = sg.dt;
    if (h <= 0.0) continue;
    const Eigen::Matrix3d dR = expSO3((sg.gyr - x.bg) * h);
    Mat19 F = Mat19::Identity();
    F.block<3, 3>(0, 0) = dR.transpose();
    F.block<3, 3>(3, 7) = I3 * h;
    Mat19 Q = Mat19::Zero();
    Q.block<3, 3>(0, 0) = I3 * (gyr_nd * gyr_nd * h);
    Q.block<3, 3>(7, 7) = I3 * (vel_rw * vel_rw * h);
    x.P = F * x.P * F.transpose() + Q;
    x.R = x.R * dR;
    x.p += x.v * h;
  }
  const Eigen::Quaterniond q(x.R);
  x.R = q.normalized().toRotationMatrix();
}

Preintegration::Preintegration(const std::vector<Segment> &segs, const Eigen::Vector3d &bg,
                               const Eigen::Vector3d &ba)
{
  for (const Segment &sg : segs)
  {
    const double h = sg.dt;
    if (h <= 0.0) continue;
    const Eigen::Vector3d w = sg.gyr - bg, a = sg.acc - ba;
    // Mid-interval rotation (as in propagate) and its gyro-bias Jacobian.
    const Eigen::Matrix3d dRk = expSO3(w * h), dRh = expSO3(w * (0.5 * h)), A = skew(a);
    const Eigen::Matrix3d dRmid = dR * dRh;
    const Eigen::Matrix3d JRmid = dRh.transpose() * JR_bg - rightJacobian(w * (0.5 * h)) * (0.5 * h);
    dP += dV * h + 0.5 * dRmid * a * h * h;
    dV += dRmid * a * h;
    JP_ba += JV_ba * h - 0.5 * dRmid * h * h;
    JP_bg += JV_bg * h - 0.5 * dRmid * A * JRmid * h * h;
    JV_ba += -dRmid * h;
    JV_bg += -dRmid * A * JRmid * h;
    JR_bg = dRk.transpose() * JR_bg - rightJacobian(w * h) * h;
    dR = dR * dRk;
    dt += h;
  }
}

StaticResult initStatic(const std::vector<ImuSample> &win, double gravity_norm,
                        double max_gyr_norm, double max_acc_std)
{
  StaticResult r;
  if (win.size() < 10) return r;
  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero(), gyr_mean = Eigen::Vector3d::Zero();
  for (const ImuSample &s : win)
  {
    acc_mean += s.acc;
    gyr_mean += s.gyr;
  }
  acc_mean /= win.size();
  gyr_mean /= win.size();
  Eigen::Vector3d acc_var = Eigen::Vector3d::Zero();
  for (const ImuSample &s : win) acc_var += (s.acc - acc_mean).cwiseAbs2();
  acc_var /= win.size();
  r.gyr_mean_norm = gyr_mean.norm();
  r.acc_std_max = acc_var.cwiseSqrt().maxCoeff();
  r.ok = r.gyr_mean_norm < max_gyr_norm && r.acc_std_max < max_acc_std;
  r.g_imu = -acc_mean.normalized() * gravity_norm;
  r.bg = gyr_mean;
  return r;
}

DynamicResult initDynamic(const std::vector<Frame> &frames,
                          const std::vector<ImuSample> &samples, double gravity_norm,
                          double min_excitation, double max_g_norm_error)
{
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();  // no namespace-scope Eigen objects:
                                                            // callers may run during static init
  DynamicResult r;
  const int N = static_cast<int>(frames.size());
  if (N < 4)
  {
    r.why = "too few frames";
    return r;
  }
  std::vector<std::vector<Segment>> segs(N - 1);
  std::vector<Preintegration> pre(N - 1);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  for (int i = 0; i + 1 < N; ++i)
  {
    segs[i] = segments(samples, frames[i].t, frames[i + 1].t, 1e9);
    pre[i] = Preintegration(segs[i], zero, zero);
    if (pre[i].dt <= 0.0)
    {
      r.why = "empty interval";
      return r;
    }
  }

  // Excitation: spread of the mean specific force between frames (VINS-Mono).
  Eigen::Vector3d mean_a = Eigen::Vector3d::Zero();
  for (const Preintegration &p : pre) mean_a += p.dV / p.dt;
  mean_a /= pre.size();
  double var = 0.0;
  for (const Preintegration &p : pre) var += (p.dV / p.dt - mean_a).squaredNorm();
  r.excitation = std::sqrt(var / pre.size());
  if (r.excitation < min_excitation)
  {
    r.why = "insufficient excitation";
    return r;
  }

  // Gyro bias from relative rotations: Log(dR^T R_i^T R_j) ~= JR_bg * bg.
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (int i = 0; i + 1 < N; ++i)
  {
    const Eigen::Vector3d res =
        logSO3(pre[i].dR.transpose() * frames[i].R.transpose() * frames[i + 1].R);
    A += pre[i].JR_bg.transpose() * pre[i].JR_bg;
    b += pre[i].JR_bg.transpose() * res;
  }
  r.bg = A.ldlt().solve(b);
  for (int i = 0; i + 1 < N; ++i) pre[i] = Preintegration(segs[i], r.bg, zero);

  // Linear alignment for v_0..v_{N-1} and g (scale fixed to 1):
  //   -v_i h - 0.5 g h^2   = R_i dP - (p_j - p_i)
  //   -v_i + v_j - g h     = R_i dV
  // then the same with g = g0 + B d and |g0| = gravity_norm (VINS RefineGravity).
  auto solve = [&](const Eigen::Vector3d *g0, Eigen::VectorXd &x) {
    const int ng = g0 ? 2 : 3, nx = 3 * N + ng;
    const Eigen::Matrix<double, 3, 2> B = g0 ? tangentBasis(g0->normalized())
                                             : Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(nx, nx) * 1e-9;
    Eigen::VectorXd z = Eigen::VectorXd::Zero(nx);
    for (int i = 0; i + 1 < N; ++i)
    {
      const double h = pre[i].dt;
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nx);
      Eigen::Matrix<double, 6, 1> y;
      J.block<3, 3>(0, 3 * i) = -I3 * h;
      J.block<3, 3>(3, 3 * i) = -I3;
      J.block<3, 3>(3, 3 * (i + 1)) = I3;
      y.head<3>() = frames[i].R * pre[i].dP - (frames[i + 1].p - frames[i].p);
      y.tail<3>() = frames[i].R * pre[i].dV;
      if (g0)
      {
        J.block<3, 2>(0, 3 * N) = -0.5 * B * h * h;
        J.block<3, 2>(3, 3 * N) = -B * h;
        y.head<3>() += 0.5 * (*g0) * h * h;
        y.tail<3>() += (*g0) * h;
      }
      else
      {
        J.block<3, 3>(0, 3 * N) = -0.5 * I3 * h * h;
        J.block<3, 3>(3, 3 * N) = -I3 * h;
      }
      H += J.transpose() * J;
      z += J.transpose() * y;
    }
    x = H.ldlt().solve(z);
    return B;
  };

  Eigen::VectorXd x;
  solve(nullptr, x);
  Eigen::Vector3d g = x.tail<3>();
  r.g_norm_free = g.norm();
  if (std::abs(r.g_norm_free - gravity_norm) > max_g_norm_error)
  {
    r.why = "gravity norm check failed";
    return r;
  }
  g = g.normalized() * gravity_norm;
  for (int it = 0; it < 4; ++it)
  {
    const Eigen::Matrix<double, 3, 2> B = solve(&g, x);
    g = (g + B * x.tail<2>()).normalized() * gravity_norm;
  }
  solve(&g, x);  // velocities consistent with the final gravity
  r.g_w = g;
  r.v.resize(N);
  for (int i = 0; i < N; ++i) r.v[i] = x.segment<3>(3 * i);
  r.ok = true;
  return r;
}

}  // namespace inertial
