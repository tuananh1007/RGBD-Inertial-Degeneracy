// Unit tests for dtf::reduce (synthetic point-to-plane information, no ROS).
// Build: catkin target test_dtf; run: rosrun rgbd_vio test_dtf
#include "dtf.h"

#include <cmath>
#include <cstdio>
#include <random>

using dtf::Mat6;
using dtf::Vec6;
using Eigen::Vector3d;

namespace {
int failures = 0;
void check(bool ok, const char *name, double value, double tol)
{
  std::printf("%-52s %s  (value %.3g, tol %.3g)\n", name, ok ? "ok  " : "FAIL", value, tol);
  if (!ok) ++failures;
}

// Point-to-plane information of a scan: points p_i on planes with normals
// n_i, residual Jacobian [ (p x n)^T  n^T ] (rotation first), unit weight.
struct Scan
{
  Mat6 A = Mat6::Zero();
  Vec6 b = Vec6::Zero();
  void add(const Vector3d &p, const Vector3d &n, double w, double z)
  {
    Vec6 h;
    h << p.cross(n), n;
    A += w * h * h.transpose();
    b += w * h * z;
  }
};

// Corridor along x: floor/ceiling (n = z) and walls (n = y) at many x, plus
// a few points on a far end wall (n = x), so x-translation is weak.
Scan corridor(double end_weight, const Vec6 &truth)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Scan s;
  auto add = [&](const Vector3d &p, const Vector3d &n, double w) {
    Vec6 h;
    h << p.cross(n), n;
    s.add(p, n, w, h.dot(truth));  // noise-free residual of the true motion
  };
  for (int i = 0; i < 400; ++i)
  {
    const double x = 1.0 + 9.0 * (0.5 * u(rng) + 0.5);
    add(Vector3d(x, u(rng), -1.2), Vector3d(0, 0, 1), 1.0);
    add(Vector3d(x, u(rng), 1.3), Vector3d(0, 0, -1), 1.0);
    add(Vector3d(x, -1.5, u(rng)), Vector3d(0, 1, 0), 1.0);
    add(Vector3d(x, 1.4, u(rng)), Vector3d(0, -1, 0), 1.0);
  }
  for (int i = 0; i < 20; ++i) add(Vector3d(12.0, u(rng), u(rng)), Vector3d(-1, 0, 0), end_weight);
  return s;
}
}  // namespace

int main()
{
  Vec6 truth;
  truth << 0.01, -0.02, 0.015, 0.3, -0.05, 0.02;

  // Well-conditioned scan: nothing is removed, (A, b) are untouched.
  {
    Scan s = corridor(40.0, truth);
    Mat6 A = s.A;
    Vec6 b = s.b;
    const dtf::Result r = dtf::reduce(A, b, dtf::Mode::Full, 0.02);
    check(r.kept_rot == 3 && r.kept_trans == 3, "well-conditioned: all directions kept", r.ratio_trans, 0.02);
    check((A - s.A).norm() < 1e-12 * s.A.norm() && (b - s.b).norm() < 1e-12 * s.b.norm(),
          "well-conditioned: (A, b) unchanged", (A - s.A).norm(), 0);
  }

  // Corridor: x-translation is weak and is removed.
  {
    Scan s = corridor(0.01, truth);
    Mat6 A = s.A;
    Vec6 b = s.b;
    const Eigen::Matrix3d St = dtf::marginal(s.A, 3);
    const dtf::Result r = dtf::reduce(A, b, dtf::Mode::Trans, 0.02);
    const Eigen::Vector3d w = (r.weak_trans * Vector3d::UnitX()).normalized();
    check(r.kept_trans == 2 && r.kept_rot == 3, "corridor: one translation direction removed", r.ratio_trans, 0.02);
    check(std::abs(w.x()) > 0.999, "corridor: removed direction is the corridor axis", std::abs(w.x()), 0.999);

    const Eigen::Matrix3d St2 = dtf::marginal(A, 3);
    check(w.dot(St2 * w) < 1e-6 * St.norm(), "reduced: no marginal information along the weak axis", w.dot(St2 * w), 1e-6 * St.norm());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(St);
    double dk = 0.0;
    for (int i = 1; i < 3; ++i)  // the two kept eigen-directions
    {
      const Vector3d v = es.eigenvectors().col(i);
      dk = std::max(dk, std::abs(v.dot(St2 * v) - es.eigenvalues()(i)) / es.eigenvalues()(i));
    }
    check(dk < 1e-6, "reduced: kept translation information unchanged", dk, 1e-6);

    // The reduced measurement still recovers the kept components of the
    // true motion exactly (noise-free residuals) and nothing along the axis.
    const Vec6 sol = A.completeOrthogonalDecomposition().solve(b);
    const Eigen::Matrix3d keep = Eigen::Matrix3d::Identity() - r.weak_trans;
    const double e = std::max((sol.head<3>() - truth.head<3>()).norm(),
                              (keep * (sol.tail<3>() - truth.tail<3>())).norm());
    check(e < 1e-6, "reduced: kept components of the motion recovered", e, 1e-6);
    check(std::abs(w.dot(sol.tail<3>())) < 1e-6, "reduced: no correction along the weak axis", std::abs(w.dot(sol.tail<3>())), 1e-6);
  }

  // Plain mode reports ratios but never changes (A, b).
  {
    Scan s = corridor(0.01, truth);
    Mat6 A = s.A;
    Vec6 b = s.b;
    const dtf::Result r = dtf::reduce(A, b, dtf::Mode::Plain, 0.02);
    check(r.kept_trans == 3 && A == s.A && b == s.b && r.ratio_trans < 0.02, "plain: unchanged, ratio still reported",
          r.ratio_trans, 0.02);
  }

  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
