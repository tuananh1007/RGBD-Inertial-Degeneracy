// Run at image build: the covariance forms of e14_restriction.hpp reproduce the two-dimensional
// example of the paper (Section 4.3): P- = [1 c; c 1], H = [1 0], R = 1, Pi = diag(1, 0) -> true covariance of the
// projected estimate [1/2 c/2; c/2 1], splice diag(1/2, 1); and the projector keeps the right directions.
#include "e14_restriction.hpp"
#include <algorithm>
#include <cstdio>

int main()
{
  const double c = 0.6;
  Eigen::Matrix2d Pm, PLx, PSx, PL, PS, Pi = Eigen::Vector2d(1, 0).asDiagonal();
  Pm << 1, c, c, 1;
  PLx << 0.5, c / 2, c / 2, 1;
  PSx << 0.5, 0, 0, 1;
  const Eigen::RowVector2d H(1, 0);
  const Eigen::Vector2d K = Pm * H.transpose() / (H * Pm * H.transpose() + 1.0);
  const Eigen::Matrix2d Pp = (Eigen::Matrix2d::Identity() - K * H) * Pm;
  e14::covariances(Pm, Pp, Pi, PL, PS);
  const double e_cov = std::max((PL - PLx).cwiseAbs().maxCoeff(), (PS - PSx).cwiseAbs().maxCoeff());

  Eigen::Matrix3d Pi_t, Q_t, Qx = Eigen::Vector3d(0, 1, 0).asDiagonal();
  double iso;
  const int n = e14::projector(Eigen::Vector3d(1.0, 0.1, 0.5).asDiagonal(), 0.22, Pi_t, Q_t, iso);
  const double e_proj = std::max((Q_t - Qx).cwiseAbs().maxCoeff(), (Pi_t + Q_t - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff());
  const int n0 = e14::projector(Eigen::Vector3d(1.0, 0.1, 0.5).asDiagonal(), 0.0, Pi_t, Q_t, iso);

  printf("[E14 check] 2-D example: max |P_L - true| , |P_S - splice| = %.2e; projector: kept %d (expected 2), "
         "iso %.3f (0.1), err %.2e; tau = 0 keeps %d (3)\n", e_cov, n, iso, e_proj, n0);
  return (e_cov < 1e-12 && n == 2 && std::abs(iso - 0.1) < 1e-12 && e_proj < 1e-12 && n0 == 3) ? 0 : 1;
}
