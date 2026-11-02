// The restricted correction of our earlier system,
// for upstream FAST-LIO2.  Same rule and covariance forms as rgbd_vio/src/RGBDMapper.cpp (applyConstrainedEKF_ITPS,
// projectedCovariances); the call site is in e14_laserMapping.patch.
#pragma once
#include <Eigen/Dense>

namespace e14 {

// Translation projector from the information F = sum n n^T of the matched planes: keeps the eigen-directions with
// lambda_i / lambda_max >= tau (Pi_t), Q_t = I - Pi_t.  Returns the number kept; iso = lambda_min / lambda_max.
inline int projector(const Eigen::Matrix3d &F, double tau, Eigen::Matrix3d &Pi_t, Eigen::Matrix3d &Q_t, double &iso)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(F);
  const Eigen::Vector3d ev = es.eigenvalues();   // ascending
  const double lmax = ev(2);
  iso = lmax > 1e-12 ? ev(0) / lmax : 0.0;
  Pi_t.setIdentity(); Q_t.setZero();
  if (lmax <= 1e-12) return 3;                   // no information (FAST-LIO2 has not updated): nothing to restrict
  Pi_t.setZero();
  int n = 0;
  for (int i = 0; i < 3; ++i)
  {
    const Eigen::Matrix3d vv = es.eigenvectors().col(i) * es.eigenvectors().col(i).transpose();
    if (ev(i) / lmax >= tau) { Pi_t += vv; ++n; }
    else Q_t += vv;
  }
  return n;
}

// Joseph form P_L of the projected gain Pi K and the splice P_S, from the prior P- and the unrestricted posterior P+,
// written without H:  I - K H = P+ (P-)^-1 =: M,  K R K^T = P+ - M P+,  I - Pi K H = (I - Pi) + Pi M.
template <class Mat>
void covariances(const Mat &Pm, const Mat &Pp, const Mat &Pi, Mat &PL, Mat &PS)
{
  const Mat I_Pi = Mat::Identity(Pm.rows(), Pm.cols()) - Pi;
  const Mat M = Pm.ldlt().solve(Pp).transpose();   // P+ (P-)^-1
  const Mat A = I_Pi + Pi * M;
  PL = A * Pm * A.transpose() + Pi * (Pp - M * Pp) * Pi.transpose();
  PL = 0.5 * (PL + PL.transpose());
  PS = Pi * Pp * Pi.transpose() + I_Pi * Pm * I_Pi.transpose();
}

}  // namespace e14
