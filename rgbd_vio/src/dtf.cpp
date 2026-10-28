#include "dtf.h"

#include <algorithm>

namespace dtf {
namespace {

// Small ridge so exactly singular information (e.g. a single plane) stays
// invertible; far below any eigenvalue that survives the threshold.
Mat6 regularised(const Mat6 &A)
{
  const double eps = 1e-9 * std::max(A.trace() / 6.0, 1e-12);
  return A + eps * Mat6::Identity();
}

}  // namespace

Eigen::Matrix3d marginal(const Mat6 &A, int block)
{
  const int other = 3 - block;
  const Mat6 R = regularised(A);
  return R.block<3, 3>(block, block) -
         R.block<3, 3>(block, other) * R.block<3, 3>(other, other).ldlt().solve(R.block<3, 3>(other, block));
}

Result reduce(Mat6 &A, Vec6 &b, Mode mode, double tau)
{
  Result r;
  Mat6 U = Mat6::Zero();  // kept directions, one per column
  int k = 0;
  for (const int blk : {0, 3})
  {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(marginal(A, blk));
    const Eigen::Vector3d lam = es.eigenvalues();  // ascending
    const double lmax = std::max(lam(2), 1e-300);
    const bool test = blk == 3 ? mode != Mode::Plain : mode == Mode::Full;
    int kept = 0;
    for (int i = 0; i < 3; ++i)
    {
      const Eigen::Vector3d v = es.eigenvectors().col(i);
      if (test && lam(i) < tau * lmax)
      {
        if (blk == 3) r.weak_trans += v * v.transpose();
        continue;
      }
      U.col(k++).segment<3>(blk) = v;
      ++kept;
    }
    (blk == 0 ? r.kept_rot : r.kept_trans) = kept;
    (blk == 0 ? r.ratio_rot : r.ratio_trans) = std::max(lam(0), 0.0) / lmax;
  }
  if (k == 6) return r;
  if (k == 0)
  {
    A.setZero();
    b.setZero();
    return r;
  }

  const Eigen::LDLT<Mat6> Ainv(regularised(A));
  const Eigen::MatrixXd Uk = U.leftCols(k);
  const Eigen::MatrixXd C = Uk.transpose() * Ainv.solve(Uk);  // kept-component covariance
  const Eigen::MatrixXd Cinv = C.ldlt().solve(Eigen::MatrixXd::Identity(k, k));
  const Vec6 y = Ainv.solve(b);                                // ICP-only pose estimate
  A = Uk * Cinv * Uk.transpose();
  A = 0.5 * (A + A.transpose());
  b = Uk * (Cinv * (Uk.transpose() * y));
  return r;
}

}  // namespace dtf
