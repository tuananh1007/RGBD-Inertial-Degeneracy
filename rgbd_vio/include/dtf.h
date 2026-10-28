/*
Degeneracy-aware depth-ICP update ("DTF", lio/dtf_mode).  Pure Eigen so it
can be unit-tested in isolation (test/test_dtf.cpp).

One IEKF iteration of the voxel-map ICP reduces to registration information
A = H^T R^-1 H and gradient b = H^T R^-1 z over the pose error
[rot 0..2 | pos 3..5].  In a degenerate scene (corridor, street, floor with
far walls) the weak directions of A are not just uninformative: the
point-to-plane residuals are biased along them and A overstates their
precision.  reduce() therefore removes them from the measurement:

  * weak directions are chosen per block from the marginal (Schur
    complement) information of translation (and, in Full mode, rotation),
    with eigenvalue / largest eigenvalue < tau;
  * the measurement is replaced by its components along the kept subspace
    U_k, with their exact marginal covariance:
        A' = U_k (U_k^T A^-1 U_k)^-1 U_k^T,   b' = A' A^-1 b.

The Kalman update with (A', b') is the exact update for that reduced
measurement, so the covariance (I - K'H)P needs no correction, and no state
(velocity, biases, gravity) receives a correction driven by the weak
directions.  Kept directions keep their marginal information unchanged.
*/
#pragma once

#include <Eigen/Dense>

namespace dtf {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

enum class Mode { Plain, Trans, Full };

struct Result
{
  int kept_rot = 3, kept_trans = 3;
  double ratio_rot = 1.0, ratio_trans = 1.0;      // smallest / largest marginal eigenvalue
  Eigen::Matrix3d weak_trans = Eigen::Matrix3d::Zero();  // projector onto weak translation dirs
};

// Replace (A, b) by the reduced measurement in place.  Plain leaves them
// unchanged but still reports the ratios.
Result reduce(Mat6 &A, Vec6 &b, Mode mode, double tau);

// Marginal information of one 3x3 block of A (0 = rotation, 3 = position).
Eigen::Matrix3d marginal(const Mat6 &A, int block);

}  // namespace dtf
