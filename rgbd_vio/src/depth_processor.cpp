/*
RGB-D VIO for UAV Pest Control — UCLA

Information-Theoretic Point Selection (ITPS) for Depth-Based VIO
================================================================

Problem statement
-----------------
Given H×W depth pixels, select K << H·W points that maximise the quality
of the downstream ICP state estimate.  In point-to-plane ICP the i-th
residual is  rᵢ = nᵢᵀ(Rpᵢ + t − cᵢ)  with Jacobian  ∂rᵢ/∂t = nᵢᵀ.
The translational Fisher Information Matrix (FIM) is

    F = Σᵢ (1/σᵢ²) · nᵢ nᵢᵀ                                        (1)

where σᵢ is the depth-dependent noise at point i.  The smallest eigenvalue
λ_min(F) bounds the worst-case positional uncertainty.  Maximising λ_min
is the E-optimality criterion (Pukelsheim 1993), and directly prevents
degenerate configurations where one translational DOF is unobservable.

Algorithm (ITPS)
----------------
Exact E-optimal subset selection is NP-hard.  We use a two-stage
approximation that is O(H·W) and empirically achieves near-isotropic FIM:

  Stage 1: Aggregate information census.
    For each valid candidate pixel, compute surface normal nᵢ from depth
    gradients and information weight wᵢ = 1/σᵢ².  Assign to a
    (direction bin d, depth bin r) cell.  Accumulate per-cell FIM and count.

  Stage 2: E-optimal budget allocation.
    Compute the aggregate FIM from all candidates.  Eigendecompose to find
    the weakest direction v₃.  Allocate more points to direction bins
    whose normals are aligned with v₃, proportionally to the information
    deficit ratio (λ₁/λ₃ − 1).  This is a closed-form approximation to one
    step of the Frank–Wolfe algorithm on the E-optimal design problem.

  Stage 3: Greedy selection.
    Re-scan the depth image.  Accept each valid pixel into its (d,r) cell
    until the cell budget is met.  Within each cell, points are visited in
    raster order; since σ ∝ z, the information weight decreases monotonically
    with depth, and raster order naturally favours high-weight (close) points
    within each depth bin.

Complexity: O(H·W) — three passes over the image, O(1) eigendecomposition
of the 3×3 FIM, O(D·R) budget allocation with D=6 direction and R depth bins.
*/

#include "depth_processor.h"
#include <ros/ros.h>
#include <cmath>

// ============================================================================
//  Constructor
// ============================================================================

DepthProcessor::DepthProcessor()
    : min_depth(0.3),
      max_depth(5.0),
      depth_scale(1e-3),
      point_filter_num(4),
      blind_sqr(0.09),
      flying_pixel_en(true),
      flying_pixel_thresh(0.10f),
      depth_bins_en(true),
      num_depth_bins(10),
      max_pts_per_bin(2500),
      info_select_en(true),
      noise_frac(0.01),
      n_dir_bins(6),
      target_points(20000),
      rebalance_alpha(0.5),
      itps_min_candidates(0),
      itps_uniform_random(false),
      itps_fill("raster"),
      inject_lateral_keep(1.0),
      camera_info_received(false),
      width(0), height(0),
      fx_(0), fy_(0), cx_(0), cy_(0),
      intrinsics_set_(false),
      rng_(1)
{}

// ============================================================================
//  Intrinsics
// ============================================================================

void DepthProcessor::setCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &info)
{
  if (intrinsics_set_) return;
  fx_ = info->K[0];
  fy_ = info->K[4];
  cx_ = info->K[2];
  cy_ = info->K[5];
  width  = info->width;
  height = info->height;
  intrinsics_set_      = true;
  camera_info_received = true;
  ROS_INFO("[DepthProcessor] Intrinsics set: fx=%.2f fy=%.2f cx=%.2f cy=%.2f "
           "size=%dx%d", fx_, fy_, cx_, cy_, width, height);
}

void DepthProcessor::setIntrinsics(double fx, double fy, double cx, double cy,
                                    int w, int h)
{
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
  width = w; height = h;
  intrinsics_set_      = true;
  camera_info_received = true;
}

// ============================================================================
//  Surface normal estimation from depth gradients
// ============================================================================
//
//  Given depth image z(u,v), the 3-D point is
//    p(u,v) = [(u−cx)z/fx,  (v−cy)z/fy,  z].
//
//  The tangent vectors along image axes (central-difference approximation):
//    tᵤ = p(u+1,v) − p(u−1,v),   tᵥ = p(u,v+1) − p(u,v−1).
//
//  The surface normal is n = tᵤ × tᵥ (unnormalised).
//
//  For efficiency we compute the cross product directly from pixel
//  coordinates and depth values, avoiding full 3-D back-projection of
//  the four neighbours:
//
//    tᵤ ≈ [2zc/fx + (u−cx)(zr−zl)/fx,  (v−cy)(zr−zl)/fy,  zr−zl]
//    tᵥ ≈ [(u−cx)(zd−zu)/fx,  2zc/fy + (v−cy)(zd−zu)/fy,  zd−zu]
//
//  where zl,zr,zu,zd are the four depth neighbours and zc is the centre.
//  The leading term dominates for smooth surfaces (small depth gradients),
//  giving the standard KinectFusion approximation.  We keep the full
//  expression for accuracy at depth discontinuities.
//
bool DepthProcessor::estimateNormal(const cv::Mat &depth_img, int u, int v,
                                     float z_centre,
                                     float &nx, float &ny, float &nz) const
{
  // Read 4-connected neighbours (raw uint16, mm)
  uint16_t raw_l = depth_img.at<uint16_t>(v,   u-1);
  uint16_t raw_r = depth_img.at<uint16_t>(v,   u+1);
  uint16_t raw_u = depth_img.at<uint16_t>(v-1, u  );
  uint16_t raw_d = depth_img.at<uint16_t>(v+1, u  );

  if (raw_l == 0 || raw_r == 0 || raw_u == 0 || raw_d == 0)
    return false;

  float zl = raw_l * depth_scale;
  float zr = raw_r * depth_scale;
  float zu = raw_u * depth_scale;
  float zd = raw_d * depth_scale;
  float zc = z_centre;

  // Full tangent vectors in camera frame
  float du_cx = (float)(u) - cx_;
  float dv_cy = (float)(v) - cy_;

  // t_u = p(u+1) - p(u-1)
  float tu_x = 2.0f * zc / fx_ + du_cx * (zr - zl) / fx_;
  float tu_y = dv_cy * (zr - zl) / fy_;
  float tu_z = zr - zl;

  // t_v = p(v+1) - p(v-1)
  float tv_x = du_cx * (zd - zu) / fx_;
  float tv_y = 2.0f * zc / fy_ + dv_cy * (zd - zu) / fy_;
  float tv_z = zd - zu;

  // n = t_u × t_v
  nx = tu_y * tv_z - tu_z * tv_y;
  ny = tu_z * tv_x - tu_x * tv_z;
  nz = tu_x * tv_y - tu_y * tv_x;

  float len = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (len < 1e-8f) return false;

  float inv_len = 1.0f / len;
  nx *= inv_len;
  ny *= inv_len;
  nz *= inv_len;

  // Ensure normal points toward camera (nz < 0 in camera frame where +z = away)
  if (nz > 0) { nx = -nx; ny = -ny; nz = -nz; }

  return true;
}

// ============================================================================
//  Normal → directional bin mapping
// ============================================================================
//
//  We partition the unit sphere into 6 regions by the dominant axis of the
//  normal vector.  Each region corresponds to a family of surfaces:
//
//    Bin 0 (+X): surfaces facing camera-right  (e.g. left wall)
//    Bin 1 (−X): surfaces facing camera-left   (e.g. right wall)
//    Bin 2 (+Y): surfaces facing camera-down   (e.g. ceiling)
//    Bin 3 (−Y): surfaces facing camera-up     (e.g. floor)
//    Bin 4 (+Z): surfaces facing toward camera (e.g. far wall, head-on)
//    Bin 5 (−Z): surfaces facing away          (rare; back-facing)
//
//  This ensures that the 6 bins roughly correspond to the 6 translational
//  constraint directions, making the budget allocation directly interpretable
//  as "how many points constrain each DOF."
//
int DepthProcessor::normalToDirBin(float nx, float ny, float nz)
{
  float ax = std::abs(nx), ay = std::abs(ny), az = std::abs(nz);
  if (ax >= ay && ax >= az) return nx > 0 ? 0 : 1;
  if (ay >= ax && ay >= az) return ny > 0 ? 2 : 3;
  return nz > 0 ? 4 : 5;
}

// ============================================================================
//  Flying-pixel check
// ============================================================================

bool DepthProcessor::isFlyingPixel(const cv::Mat &depth_img, int u, int v,
                                    uint16_t raw_depth) const
{
  if (!flying_pixel_en) return false;
  if (u <= 0 || u >= depth_img.cols - 1 || v <= 0 || v >= depth_img.rows - 1)
    return false;

  const float thresh_raw = flying_pixel_thresh * raw_depth;
  uint16_t zl = depth_img.at<uint16_t>(v,   u-1);
  uint16_t zr = depth_img.at<uint16_t>(v,   u+1);
  uint16_t zu = depth_img.at<uint16_t>(v-1, u  );
  uint16_t zd = depth_img.at<uint16_t>(v+1, u  );
  return (zl != 0 && std::abs((int)zl - (int)raw_depth) > thresh_raw) ||
         (zr != 0 && std::abs((int)zr - (int)raw_depth) > thresh_raw) ||
         (zu != 0 && std::abs((int)zu - (int)raw_depth) > thresh_raw) ||
         (zd != 0 && std::abs((int)zd - (int)raw_depth) > thresh_raw);
}

// ============================================================================
//  ITPS — Information-Theoretic Point Selection
// ============================================================================

void DepthProcessor::processITPS(const cv::Mat &depth_img,
                                  const cv::Mat &rgb_img,
                                  PointCloudXYZI::Ptr &pcl_out,
                                  SelectionInfo &info)
{
  const int H = depth_img.rows;
  const int W = depth_img.cols;
  const int D = n_dir_bins;     // 6
  const int R = num_depth_bins; // 10
  const double depth_bin_width = (max_depth - min_depth) / std::max(1, R);
  const int stride = std::max(1, point_filter_num / 2);

  bool have_rgb = !rgb_img.empty() &&
                  rgb_img.rows == H && rgb_img.cols == W;

  // ------------------------------------------------------------------
  //  Stage 1: Candidate census — accumulate per-cell FIM and counts
  // ------------------------------------------------------------------
  //
  //  We scan the depth image at fine stride.  For each valid pixel we
  //  compute the surface normal and classify it into a (direction, depth)
  //  cell.  We accumulate:
  //    cell_fim[d][r]   — the 3×3 FIM contribution from all candidates in (d,r)
  //    cell_count[d][r] — the number of candidates
  //    total_fim        — the aggregate FIM from all candidates

  // Flatten cell index: cell = d * R + r
  const int n_cells = D * R;
  std::vector<Eigen::Matrix3d> cell_fim(n_cells, Eigen::Matrix3d::Zero());
  std::vector<int>             cell_count(n_cells, 0);
  Eigen::Matrix3d              total_fim = Eigen::Matrix3d::Zero();
  int                          total_candidates = 0;
  // inject_lateral_keep: keep the j-th lateral candidate when floor((j+1) keep) > floor(j keep); both scans visit the
  // candidates in the same order, so they drop the same ones
  auto injected_out = [this](int d_bin, long long &lateral_seen) {
    if (inject_lateral_keep >= 1.0 || d_bin > 1) return false;
    const long long j = lateral_seen++;
    return std::floor((j + 1) * inject_lateral_keep) == std::floor(j * inject_lateral_keep);
  };
  long long lateral_seen = 0;

  for (int v = stride; v < H - 1; v += stride)
  {
    const uint16_t *row = depth_img.ptr<uint16_t>(v);
    for (int u = stride; u < W - 1; u += stride)
    {
      uint16_t raw = row[u];
      if (raw == 0) continue;

      float z = raw * depth_scale;
      if (z < min_depth || z > max_depth) continue;
      if (isFlyingPixel(depth_img, u, v, raw)) continue;

      float x = (u - cx_) * z / fx_;
      float y = (v - cy_) * z / fy_;
      if (x * x + y * y + z * z < blind_sqr) continue;

      // Normal estimation from depth gradients
      float nx, ny, nz;
      if (!estimateNormal(depth_img, u, v, z, nx, ny, nz)) continue;

      // Information weight: w = 1 / σ² where σ = noise_frac · z
      // For the FIM contribution: F_i = w · n · nᵀ
      double sigma = noise_frac * z;
      double w     = 1.0 / (sigma * sigma);

      // Directional and depth bin
      int d_bin = normalToDirBin(nx, ny, nz);
      if (injected_out(d_bin, lateral_seen)) continue;
      int r_bin = static_cast<int>((z - min_depth) / depth_bin_width);
      r_bin = std::max(0, std::min(r_bin, R - 1));

      int cell = d_bin * R + r_bin;
      Eigen::Vector3d n_vec(nx, ny, nz);
      Eigen::Matrix3d F_i = w * n_vec * n_vec.transpose();

      cell_fim[cell]   += F_i;
      cell_count[cell] += 1;
      total_fim        += F_i;
      total_candidates += 1;
    }
  }

  info.n_candidates = total_candidates;

  if (total_candidates == 0)
  {
    info.n_selected = 0;
    return;
  }

  // ------------------------------------------------------------------
  //  Stage 2: E-optimal budget allocation
  // ------------------------------------------------------------------
  //
  //  Eigendecompose the aggregate FIM to find the information geometry.
  //  The weakest direction v₃ (eigenvector of λ_min) is the one that
  //  would benefit most from additional measurement points.
  //
  //  Budget boost for direction bin d:
  //    boost[d] = 1 + α · |v₃ · ê_d| · max(0, κ − 1)
  //
  //  where ê_d is the centre normal of bin d, α is the rebalancing
  //  strength, and κ = λ_max / λ_min is the condition number.
  //  This multiplicatively increases the budget for bins whose normals
  //  align with the degenerate direction.

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(total_fim);
  const Eigen::Vector3d &ev = eig.eigenvalues();   // ascending: ev[0] ≤ ev[1] ≤ ev[2]
  double lam_min = std::max(ev[0], 1e-12);
  double lam_max = std::max(ev[2], 1e-12);
  Eigen::Vector3d v_min = eig.eigenvectors().col(0); // weakest direction

  info.fim        = total_fim;
  info.lambda_min = lam_min;
  info.lambda_max = lam_max;
  info.degen_dir  = v_min;
  info.isotropy   = lam_min / lam_max;
  info.condition  = lam_max / lam_min;

  // Centre normals for the 6 axis-aligned direction bins
  static const Eigen::Vector3d bin_centres[6] = {
    { 1, 0, 0}, {-1, 0, 0},
    { 0, 1, 0}, { 0,-1, 0},
    { 0, 0, 1}, { 0, 0,-1}
  };

  // Compute per-direction boost factor
  double kappa = lam_max / lam_min;
  std::vector<double> dir_boost(D, 1.0);
  for (int d = 0; d < D; d++)
  {
    double alignment = std::abs(v_min.dot(bin_centres[d]));
    dir_boost[d] = 1.0 + rebalance_alpha * alignment * std::max(0.0, kappa - 1.0);
    // Cap the boost to prevent extreme over-allocation
    dir_boost[d] = std::min(dir_boost[d], 5.0);
  }

  // Distribute the target budget across cells.
  // Base allocation: proportional to candidate count (like stratified sampling).
  // Then multiply by the direction-specific boost.
  std::vector<int> cell_budget(n_cells, 0);
  double total_weight = 0;
  std::vector<double> cell_raw_budget(n_cells, 0);

  for (int d = 0; d < D; d++)
  {
    for (int r = 0; r < R; r++)
    {
      int cell = d * R + r;
      // Raw budget ∝ candidate count × direction boost
      cell_raw_budget[cell] = cell_count[cell] * dir_boost[d];
      total_weight += cell_raw_budget[cell];
    }
  }

  // Normalise to target_points total, ensuring at least 1 per non-empty cell
  if (total_weight > 0)
  {
    double scale = (double)target_points / total_weight;
    int allocated = 0;
    for (int cell = 0; cell < n_cells; cell++)
    {
      if (cell_count[cell] > 0)
      {
        cell_budget[cell] = std::max(1, (int)std::round(cell_raw_budget[cell] * scale));
        // Cannot exceed candidate count
        cell_budget[cell] = std::min(cell_budget[cell], cell_count[cell]);
      }
      allocated += cell_budget[cell];
    }
  }

  // ------------------------------------------------------------------
  //  Stage 3: Selection — re-scan and accept points up to cell budgets
  // ------------------------------------------------------------------
  //
  //  We re-iterate the depth image at the same stride.  For each valid
  //  pixel, we check whether its (direction, depth) cell still has
  //  remaining budget.  If so, we emit the point and decrement the
  //  cell counter.
  //
  //  Within each cell, points are visited in raster order.  Since σ ∝ z
  //  and the depth bins are narrow, this naturally produces high-weight
  //  selections (close points come first for a given depth range).

  std::vector<int> cell_accepted(n_cells, 0);
  std::vector<long long> cell_seen(n_cells, 0);   // itps_fill "strided": candidates of the cell visited so far
  const int fill = itps_fill == "random" ? 1 : itps_fill == "strided" ? 2 : 0;
  pcl_out->reserve(target_points);
  const double keep_prob = std::min(1.0, (double)target_points / total_candidates);
  Eigen::Matrix3d selected_fim = Eigen::Matrix3d::Zero();
  lateral_seen = 0;

  for (int v = stride; v < H - 1; v += stride)
  {
    const uint16_t *row = depth_img.ptr<uint16_t>(v);
    for (int u = stride; u < W - 1; u += stride)
    {
      uint16_t raw = row[u];
      if (raw == 0) continue;

      float z = raw * depth_scale;
      if (z < min_depth || z > max_depth) continue;
      if (isFlyingPixel(depth_img, u, v, raw)) continue;

      float x = (u - cx_) * z / fx_;
      float y = (v - cy_) * z / fy_;
      if (x * x + y * y + z * z < blind_sqr) continue;

      float nx, ny, nz;
      if (!estimateNormal(depth_img, u, v, z, nx, ny, nz)) continue;

      int d_bin = normalToDirBin(nx, ny, nz);
      if (injected_out(d_bin, lateral_seen)) continue;
      int r_bin = static_cast<int>((z - min_depth) / depth_bin_width);
      r_bin = std::max(0, std::min(r_bin, R - 1));
      int cell = d_bin * R + r_bin;

      // Check budget (itps_uniform_random: a uniform random subset of the candidates of expected size K)
      if (itps_uniform_random)
      {
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) >= keep_prob) continue;
      }
      else if (fill == 1)
      {
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * cell_count[cell] >= cell_budget[cell]) continue;
      }
      else if (fill == 2)
      {
        // keep the i-th candidate when floor((i+1) k / n) > floor(i k / n): exactly k, evenly spaced in row order
        const long long i = cell_seen[cell]++, k = cell_budget[cell], n = cell_count[cell];
        if ((i + 1) * k / n == i * k / n) continue;
      }
      else
      {
        if (cell_accepted[cell] >= cell_budget[cell]) continue;
        cell_accepted[cell]++;
      }

      // Emit point
      PointType pt;
      pt.x         = x;
      pt.y         = y;
      pt.z         = z;
      pt.intensity = 0.f;
      pt.curvature = 0.f;

      if (have_rgb)
      {
        const cv::Vec3b &bgr = rgb_img.at<cv::Vec3b>(v, u);
        pt.normal_x = static_cast<float>(bgr[2]); // R
        pt.normal_y = static_cast<float>(bgr[1]); // G
        pt.normal_z = static_cast<float>(bgr[0]); // B
      }

      pcl_out->push_back(pt);

      // Update the selected-set FIM for the output
      double sigma = noise_frac * z;
      double w     = 1.0 / (sigma * sigma);
      Eigen::Vector3d n_vec(nx, ny, nz);
      selected_fim += w * n_vec * n_vec.transpose();
    }
  }

  // Recompute FIM eigenvalues for the actually selected set
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_sel(selected_fim);
  const Eigen::Vector3d &ev_sel = eig_sel.eigenvalues();
  info.fim        = selected_fim;
  info.lambda_min = std::max(ev_sel[0], 1e-12);
  info.lambda_max = std::max(ev_sel[2], 1e-12);
  info.degen_dir  = eig_sel.eigenvectors().col(0);
  info.isotropy   = info.lambda_min / info.lambda_max;
  info.condition  = info.lambda_max / info.lambda_min;
  info.n_selected = pcl_out->size();
}

// ============================================================================
//  Heuristic fallback (original depth-binning / uniform-stride path)
// ============================================================================

void DepthProcessor::processFallback(const cv::Mat &depth_img,
                                      const cv::Mat &rgb_img,
                                      PointCloudXYZI::Ptr &pcl_out)
{
  bool have_rgb = !rgb_img.empty() &&
                  rgb_img.rows == depth_img.rows &&
                  rgb_img.cols == depth_img.cols;

  if (depth_bins_en)
  {
    const double bin_width = (max_depth - min_depth) / std::max(1, num_depth_bins);
    std::vector<int> bin_counts(num_depth_bins, 0);
    const int fine_stride = std::max(1, point_filter_num / 2);
    pcl_out->reserve(num_depth_bins * max_pts_per_bin);

    for (int v = fine_stride / 2; v < depth_img.rows - 1; v += fine_stride)
    {
      if (fallback_total_cap > 0 && static_cast<int>(pcl_out->size()) >= fallback_total_cap) break;
      const uint16_t *row = depth_img.ptr<uint16_t>(v);
      for (int u = fine_stride / 2; u < depth_img.cols - 1; u += fine_stride)
      {
        if (fallback_total_cap > 0 && static_cast<int>(pcl_out->size()) >= fallback_total_cap) break;
        uint16_t raw = row[u];
        if (raw == 0) continue;
        double z = raw * depth_scale;
        if (z < min_depth || z > max_depth) continue;
        if (isFlyingPixel(depth_img, u, v, raw)) continue;

        int bin = static_cast<int>((z - min_depth) / bin_width);
        bin = std::max(0, std::min(bin, num_depth_bins - 1));
        if (bin_counts[bin] >= max_pts_per_bin) continue;

        double x = (u - cx_) * z / fx_;
        double y = (v - cy_) * z / fy_;
        if (x * x + y * y + z * z < blind_sqr) continue;

        PointType pt;
        pt.x = x; pt.y = y; pt.z = z;
        pt.intensity = 0.f;
        pt.curvature = 0.f;
        if (have_rgb) {
          const cv::Vec3b &bgr = rgb_img.at<cv::Vec3b>(v, u);
          pt.normal_x = bgr[2]; pt.normal_y = bgr[1]; pt.normal_z = bgr[0];
        }
        pcl_out->push_back(pt);
        bin_counts[bin]++;
      }
    }
  }
  else
  {
    pcl_out->reserve((depth_img.rows / point_filter_num) *
                     (depth_img.cols / point_filter_num));
    for (int v = 0; v < depth_img.rows; v += point_filter_num)
    {
      const uint16_t *row = depth_img.ptr<uint16_t>(v);
      for (int u = 0; u < depth_img.cols; u += point_filter_num)
      {
        uint16_t raw = row[u];
        if (raw == 0) continue;
        double z = raw * depth_scale;
        if (z < min_depth || z > max_depth) continue;
        if (isFlyingPixel(depth_img, u, v, raw)) continue;

        double x = (u - cx_) * z / fx_;
        double y = (v - cy_) * z / fy_;
        if (x * x + y * y + z * z < blind_sqr) continue;

        PointType pt;
        pt.x = x; pt.y = y; pt.z = z;
        pt.intensity = 0.f;
        pt.curvature = 0.f;
        if (!rgb_img.empty() && rgb_img.rows == depth_img.rows) {
          const cv::Vec3b &bgr = rgb_img.at<cv::Vec3b>(v, u);
          pt.normal_x = bgr[2]; pt.normal_y = bgr[1]; pt.normal_z = bgr[0];
        }
        pcl_out->push_back(pt);
      }
    }
  }
}

// ============================================================================
//  Main entry point
// ============================================================================

void DepthProcessor::process(const cv::Mat &depth_img,
                              const cv::Mat &rgb_img,
                              double         timestamp,
                              PointCloudXYZI::Ptr &pcl_out,
                              SelectionInfo *info)
{
  if (!intrinsics_set_)
  {
    ROS_WARN_THROTTLE(2.0, "[DepthProcessor] Camera intrinsics not set yet.");
    return;
  }

  pcl_out->clear();

  bool itps = info_select_en && info;
  if (itps)
  {
    processITPS(depth_img, rgb_img, pcl_out, *info);
    ROS_INFO_THROTTLE(1.0, "[ITPS] candidates=%d selected=%d isotropy=%.4f "
                      "cond=%.1f lmin=%.2f lmax=%.2f",
                      info->n_candidates, info->n_selected,
                      info->isotropy, info->condition,
                      info->lambda_min, info->lambda_max);
    // Sparse depth (e.g. rendered from LiDAR) leaves too few pixels with a
    // valid normal for the directional census; use depth-bin sampling instead.
    if (info->n_candidates < itps_min_candidates)
    {
      ROS_WARN_THROTTLE(5.0, "[ITPS] %d candidates < %d: depth-bin fallback",
                        info->n_candidates, itps_min_candidates);
      pcl_out->clear();
      itps = false;
    }
  }
  if (!itps)
  {
    processFallback(depth_img, rgb_img, pcl_out);
    // Populate a minimal SelectionInfo even in fallback mode
    if (info)
    {
      info->n_candidates = 0;
      info->n_selected   = pcl_out->size();
      info->fim          = Eigen::Matrix3d::Identity();
      info->lambda_min   = 1.0;
      info->lambda_max   = 1.0;
      info->isotropy     = 1.0;
      info->condition    = 1.0;
    }
  }

  pcl_out->header.stamp    = static_cast<uint64_t>(timestamp * 1e6);
  pcl_out->header.frame_id = "depth_camera";

  ROS_DEBUG("[DepthProcessor] Output: %zu points from %dx%d depth image"
            " (ITPS=%s)",
            pcl_out->size(), depth_img.cols, depth_img.rows,
            info_select_en ? "on" : "off");
}
