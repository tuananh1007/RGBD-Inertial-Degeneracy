/*
RGB-D VIO for UAV Pest Control — UCLA
Based on FAST-LIVO2 (GPLv2). See ../FAST-LIVO2-main/LICENSE.

DepthProcessor converts Intel RealSense D455 depth images into the same
PointCloudXYZI format that the FAST-LIVO2 LIO pipeline expects, effectively
replacing the LiDAR sensor with the structured-light depth camera.

Key differences vs LiDAR:
  - All points share the same timestamp  →  curvature = 0 (no motion distortion)
  - Depth values are in millimetres      →  scaled by depth_scale (1e-3)
  - Point density is much higher         →  use point_filter_num to subsample
  - RGB colour available                 →  stored in normal_x/y/z as float R/G/B

Information-Theoretic Point Selection (ITPS):
  When info_select_en = true, the processor replaces heuristic depth binning
  with a principled E-optimal point selection that maximises the minimum
  eigenvalue of the translational Fisher Information Matrix (FIM).  Each
  candidate point's information contribution depends on (a) its surface
  normal direction, which determines which state DOFs it constrains, and
  (b) its depth-dependent measurement noise.  The algorithm bins candidates
  by normal direction AND depth range, then allocates per-bin budgets to
  equalise information across the six principal constraint directions.
  This unifies depth-stratified sampling, degeneracy detection, and VIO
  covariance adaptation into a single optimisation objective.

  Reference theory:
    - Pukelsheim (1993), Optimal Design of Experiments, Chapter 9 (E-optimality)
    - Zhang et al. (2016), "On degeneracy of optimization-based state estimation
      problems" — motivation for observability-aware point selection
*/

#ifndef DEPTH_PROCESSOR_H_
#define DEPTH_PROCESSOR_H_

#include <common_lib.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <memory>
#include <random>

// ============================================================================
//  SelectionInfo — per-frame information summary returned by ITPS
// ============================================================================
//
//  The Fisher Information Matrix (FIM) for translation is:
//
//    F = Σ_{selected i}  (1/σ_i²) · n_i · n_i^T
//
//  where n_i is the camera-frame surface normal at point i and σ_i is the
//  depth-dependent noise standard deviation.  The eigenvalues λ₁≥λ₂≥λ₃ of F
//  characterise the observability of translation along each principal axis:
//
//    λ₃ ≈ 0  →  degenerate (tunnel/corridor: all normals perpendicular to axis)
//    λ₃ ≈ λ₁ →  isotropic (normals evenly distributed, best case)
//
//  The condition number κ = λ₁/λ₃ and the isotropy ratio ι = λ₃/λ₁ provide
//  scalar summaries of the information geometry.
//
struct SelectionInfo
{
  Eigen::Matrix3d fim;         // Fisher Information Matrix (camera frame, 3×3)
  double          lambda_min;  // smallest eigenvalue of FIM
  double          lambda_max;  // largest eigenvalue of FIM
  Eigen::Vector3d degen_dir;   // eigenvector of λ_min (weakest direction, camera frame)
  double          isotropy;    // λ_min / λ_max ∈ [0, 1]; 1 = perfectly isotropic
  double          condition;   // λ_max / λ_min; ∞ = degenerate
  int             n_candidates; // valid pixels before selection
  int             n_selected;   // points after selection

  SelectionInfo()
      : fim(Eigen::Matrix3d::Zero()),
        lambda_min(0), lambda_max(0),
        degen_dir(Eigen::Vector3d::UnitZ()),
        isotropy(0), condition(1e9),
        n_candidates(0), n_selected(0) {}
};

// ============================================================================
//  DepthProcessor
// ============================================================================

class DepthProcessor
{
public:
  DepthProcessor();
  ~DepthProcessor() {}

  // ---- Intrinsics ----------------------------------------------------------
  void setCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &info);
  void setIntrinsics(double fx, double fy, double cx, double cy,
                     int width, int height);

  // ---- Main conversion -----------------------------------------------------
  // depth_img : 16-bit CV_16UC1 image, values in millimetres.
  // rgb_img   : optional 8-bit BGR image (same resolution after alignment).
  // timestamp : frame stamp in seconds.
  // pcl_out   : output cloud in body/depth-camera frame.
  // info      : if non-null, receives the per-frame FIM and degeneracy info.
  //
  // When info_select_en = true, uses Information-Theoretic Point Selection.
  // When info_select_en = false, falls back to the heuristic path
  // (depth binning or uniform stride).
  void process(const cv::Mat &depth_img,
               const cv::Mat &rgb_img,
               double timestamp,
               PointCloudXYZI::Ptr &pcl_out,
               SelectionInfo *info = nullptr);

  // ---- Accessors for downstream consumers ----------------------------------
  double getFx() const { return fx_; }
  double getFy() const { return fy_; }
  double getCx() const { return cx_; }
  double getCy() const { return cy_; }

  // ---- Configuration parameters --------------------------------------------
  double min_depth;        // metres, default 0.3
  double max_depth;        // metres, default 5.0
  double depth_scale;      // mm → metres, default 1e-3
  int    point_filter_num; // keep 1-in-N pixels (fallback path), default 4
  double blind_sqr;        // squared blind-zone radius (m²), default 0.09

  // ---- Flying-pixel rejection ----------------------------------------------
  bool  flying_pixel_en;      // default true
  float flying_pixel_thresh;  // relative depth-jump limit, default 0.10

  // ---- Depth-stratified sampling (heuristic fallback) ----------------------
  bool depth_bins_en;     // default true
  int  num_depth_bins;    // default 10
  int  max_pts_per_bin;   // default 2500
  int  fallback_total_cap = 0; // stop the depth-bin sampler at this many points (0 = off); matched-budget ablation

  // ---- Information-Theoretic Point Selection (ITPS) ------------------------
  //
  //  The algorithm proceeds in three passes over the depth image:
  //
  //  Pass 1 — Candidate census:
  //    Iterate at fine stride.  For each valid pixel, compute the image-space
  //    surface normal from depth gradients and the depth-dependent noise σ.
  //    Assign the pixel to a (normal-direction, depth-range) bin.
  //    Accumulate the per-bin aggregate FIM.
  //
  //  Pass 2 — Budget allocation (E-optimal rebalancing):
  //    Compute the total FIM from all candidates.  Eigendecompose to find the
  //    weakest constraint direction v₃.  Boost the budget for normal-direction
  //    bins whose normals are aligned with v₃.  This steers point selection
  //    toward surfaces that reduce the worst-case positional uncertainty.
  //
  //  Pass 3 — Selection:
  //    Re-iterate the depth image (or use cached candidates).  Accept each
  //    pixel into its (dir, depth) bin until the bin's adjusted budget is met.
  //    Points are accepted in order of decreasing information weight w = 1/σ²
  //    (i.e. closer points first within each bin, since σ ∝ z).
  //
  //  Complexity: O(H·W) per frame (three passes, no sorting needed thanks to
  //  the monotonic relationship between depth and information weight).
  //
  bool   info_select_en;      // enable ITPS (default: true)
  double noise_frac;          // D455 depth noise as fraction of range, default 0.01
  int    n_dir_bins;          // normal-direction bins (6 = axis-aligned), default 6
  int    target_points;       // total point budget K, default 20000
  double rebalance_alpha;     // degeneracy rebalancing strength α ∈ [0,1], default 0.5
  int    itps_min_candidates; // fewer ITPS candidates (sparse depth) → depth-bin fallback; 0 = off
  bool   itps_uniform_random; // ablation: keep each candidate with probability K / candidates instead of the cell budgets
  std::string itps_fill;      // ablation: how a cell budget is filled: "raster" (first candidates in row order),
                              // "random" (each kept with probability budget / count), "strided" (evenly over the cell)
  double inject_lateral_keep; // ablation (degeneracy injection): keep this fraction of the ITPS candidates whose normal
                              // is lateral (camera +-x), evenly spaced in scan order; 1 = off

  bool camera_info_received;
  int  width, height;

private:
  double fx_, fy_, cx_, cy_;
  bool   intrinsics_set_;
  std::mt19937 rng_;          // itps_uniform_random only (fixed seed)

  // ---- ITPS internals ------------------------------------------------------

  // Classify a camera-frame normal into one of 6 axis-aligned directional bins.
  // Returns bin index ∈ {0..5}: +X, −X, +Y, −Y, +Z, −Z.
  static int normalToDirBin(float nx, float ny, float nz);

  // Estimate surface normal at pixel (u,v) from depth central differences.
  // Returns false if any required neighbour is invalid.
  bool estimateNormal(const cv::Mat &depth_img, int u, int v,
                      float z_centre,
                      float &nx, float &ny, float &nz) const;

  // Check flying-pixel condition for pixel (u,v) with raw depth value.
  bool isFlyingPixel(const cv::Mat &depth_img, int u, int v,
                     uint16_t raw_depth) const;

  // Heuristic fallback (original depth-bin or uniform-stride path).
  void processFallback(const cv::Mat &depth_img,
                       const cv::Mat &rgb_img,
                       PointCloudXYZI::Ptr &pcl_out);

  // ITPS selection.
  void processITPS(const cv::Mat &depth_img,
                   const cv::Mat &rgb_img,
                   PointCloudXYZI::Ptr &pcl_out,
                   SelectionInfo &info);
};

typedef std::shared_ptr<DepthProcessor> DepthProcessorPtr;

#endif // DEPTH_PROCESSOR_H_
