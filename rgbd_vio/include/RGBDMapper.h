/*
RGB-D VIO for UAV Pest Control — UCLA
Based on FAST-LIVO2 (GPLv2). See ../FAST-LIVO2-main/LICENSE.

RGBDMapper replaces LIVMapper:
  • Subscribes to D455 depth image (instead of LiDAR PointCloud2 / Livox)
  • Subscribes to D455 depth CameraInfo (for intrinsics)
  • Subscribes to D455 colour image  (same role as before)
  • Subscribes to D455 IMU           (same role as before)

The depth image is converted to PointCloudXYZI by DepthProcessor and then
fed into the unmodified FAST-LIVO2 LIO pipeline (voxel map + EKF).  The
colour image is fed into the unmodified VIO pipeline (patch tracking + EKF).

Two runtime modes (set via config):
  RGBD_LIVO (depth_en=1, img_en=1) — depth LIO + colour VIO + IMU  [default]
  ONLY_VIO  (depth_en=0, img_en=1) — colour VIO + IMU only
                                      (depth NOT used; no LIO step)
*/

#ifndef RGBD_MAPPER_H
#define RGBD_MAPPER_H

#include "depth_processor.h"
#include "IMU_Processing.h"
#include "vio.h"
#include "inertial_init.h"
#include "dtf.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <vikit/camera_loader.h>
#include <message_filters/subscriber.h>

class RGBDMapper
{
public:
  explicit RGBDMapper(ros::NodeHandle &nh);
  ~RGBDMapper();

  // ---- Lifecycle ---------------------------------------------------------
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh,
                                          image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();

  // ---- Processing --------------------------------------------------------
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();        // depth-as-LiDAR: ICP + voxel map update + EKF
  void handleDepthInit();  // depth-aided VIO: build _pv_list with normals, NO ICP
  void savePCD();
  void processImu();

  // ---- Staged IMU mode (imu/init_mode: staged) -----------------------------
  void stagedProcessImu();   // replaces ImuProcess::Process2 for depth frames
  void stagedAfterLIO();     // records the corrected pose, tries initialisation
  void stagedFinishInit(const V3D &g_w, const V3D &v, const V3D &bg, double sigma_v,
                        double sigma_g, const char *how);

  // ---- Degeneracy-aware ICP update (lio/dtf_mode, see dtf.h) ---------------
  void dtfStateEstimation(StatesGroup &state_propagat);  // StateEstimation + dtf::reduce

  // ---- Synchronisation ---------------------------------------------------
  // Returns true when a complete (depth cloud OR image) + IMU bundle is ready.
  bool sync_packages(LidarMeasureGroup &meas);

  // ---- IMU propagation timer (high-rate odometry for flight controller) --
  void prop_imu_once(StatesGroup &state, double dt, V3D acc, V3D angvel);
  void imu_prop_callback(const ros::TimerEvent &e);

  // ---- Sensor callbacks --------------------------------------------------
  void depth_cbk(const sensor_msgs::ImageConstPtr &depth_msg);
  void depth_info_cbk(const sensor_msgs::CameraInfo::ConstPtr &info_msg);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);

  // ---- Point-cloud helpers -----------------------------------------------
  void transformDepthCloud(const Eigen::Matrix3d &rot,
                            const Eigen::Vector3d &t,
                            const PointCloudXYZI::Ptr &in,
                            PointCloudXYZI::Ptr &out);
  void pointBodyToWorld(const PointType &pi, PointType &po);
  void RGBpointBodyToWorld(PointType const *pi, PointType *po);
  void RGBpointBodyDepthToIMU(PointType const *pi, PointType *po);

  // ---- Publishing helpers ------------------------------------------------
  void publish_frame_world(const ros::Publisher &pub, VIOManagerPtr vio);
  void publish_img_rgb(const image_transport::Publisher &pub, VIOManagerPtr vio);
  void publish_odometry(const ros::Publisher &pub);
  void publish_mavros(const ros::Publisher &pub);
  void publish_path(const ros::Publisher &pub);

  template <typename T> void set_posestamp(T &out);

  // ---- Parameter loading -------------------------------------------------
  void readParameters(ros::NodeHandle &nh);

  // ---- Observability-Constrained EKF (OC-EKF) --------------------------------
  //
  // Replaces heuristic Tikhonov suppression with a principled constrained-EKF
  // formulation.  The key idea (Zhang et al., 2016; Hinduja et al., 2019):
  //
  //   The ICP measurement Jacobian H ∈ R^{N×6} has column rank ≤ 6.  When the
  //   scene geometry is degenerate (e.g. tunnel, flat wall), some translational
  //   or rotational DOFs become unobservable: the FIM F = H^T R^{-1} H has
  //   near-zero eigenvalues along those directions.
  //
  //   Rather than using the full Kalman gain K = P H^T (H P H^T + R)^{-1},
  //   which allows noise to corrupt unobservable states, we decompose F via
  //   eigendecomposition:
  //
  //     F = V Λ V^T,   Λ = diag(λ₁, ..., λ₆)
  //
  //   and construct a projection matrix Π that retains only the well-observed
  //   subspace (eigenvalues above a threshold):
  //
  //     Π = Σ_{i: λᵢ > λ_thresh} vᵢ vᵢ^T
  //
  //   The constrained EKF update becomes:
  //
  //     δx_constrained = Π_full · δx_full
  //
  //   where Π_full ∈ R^{DIM_STATE × DIM_STATE} embeds Π into the full state
  //   space (identity for IMU bias/gravity states that are propagated, not
  //   observed by ICP).
  //
  //   The state covariance is updated as:
  //     P_constrained = Π_full · P_ekf · Π_full^T + (I - Π_full) · P_prop · (I - Π_full)^T
  //
  //   This preserves the propagated (IMU) uncertainty along unobservable
  //   directions while incorporating the ICP correction along observable ones.
  //
  // Two entry points:
  //   applyConstrainedEKF_ITPS()  — uses the depth-front-end FIM (preferred)
  //   applyConstrainedEKF_ICP()   — SVD on ICP Hessian (fallback)
  //
  // References:
  //   [1] Zhang et al., "On degeneracy of optimization-based state estimation
  //       problems," ICRA 2016.
  //   [2] Hinduja et al., "Degeneracy-aware factors with applications to
  //       underwater SLAM," IROS 2019.
  //   [3] Tagliabue et al., "LION: Lidar-Inertial Observability-Aware
  //       Navigator," RAL 2021.
  //
  void applyConstrainedEKF_ITPS(const SelectionInfo &info,
                                const StatesGroup &state_prop);
  void applyConstrainedEKF_ICP(const std::vector<pointWithVar> &pv_list,
                               const StatesGroup &state_prop);

  // Legacy wrappers (kept for config compatibility)
  void applyITPSDegeneracy(const SelectionInfo &info);
  void computeDegeneracy(const std::vector<pointWithVar> &pv_list);

  // ---- Pose linearity check -------------------------------------------------
  // PCA on sliding window of world-frame positions.  Linearity score ℓ ∈ [0,1]:
  //   ℓ → 1  near-linear motion → inflate VIO covariance.
  //   ℓ → 0  rich 3-D trajectory.
  void updateLinearityScore();

  // ---- Sprint 3: Long-term visual map (#11) --------------------------------
  // updateLVMap()              — called every lv_update_interval_ frames from
  //                              handleLIO(); samples _pv_list at 1/10 density
  //                              into a coarse spatial hash (cell size lv_voxel_size_).
  //                              Evicts the oldest half when the map exceeds
  //                              lv_map_max_pts_ entries.
  // mergeLVMapToCurrentFrame() — called in stateEstimationAndMapping() before
  //                              VIO; appends LV map points that are within the
  //                              depth-camera frustum to _pv_list so that VIO
  //                              can initialise long-term visual features.
  void updateLVMap();
  void mergeLVMapToCurrentFrame();

  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &msg);

  // ========================================================================
  // State
  // ========================================================================
  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;

  string root_dir;
  string depth_topic, depth_info_topic, imu_topic, img_topic, seq_name;

  // Extrinsics: depth-camera frame → IMU frame
  V3D extT;   // translation  (metres)
  M3D extR;   // rotation matrix

  int feats_down_size = 0, max_iterations = 0;

  // Timing
  double last_timestamp_depth = -1.0;
  double last_timestamp_imu   = -1.0;
  double last_timestamp_img   = -1.0;
  double _first_depth_time    = 0.0;
  double imu_time_offset      = 0.0;
  double img_time_offset      = 0.0;
  double depth_time_offset    = 0.0;
  double exposure_time_init   = 0.0;

  // Filter sizes
  double filter_size_surf_min = 0.1;
  double filter_size_pcd      = 0.15;

  // Covariances
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;

  // Flags
  bool lidar_map_inited   = false;
  bool pcd_save_en        = false;
  bool img_save_en        = false;
  bool pub_effect_point_en= false;
  bool pose_output_en     = false;
  bool cov_output_en      = false; // with pose_output_en: pose covariance per pose (<seq>.cov): position block (world
                                   // frame, 6 upper-triangle values), then rotation block (body-frame error, 6), then
                                   // rotation-position cross block (9, row-major)
  bool depth_en           = true;  // collect depth images at all
  bool lio_en             = true;  // true  → depth-as-LiDAR (ICP + voxel map)
                                   // false → depth-aided VIO (normals only, no ICP)
  bool imu_en             = true;
  bool gravity_est_en     = true;
  bool ba_bg_est_en       = true;
  bool normal_en          = true;
  bool exposure_estimate_en = false;
  bool inverse_composition_en = false;
  bool raycast_en         = false;
  bool gravity_align_en   = false;
  bool gravity_align_finished = false;
  bool dense_map_en       = false;
  bool colmap_output_en   = false;
  bool is_first_frame     = false;

  int  img_en             = 1;
  int  imu_int_frame      = 3;
  int  img_save_interval  = 1;
  int  pcd_save_interval  = -1;
  int  pcd_save_type      = 0;
  int  pub_scan_num       = 1;
  int  grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;

  double outlier_threshold;
  double IMG_POINT_COV;
  double vio_gate_chi2;   // photometric consistency gate (vio/gate_chi2, 0 = off)
  bool vio_lba_en;        // local photometric bundle adjustment (vio/lba_*, off by default)
  int vio_lba_window, vio_lba_period, vio_lba_iters, vio_lba_max_points, vio_lba_max_obs;
  double vio_lba_sigma_rot, vio_lba_sigma_pos, vio_lba_huber, vio_lba_max_move;
  // Depth noise model (used by handleDepthInit)
  double depth_noise_frac  = 0.01;  // axial noise = noise_frac × range (e.g. 1 %)
  double lateral_noise_m   = 0.005; // absolute lateral noise (metres, e.g. 5 mm)
  double blind_rgb_points = 0.0;
  double plot_time;
  int    frame_cnt;

  // ---- Buffers -----------------------------------------------------------
  deque<PointCloudXYZI::Ptr> depth_cloud_buffer;
  deque<double>               depth_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat>              img_buffer;
  deque<double>               img_time_buffer;

  // ---- Point-cloud work buffers ------------------------------------------
  vector<pointWithVar>        _pv_list;
  vector<double>              extrinT, extrinR;
  vector<double>              cameraextrinT, cameraextrinR;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr  pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup DepthMeasures; // re-uses LidarMeasureGroup struct
  StatesGroup _state;
  StatesGroup state_propagat;

  nav_msgs::Path        path;
  nav_msgs::Odometry    odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  // ---- Sub-system managers -----------------------------------------------
  ImuProcessPtr      p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr      vio_manager;
  DepthProcessorPtr  depth_proc;

  // ---- ROS publishers / subscribers / timers -----------------------------
  ros::Subscriber sub_depth, sub_depth_info, sub_imu, sub_img;
  ros::Publisher  pubLaserCloudFullRes, pubNormal, pubSubVisualMap;
  ros::Publisher  pubLaserCloudEffect, pubLaserCloudMap;
  ros::Publisher  pubOdomAftMapped, pubPath;
  ros::Publisher  mavros_pose_publisher;
  ros::Publisher  pubImuPropOdom;
  image_transport::Publisher pubImage;
  ros::Timer      imu_prop_timer;

  // ---- IMU propagation (high-rate output for flight controller) ----------
  bool   new_imu = false, state_update_flg = false;
  bool   imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  StatesGroup imu_propagate, latest_ekf_state;
  nav_msgs::Odometry imu_prop_odom;

  // ---- Statistics --------------------------------------------------------
  int    frame_num   = 0;
  double aver_time_consu    = 0;
  double aver_time_icp      = 0;
  double aver_time_map_inre = 0;

  // ========================================================================
  // Sprint 1 — Adaptive VIO covariance scaling (#6)
  // IMG_POINT_COV is the baseline; vio_cov_scale_ multiplies it at runtime.
  // Scale increases when the pose trajectory is near-linear (low parallax →
  // poor depth observability for VIO) and resets toward 1.0 otherwise.
  // ========================================================================
  double vio_cov_scale_ = 1.0;

  // ========================================================================
  // Sprint 1 — Pose linearity check (#9)
  // ========================================================================
  std::deque<V3D> pose_history_;
  int             linearity_window_ = 20;   // number of poses in the window
  double          linearity_score_  = 0.0;  // 0 = 3-D rich, 1 = straight line

  // ========================================================================
  // Observability-Constrained EKF (OC-EKF)
  //
  // The FIM eigendecomposition identifies observable/unobservable DOFs.
  // The constrained EKF projects out the ICP update along unobservable
  // directions and preserves the IMU-propagated uncertainty there.
  //
  // oc_ekf_en_:         master switch for OC-EKF (default true)
  // oc_lambda_thresh_:  eigenvalue ratio threshold for observability
  //                     (λ_i / λ_max < thresh → unobservable)
  // oc_cov_inflate_:    inflation factor for covariance along unobs. dirs
  // ========================================================================
  SelectionInfo last_selection_info_;                      // per-frame ITPS output
  bool    oc_ekf_en_                  = true;  // enable OC-EKF (default: true)
  double  oc_lambda_thresh_           = 0.05;  // base eigenvalue ratio for observability
  double  oc_lambda_lo_               = 0.01;  // adaptive thresh lower bound (well-conditioned)
  double  oc_lambda_hi_               = 0.10;  // adaptive thresh upper bound (ill-conditioned)
  double  oc_adapt_kappa_knee_        = 20.0;  // condition number at sigmoid midpoint
  bool    oc_adaptive_thresh_en_      = true;  // enable adaptive threshold (false = fixed)
  double  oc_cov_inflate_             = 10.0;  // inflate cov along unobs. dirs by this factor
  std::string oc_cov_mode_            = "splice";  // "splice" (earlier system) | "joseph" (consistent covariance of the projected gain)
  bool    oc_sched_from_ratio_        = false; // keep the VIO scheduling signal from the plain depth update (rho_t) instead of the selected-normal isotropy
  bool    oc_icp_cov_from_config_     = false; // sparse-depth (ICP) projection path: apply lio/oc_cov_mode and oc_cov_inflate (false: splice, no inflation, as our earlier system)
  bool    init_cov_reset_             = true;  // staged initialization restarts the inertial covariance blocks (false: keep the covariance; E12 diagnostic)
  bool    oc_after_plain_             = false; // with dtf_mode plain: apply the projection after the plain update (which sets rho_t, as in Ours)
  bool    legacy_init_                = false; // staged mode with FAST-LIVO2-style initialization (first-samples gravity, bias = mean gyro, accel rescale)
  bool    oc_persist_guard_en_        = false; // temporal persistence guard
  int     oc_persist_frames_          = 30;    // disable OC-EKF after this many consecutive degenerate frames
  double  deg_threshold_              = 0.07;  // ICP-normal σ_min threshold (fallback)
  double  deg_isotropy_thresh_        = 0.10;  // ITPS isotropy threshold for degeneracy
  double  last_sigma_min_             = 1.0;
  bool    is_degenerate_              = false;
  int     consecutive_degenerate_frames_ = 0;
  V3D     degen_dir_                  = V3D(0, 0, 1); // most degenerate direction (world)
  int     oc_observable_dims_         = 6;     // number of observable DOFs (diagnostic)

  // lio/dtf_mode: "upstream" = VoxelMapManager::StateEstimation followed by
  // the OC-EKF / Tikhonov post-processing above; "plain", "trans", "full" =
  // dtfStateEstimation with dtf::Mode Plain / Trans / Full and no post-processing.
  bool      dtf_ours_                 = false;
  dtf::Mode dtf_mode_                 = dtf::Mode::Plain;
  double    dtf_thresh_               = 0.05;  // marginal eigenvalue ratio below which a direction is removed
  double    dtf_inflate_              = 1.0;   // >1: scale position covariance along removed directions
  double    dept_err_quad_            = 0.0;   // dtf modes: range std = dept_err + k z^2 (stereo depth)

  // ========================================================================
  // Sprint 2 — Degeneration-aware adaptive VIO frame selector (#10)
  // When LIO is well-constrained (σ_min > skip_vio_sigma_thresh_) every
  // other VIO frame is skipped, cutting CPU/power consumption ~30 %.
  // When LIO is degenerate all VIO frames are used for compensation.
  // ========================================================================
  double skip_vio_sigma_thresh_ = 0.3; // above this → can skip every 2nd VIO
  double meas_var_scale_ = 1.0;        // multiplies the variance of every depth residual (E10 diagnostic)
  int    vio_skip_counter_      = 0;

  // ========================================================================
  // Sprint 3 — Long-term visual map (LV Map) (#11)
  // Sparse spatial hash: cell key → one representative pointWithVar.
  // Populated every lv_update_interval_ LIO frames at 1/10 density from
  // _pv_list.  Appended to _pv_list before each VIO step for points that
  // fall inside the current depth-camera frustum.
  // ========================================================================
  std::unordered_map<std::size_t, pointWithVar> lv_map_;
  int    lv_map_max_pts_    = 50000; // evict oldest half when exceeded
  double lv_voxel_size_     = 2.0;   // LV map spatial cell size (metres)
  int    lv_update_interval_= 30;    // update LV map every N LIO frames
  int    lv_frame_counter_  = 0;     // counts LIO frames

  // ========================================================================
  // Staged IMU mode (imu/init_mode: staged; default legacy = FAST-LIVO2).
  // Gyro-aided depth-ICP warm-up, then static or excitation-gated dynamic
  // initialisation of gravity / gyro bias / velocity (inertial_init.h), then
  // strapdown propagation with continuous-time noise and free gravity.
  // ========================================================================
  bool   staged_imu_          = false;
  bool   staged_inited_       = false;
  bool   staged_bias_inflated_= false;
  double staged_init_time_    = -1.0;
  double staged_last_prop_    = -1.0;   // end time of the last propagation
  inertial::NoiseParams imu_noise_;
  double gravity_norm_        = 9.81;
  double acc_scale_           = 1.0;    // accelerometer unit factor
  double warmup_vel_rw_       = 2.0;    // m/s/sqrt(s), CV-model velocity walk
  double static_window_       = 0.5;    // s
  double static_max_gyr_      = 0.03;   // rad/s, |mean gyro|
  double static_max_acc_std_  = 0.25;   // m/s^2, per-axis std
  double static_max_motion_   = 0.02;   // m, ICP displacement over the window
  double dyn_window_          = 1.5;    // s
  int    dyn_min_frames_      = 15;
  double dyn_min_excitation_  = 0.25;   // m/s^2
  double dyn_max_g_error_     = 0.5;    // m/s^2, | |g_free| - g |
  double init_sigma_v_static_ = 0.01, init_sigma_v_dyn_ = 0.1;
  double init_sigma_g_static_ = 0.02, init_sigma_g_dyn_ = 0.1;
  double init_sigma_bg_       = 0.005, init_sigma_ba_  = 0.1;
  double bias_inflate_after_  = 5.0;    // s after initialisation, once
  double bias_inflate_bg_     = 0.02, bias_inflate_ba_ = 0.2;
  std::deque<inertial::ImuSample> staged_imu_hist_;
  std::deque<inertial::Frame>     staged_frames_;
  bool   staged_have_last_    = false;
  inertial::ImuSample staged_last_sample_;
};

#endif // RGBD_MAPPER_H
