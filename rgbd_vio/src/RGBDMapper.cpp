/*
RGB-D VIO for UAV Pest Control — UCLA
Based on FAST-LIVO2 (GPLv2). See ../FAST-LIVO2-main/LICENSE.

Key design decisions
--------------------
1. The D455 depth image is converted to PointCloudXYZI (DepthProcessor) and
   treated as a "LiDAR scan" with zero scan duration (all curvature = 0).
   This lets us reuse the FAST-LIVO2 LIO pipeline (voxel map + ICP + EKF)
   without any changes to voxel_map.cpp or IMU_Processing.cpp.

2. Colour (RGB) images go through the unmodified VIO pipeline.

3. sync_packages() implements a two-step state machine per depth frame:
     WAIT → LIO  (depth cloud + IMU)
     LIO  → VIO  (matching colour frame, if img_en=1)
     VIO  → WAIT (ready for next depth frame)
   This mirrors the original LIVO interleaving but simplified for the
   instantaneous (zero-duration) depth frame.

4. IMU-based high-rate odometry is published on /LIVO2/imu_propagate for
   flight controller consumption (same as original).
*/

#include "RGBDMapper.h"
#include <omp.h>
#include <iomanip>

// ============================================================================
//  Constructor / destructor
// ============================================================================

RGBDMapper::RGBDMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0), extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_imu.reset(new ImuProcess());
  depth_proc.reset(new DepthProcessor());

  readParameters(nh);

  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());

  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());

  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();

  path.header.stamp    = ros::Time::now();
  path.header.frame_id = "camera_init";
}

RGBDMapper::~RGBDMapper() {}

// ============================================================================
//  Parameter loading
// ============================================================================

void RGBDMapper::readParameters(ros::NodeHandle &nh)
{
  // ---- Sensor topics -----
  nh.param<string>("common/depth_topic",      depth_topic,      "/camera/depth/image_rect_raw");
  nh.param<string>("common/depth_info_topic", depth_info_topic, "/camera/depth/camera_info");
  nh.param<string>("common/imu_topic",        imu_topic,        "/camera/imu");
  nh.param<string>("common/img_topic",        img_topic,        "/camera/color/image_raw");
  nh.param<int>   ("common/img_en",           img_en,           1);
  nh.param<bool>  ("common/depth_en",          depth_en,         true);
  nh.param<bool>  ("common/lio_en",            lio_en,           true);

  // ---- VIO -----
  nh.param<bool>  ("vio/normal_en",              normal_en,              true);
  nh.param<bool>  ("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>   ("vio/max_iterations",         max_iterations,         5);
  nh.param<double>("vio/img_point_cov",          IMG_POINT_COV,          100.0);
  nh.param<bool>  ("vio/raycast_en",             raycast_en,             false);
  nh.param<bool>  ("vio/exposure_estimate_en",   exposure_estimate_en,   false);
  nh.param<double>("vio/inv_expo_cov",           inv_expo_cov,           0.2);
  nh.param<int>   ("vio/grid_size",              grid_size,              5);
  nh.param<int>   ("vio/grid_n_height",          grid_n_height,          17);
  nh.param<int>   ("vio/patch_pyrimid_level",    patch_pyrimid_level,    3);
  nh.param<int>   ("vio/patch_size",             patch_size,             8);
  nh.param<double>("vio/outlier_threshold",      outlier_threshold,      1000.0);
  nh.param<double>("vio/gate_chi2",              vio_gate_chi2,          0.0);
  nh.param<bool>  ("vio/lba_en",                 vio_lba_en,             false);
  nh.param<int>   ("vio/lba_window",             vio_lba_window,         6);
  nh.param<int>   ("vio/lba_period",             vio_lba_period,         5);
  nh.param<int>   ("vio/lba_iters",              vio_lba_iters,          4);
  nh.param<int>   ("vio/lba_max_points",         vio_lba_max_points,     300);
  nh.param<int>   ("vio/lba_max_obs",            vio_lba_max_obs,        6);
  nh.param<double>("vio/lba_sigma_rot",          vio_lba_sigma_rot,      0.005);
  nh.param<double>("vio/lba_sigma_pos",          vio_lba_sigma_pos,      0.02);
  nh.param<double>("vio/lba_huber",              vio_lba_huber,          20.0);
  nh.param<double>("vio/lba_max_move",           vio_lba_max_move,       0.1);

  // ---- Time offsets -----
  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset",    img_time_offset,    0.0);
  nh.param<double>("time_offset/imu_time_offset",    imu_time_offset,    0.0);
  nh.param<double>("time_offset/depth_time_offset",  depth_time_offset,  0.0);

  // ---- UAV / flight controller -----
  nh.param<bool>("uav/imu_rate_odom",    imu_prop_enable,  false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  // ---- EVO / logging -----
  nh.param<string>("evo/seq_name",       seq_name,       "d455_01");
  nh.param<bool>  ("evo/pose_output_en", pose_output_en, false);
  nh.param<bool>  ("evo/cov_output_en",  cov_output_en,  false);

  // ---- IMU noise model -----
  nh.param<double>("imu/gyr_cov",      gyr_cov,       1.0);
  nh.param<double>("imu/acc_cov",      acc_cov,       1.0);
  nh.param<int>   ("imu/imu_int_frame",imu_int_frame, 3);
  nh.param<bool>  ("imu/imu_en",       imu_en,        true);
  nh.param<bool>  ("imu/gravity_est_en",gravity_est_en,true);
  nh.param<bool>  ("imu/ba_bg_est_en", ba_bg_est_en,  true);

  // ---- Staged IMU mode (see inertial_init.h); "legacy" = FAST-LIVO2 -----
  std::string init_mode;
  nh.param<string>("imu/init_mode",           init_mode,                "legacy");
  staged_imu_ = (init_mode == "staged");
  nh.param<double>("imu/gyr_noise_density",   imu_noise_.gyr_nd,        2e-4);
  nh.param<double>("imu/acc_noise_density",   imu_noise_.acc_nd,        2e-3);
  nh.param<double>("imu/gyr_random_walk",     imu_noise_.gyr_rw,        2e-5);
  nh.param<double>("imu/acc_random_walk",     imu_noise_.acc_rw,        5e-4);
  nh.param<double>("imu/gravity_random_walk", imu_noise_.grav_rw,       1e-3);
  nh.param<double>("imu/gap_thresh",          imu_noise_.gap_thresh,    0.025);
  nh.param<double>("imu/gap_factor",          imu_noise_.gap_factor,    100.0);
  nh.param<bool>  ("imu/legacy_noise_scaling",imu_noise_.discrete,      false);
  nh.param<double>("imu/gyr_cov",             imu_noise_.gyr_cov,       0.1);
  nh.param<double>("imu/acc_cov",             imu_noise_.acc_cov,       0.1);
  nh.param<double>("imu/b_gyr_cov",           imu_noise_.bgyr_cov,      1e-4);
  nh.param<double>("imu/b_acc_cov",           imu_noise_.bacc_cov,      1e-4);
  nh.param<bool>  ("imu/legacy_init",         legacy_init_,             false);
  nh.param<bool>  ("imu/init_cov_reset",      init_cov_reset_,          true);
  nh.param<double>("imu/gravity_norm",        gravity_norm_,            9.81);
  nh.param<double>("imu/acc_scale",           acc_scale_,               1.0);  // 9.81 for IMUs reporting in g (Livox)
  nh.param<double>("imu/warmup_vel_rw",       warmup_vel_rw_,           2.0);
  nh.param<double>("imu/static_window",       static_window_,           0.5);
  nh.param<double>("imu/static_max_gyr",      static_max_gyr_,          0.03);
  nh.param<double>("imu/static_max_acc_std",  static_max_acc_std_,      0.25);
  nh.param<double>("imu/static_max_motion",   static_max_motion_,       0.02);
  nh.param<double>("imu/dyn_window",          dyn_window_,              1.5);
  nh.param<int>   ("imu/dyn_min_frames",      dyn_min_frames_,          15);
  nh.param<double>("imu/dyn_min_excitation",  dyn_min_excitation_,      0.25);
  nh.param<double>("imu/dyn_max_g_error",     dyn_max_g_error_,         0.5);
  nh.param<double>("imu/init_sigma_v_static", init_sigma_v_static_,     0.01);
  nh.param<double>("imu/init_sigma_v_dyn",    init_sigma_v_dyn_,        0.1);
  nh.param<double>("imu/init_sigma_g_static", init_sigma_g_static_,     0.02);
  nh.param<double>("imu/init_sigma_g_dyn",    init_sigma_g_dyn_,        0.1);
  nh.param<double>("imu/init_sigma_bg",       init_sigma_bg_,           0.005);
  nh.param<double>("imu/init_sigma_ba",       init_sigma_ba_,           0.1);
  nh.param<double>("imu/bias_inflate_after",  bias_inflate_after_,      5.0);
  nh.param<double>("imu/bias_inflate_bg",     bias_inflate_bg_,         0.02);
  nh.param<double>("imu/bias_inflate_ba",     bias_inflate_ba_,         0.2);
  if (staged_imu_)
  {
    gravity_align_en = false;  // world = map frame; gravity is a free state
    ROS_INFO("[StagedIMU] enabled: gyr_nd=%.2e acc_nd=%.2e gyr_rw=%.2e acc_rw=%.2e grav_rw=%.2e",
             imu_noise_.gyr_nd, imu_noise_.acc_nd, imu_noise_.gyr_rw, imu_noise_.acc_rw,
             imu_noise_.grav_rw);
  }

  // ---- Depth pre-processing -----
  nh.param<double>("depth/min_depth",        depth_proc->min_depth,        0.3);
  nh.param<double>("depth/max_depth",        depth_proc->max_depth,        5.0);
  nh.param<double>("depth/depth_scale",      depth_proc->depth_scale,      1e-3);
  nh.param<int>   ("depth/point_filter_num", depth_proc->point_filter_num, 4);

  // ---- LIO filter / voxel -----
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.1);

  // ---- Extrinsics (depth-cam body → IMU body) -----
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  // Camera (colour) → depth-cam body
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());

  // ---- Debug -----
  nh.param<double>("debug/plot_time", plot_time, -10.0);
  nh.param<int>   ("debug/frame_cnt", frame_cnt,  6);

  // ---- Publishing -----
  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>   ("publish/pub_scan_num",     pub_scan_num,     1);
  nh.param<bool>  ("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>  ("publish/dense_map_en",     dense_map_en,     false);

  // ---- PCD saving -----
  nh.param<int>   ("pcd_save/interval",        pcd_save_interval, -1);
  nh.param<bool>  ("pcd_save/pcd_save_en",     pcd_save_en,       false);
  nh.param<int>   ("pcd_save/type",            pcd_save_type,     0);
  nh.param<bool>  ("pcd_save/colmap_output_en",colmap_output_en,  false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd,   0.15);

  nh.param<bool>  ("image_save/img_save_en",   img_save_en,       false);
  nh.param<int>   ("image_save/interval",      img_save_interval, 1);

  // Depth noise model
  nh.param<double>("depth/noise_frac",    depth_noise_frac,  0.01);
  nh.param<double>("depth/lateral_noise", lateral_noise_m,   0.005);

  // Flying-pixel rejection
  nh.param<bool> ("depth/flying_pixel_en",    depth_proc->flying_pixel_en,    true);
  nh.param<float>("depth/flying_pixel_thresh", depth_proc->flying_pixel_thresh, 0.10f);

  // Depth-stratified sampling (heuristic fallback)
  nh.param<bool>("depth/depth_bins_en",    depth_proc->depth_bins_en,    true);
  nh.param<int> ("depth/num_depth_bins",   depth_proc->num_depth_bins,   10);
  nh.param<int> ("depth/max_pts_per_bin",  depth_proc->max_pts_per_bin,  2500);
  nh.param<int> ("depth/fallback_total_cap", depth_proc->fallback_total_cap, 0);

  // Information-Theoretic Point Selection (ITPS)
  nh.param<bool>  ("depth/info_select_en",    depth_proc->info_select_en,    true);
  nh.param<double>("depth/noise_frac",        depth_proc->noise_frac,        0.01);
  nh.param<int>   ("depth/target_points",     depth_proc->target_points,     20000);
  nh.param<double>("depth/rebalance_alpha",   depth_proc->rebalance_alpha,   0.5);
  nh.param<int>   ("depth/itps_min_candidates", depth_proc->itps_min_candidates, 0);
  nh.param<bool>  ("depth/itps_uniform_random", depth_proc->itps_uniform_random, false);
  nh.param<std::string>("depth/itps_fill", depth_proc->itps_fill, "raster");
  nh.param<double>("depth/inject_lateral_keep", depth_proc->inject_lateral_keep, 1.0);

  // Degeneracy detection (ITPS uses isotropy threshold; fallback uses σ_min)
  nh.param<double>("lio/deg_threshold",          deg_threshold_,          0.07);
  nh.param<double>("lio/deg_isotropy_thresh",    deg_isotropy_thresh_,    0.10);
  nh.param<double>("lio/skip_vio_sigma_thresh",  skip_vio_sigma_thresh_,  0.3);
  nh.param<double>("lio/meas_var_scale",         meas_var_scale_,         1.0);

  // Observability-Constrained EKF (OC-EKF)
  nh.param<bool>  ("lio/oc_ekf_en",            oc_ekf_en_,            true);
  nh.param<double>("lio/oc_lambda_thresh",     oc_lambda_thresh_,     0.05);
  nh.param<double>("lio/oc_lambda_lo",         oc_lambda_lo_,         0.01);
  nh.param<double>("lio/oc_lambda_hi",         oc_lambda_hi_,         0.10);
  nh.param<double>("lio/oc_adapt_kappa_knee",  oc_adapt_kappa_knee_,  20.0);
  nh.param<bool>  ("lio/oc_adaptive_thresh_en",oc_adaptive_thresh_en_,true);
  nh.param<double>("lio/oc_cov_inflate",       oc_cov_inflate_,       10.0);
  nh.param<bool>  ("lio/oc_persist_guard_en", oc_persist_guard_en_,  false);
  nh.param<int>   ("lio/oc_persist_frames",   oc_persist_frames_,    30);
  nh.param<std::string>("lio/oc_cov_mode",    oc_cov_mode_,          std::string("splice"));
  nh.param<bool>  ("lio/oc_sched_from_ratio", oc_sched_from_ratio_,  false);
  nh.param<bool>  ("lio/oc_icp_cov_from_config", oc_icp_cov_from_config_, false);
  nh.param<bool>  ("lio/oc_after_plain",     oc_after_plain_,       false);

  // Degeneracy-aware ICP update (dtf.h)
  std::string dtf_mode;
  nh.param<string>("lio/dtf_mode",    dtf_mode,     "upstream");
  nh.param<double>("lio/dtf_thresh",  dtf_thresh_,  0.05);
  nh.param<double>("lio/dtf_inflate", dtf_inflate_, 1.0);
  nh.param<double>("lio/dept_err_quad", dept_err_quad_, 0.0);  // stereo depth: sigma_z += k z^2
  dtf_ours_ = (dtf_mode != "upstream");
  dtf_mode_ = dtf_mode == "trans" ? dtf::Mode::Trans
            : dtf_mode == "full"  ? dtf::Mode::Full : dtf::Mode::Plain;
  if (dtf_ours_)
    ROS_INFO("[DTF] mode=%s thresh=%.3f inflate=%.1f dept_err_quad=%.4f (OC-EKF / Tikhonov post-processing off)",
             dtf_mode.c_str(), dtf_thresh_, dtf_inflate_, dept_err_quad_);

  // Sprint 1 #9: Pose linearity window
  nh.param<int>("debug/linearity_window", linearity_window_, 20);

  // Sprint 3 #11: Long-term visual map
  nh.param<int>   ("local_map/lv_max_pts",        lv_map_max_pts_,    50000);
  nh.param<double>("local_map/lv_voxel_size",      lv_voxel_size_,     2.0);
  nh.param<int>   ("local_map/lv_update_interval", lv_update_interval_,30);

  // Derived blind-zone for depth processor
  depth_proc->blind_sqr = depth_proc->min_depth * depth_proc->min_depth;
}

// ============================================================================
//  Component initialisation
// ============================================================================

void RGBDMapper::initializeComponents()
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min,
                                  filter_size_surf_min,
                                  filter_size_surf_min);

  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs("rgbdMapping", vio_manager->cam))
    throw std::runtime_error("Camera model not correctly specified in laserMapping NS.");

  vio_manager->grid_size             = grid_size;
  vio_manager->patch_size            = patch_size;
  vio_manager->outlier_threshold     = outlier_threshold;
  vio_manager->gate_chi2             = vio_gate_chi2;
  vio_manager->lba_en                = vio_lba_en;
  vio_manager->lba_window            = vio_lba_window;
  vio_manager->lba_period            = vio_lba_period;
  vio_manager->lba_iters             = vio_lba_iters;
  vio_manager->lba_max_points        = vio_lba_max_points;
  vio_manager->lba_max_obs           = vio_lba_max_obs;
  vio_manager->lba_sigma_rot         = vio_lba_sigma_rot;
  vio_manager->lba_sigma_pos         = vio_lba_sigma_pos;
  vio_manager->lba_huber             = vio_lba_huber;
  vio_manager->lba_max_move          = vio_lba_max_move;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state                 = &_state;
  vio_manager->state_propagat        = &state_propagat;
  vio_manager->max_iterations        = max_iterations;
  vio_manager->img_point_cov         = IMG_POINT_COV;
  vio_manager->normal_en             = normal_en;
  vio_manager->inverse_composition_en= inverse_composition_en;
  vio_manager->raycast_en            = raycast_en;
  vio_manager->grid_n_width          = grid_n_width;
  vio_manager->grid_n_height         = grid_n_height;
  vio_manager->patch_pyrimid_level   = patch_pyrimid_level;
  vio_manager->exposure_estimate_en  = exposure_estimate_en;
  vio_manager->colmap_output_en      = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en)             p_imu->disable_imu();
  if (!gravity_est_en)     p_imu->disable_gravity_est();
  if (!ba_bg_est_en)       p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  // SLAM mode selection
  slam_mode_ = (img_en && depth_en) ? LIVO
             : imu_en               ? ONLY_LIO
                                    : ONLY_LO;
}

void RGBDMapper::initializeFiles()
{
  fout_pre.open(std::string(ROOT_DIR) + "Log/mat_pre.txt",  std::ios::out);
  fout_out.open(std::string(ROOT_DIR) + "Log/mat_out.txt",  std::ios::out);
}

// ============================================================================
//  ROS subscribers / publishers
// ============================================================================

void RGBDMapper::initializeSubscribersAndPublishers(
    ros::NodeHandle &nh, image_transport::ImageTransport &it)
{
  // Depth image + camera info
  sub_depth      = nh.subscribe(depth_topic,      200000,
                                &RGBDMapper::depth_cbk, this);
  sub_depth_info = nh.subscribe(depth_info_topic, 1,
                                &RGBDMapper::depth_info_cbk, this);
  // IMU
  sub_imu = nh.subscribe(imu_topic, 200000, &RGBDMapper::imu_cbk, this);
  // Colour image
  sub_img = nh.subscribe(img_topic, 200000, &RGBDMapper::img_cbk, this);

  // Publishers
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal            = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap      = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map", 100);
  pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap     = nh.advertise<sensor_msgs::PointCloud2>("/depth_map", 100);
  pubOdomAftMapped     = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath              = nh.advertise<nav_msgs::Path>("/path", 10);
  mavros_pose_publisher= nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage             = it.advertise("/rgb_img", 1);
  pubImuPropOdom       = nh.advertise<nav_msgs::Odometry>("/RGBD_VIO/imu_propagate", 10000);
  voxelmap_manager->voxel_map_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

  imu_prop_timer = nh.createTimer(ros::Duration(0.004),
                                  &RGBDMapper::imu_prop_callback, this);
}

// ============================================================================
//  Sensor callbacks
// ============================================================================

void RGBDMapper::depth_info_cbk(const sensor_msgs::CameraInfo::ConstPtr &info_msg)
{
  depth_proc->setCameraInfo(info_msg);
}

void RGBDMapper::depth_cbk(const sensor_msgs::ImageConstPtr &depth_msg)
{
  if (!depth_en) return;

  double cur_time = depth_msg->header.stamp.toSec() + depth_time_offset;

  // Guard against time reversals
  if (cur_time < last_timestamp_depth)
  {
    ROS_ERROR("[RGBDMapper] Depth time loop-back, clearing buffer.");
    depth_cloud_buffer.clear();
    depth_time_buffer.clear();
  }

  // Convert ROS depth image → OpenCV Mat
  cv::Mat depth_cv;
  try {
    depth_cv = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge depth exception: %s", e.what());
    return;
  }

  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  const double t_sel = omp_get_wtime();
  depth_proc->process(depth_cv, cv::Mat(), cur_time, cloud,
                      &last_selection_info_);
  printf("[Timing] depth selection %.6f s, %zu points\n", omp_get_wtime() - t_sel, cloud->size());

  if (!cloud || cloud->empty())
  {
    ROS_WARN_THROTTLE(2.0, "[RGBDMapper] Empty depth cloud — check intrinsics.");
    return;
  }

  mtx_buffer.lock();
  depth_cloud_buffer.push_back(cloud);
  depth_time_buffer.push_back(cur_time);
  last_timestamp_depth = cur_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void RGBDMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;
  if (last_timestamp_depth < 0.0) return;

  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp =
      ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double ts = msg->header.stamp.toSec();

  mtx_buffer.lock();
  if (last_timestamp_imu > 0.0 && ts < last_timestamp_imu)
  {
    ROS_ERROR("[RGBDMapper] IMU loop-back: %.4f", last_timestamp_imu - ts);
    mtx_buffer.unlock();
    return;
  }
  last_timestamp_imu = ts;
  imu_buffer.push_back(msg);
  mtx_buffer.unlock();

  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (!p_imu->imu_need_init) prop_imu_buffer.push_back(*msg);
    newest_imu = *msg;
    new_imu    = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat RGBDMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &msg)
{
  return cv_bridge::toCvCopy(msg, "bgr8")->image;
}

void RGBDMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;

  double ts = msg_in->header.stamp.toSec() + img_time_offset;
  if (std::abs(ts - last_timestamp_img) < 0.001) return;
  if (last_timestamp_depth < 0) return;
  if (ts < last_timestamp_img)
  {
    ROS_ERROR("[RGBDMapper] Image loop-back.");
    return;
  }
  if (ts - last_timestamp_img < 0.02)
  {
    ROS_WARN_THROTTLE(1.0, "[RGBDMapper] Image rate too high, dropping.");
    return;
  }

  mtx_buffer.lock();
  img_buffer.push_back(getImageFromMsg(msg_in));
  img_time_buffer.push_back(ts);
  last_timestamp_img = ts;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

// ============================================================================
//  Synchronisation — simplified two-step state machine for depth camera
//
//  Per depth frame the sequence is:
//    (1) Return LIO bundle   →  handleLIO()  builds/updates voxel map
//    (2) Return VIO bundle   →  handleVIO()  tracks colour patches
//    (3) WAIT for next frame
// ============================================================================

bool RGBDMapper::sync_packages(LidarMeasureGroup &meas)
{
  // Guard: need at least one depth frame and, if enabled, IMU data
  if (depth_cloud_buffer.empty() && depth_en) return false;
  if (imu_buffer.empty() && imu_en)           return false;

  EKF_STATE last_flag = meas.lio_vio_flg;

  // ---- After VIO: reset and fall through to pick up next depth frame ----
  if (last_flag == VIO)
  {
    meas.measures.clear();
    meas.lio_vio_flg = WAIT;
    last_flag        = WAIT;
  }

  // ---- After LIO: look for a colour image to run VIO -------------------
  if (last_flag == LIO)
  {
    if (img_en && !img_buffer.empty())
    {
      double target = meas.last_lio_update_time;

      // Discard stale colour frames (more than 50 ms before depth frame)
      mtx_buffer.lock();
      while (img_buffer.size() > 1 &&
             img_time_buffer.front() < target - 0.05)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
      }
      // Use the colour frame nearest to the depth frame, not the oldest one in the window: with
      // equal-rate streams the oldest is often the previous frame (33 ms stale at 30 Hz), and the
      // state cannot be propagated back to it.
      while (img_buffer.size() > 1 &&
             std::abs(img_time_buffer[1] - target) <= std::abs(img_time_buffer.front() - target))
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
      }

      bool has_match = (!img_buffer.empty() &&
                        std::abs(img_time_buffer.front() - target) < 0.05);
      mtx_buffer.unlock();

      if (has_match)
      {
        struct MeasureGroup m;
        mtx_buffer.lock();
        m.vio_time = img_time_buffer.front();
        m.img      = img_buffer.front();
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();

        static int n_pairs = 0, n_before = 0;
        static double sum_dt = 0.0;
        const double dt_img = m.vio_time - target;
        ++n_pairs; sum_dt += dt_img; n_before += (dt_img < -1e-3);
        ROS_INFO_THROTTLE(10.0, "[VIO] colour-depth pairing: mean offset %.1f ms, colour before depth %.0f%% (%d pairs)",
                          1e3 * sum_dt / n_pairs, 100.0 * n_before / n_pairs, n_pairs);

        meas.measures.push_back(m);
        meas.lio_vio_flg = VIO;
        return true;
      }
    }
    // No matching image — skip VIO this cycle, reset for next depth frame
    meas.measures.clear();
    meas.lio_vio_flg = WAIT;
    last_flag        = WAIT;
  }

  // ---- WAIT: prepare next depth frame for LIO --------------------------
  if (last_flag == WAIT || last_flag == LO)
  {
    if (!depth_en || depth_cloud_buffer.empty()) return false;

    double depth_time = depth_time_buffer.front();

    // Wait until IMU data reaches past this depth frame
    if (imu_en && last_timestamp_imu < depth_time) return false;

    mtx_buffer.lock();
    PointCloudXYZI::Ptr cloud = depth_cloud_buffer.front();
    depth_cloud_buffer.pop_front();
    depth_time_buffer.pop_front();
    mtx_buffer.unlock();

    if (!cloud || cloud->size() <= 1) return false;

    // Depth frames are instantaneous: begin_time == end_time
    meas.lidar              = cloud;
    meas.lidar_frame_beg_time = depth_time;
    meas.lidar_frame_end_time = depth_time;
    if (meas.last_lio_update_time < 0.0)
      meas.last_lio_update_time = depth_time;
    meas.last_lio_update_time = depth_time;
    meas.pcl_proc_cur         = cloud;

    // Collect IMU from last update to current depth frame
    struct MeasureGroup m;
    m.lio_time = depth_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty() &&
           imu_buffer.front()->header.stamp.toSec() <= depth_time)
    {
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    mtx_buffer.unlock();

    meas.measures.clear();
    meas.measures.push_back(m);
    meas.lio_vio_flg = LIO;
    return true;
  }

  return false;
}

// ============================================================================
//  Main run loop
// ============================================================================

void RGBDMapper::handleFirstFrame()
{
  if (!is_first_frame)
  {
    _first_depth_time          = DepthMeasures.last_lio_update_time;
    p_imu->first_lidar_time    = _first_depth_time;
    is_first_frame             = true;
    ROS_INFO("[RGBDMapper] First depth frame registered.");
  }
}

void RGBDMapper::gravityAlignment()
{
  if (!p_imu->imu_need_init && !gravity_align_finished)
  {
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0         = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;

    gravity_align_finished = true;
    ROS_INFO("[RGBDMapper] Gravity alignment done.");
  }
}

void RGBDMapper::processImu()
{
  if (staged_imu_ && imu_en)
  {
    stagedProcessImu();
    return;
  }
  // Process2() undistorts the point cloud and propagates IMU state.
  // For depth frames all curvature=0 → undistortion is identity (correct).
  p_imu->Process2(DepthMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat                    = _state;
  voxelmap_manager->state_           = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;
}

// ============================================================================
//  Staged IMU mode (imu/init_mode: staged) — see include/inertial_init.h
// ============================================================================

static_assert(DIM_STATE == inertial::kDim, "inertial_init assumes FAST-LIVO2's 19-D state");

void RGBDMapper::stagedProcessImu()
{
  MeasureGroup &meas = DepthMeasures.measures.back();
  auto sample = [this](const sensor_msgs::Imu::ConstPtr &m) {
    return inertial::ImuSample{
        m->header.stamp.toSec(),
        acc_scale_ * V3D(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z),
        V3D(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z)};
  };
  const bool lio = (DepthMeasures.lio_vio_flg == LIO);
  std::vector<inertial::ImuSample> s;
  if (staged_have_last_) s.push_back(staged_last_sample_);
  if (lio)
  {
    for (const auto &m : meas.imu)
    {
      s.push_back(sample(m));
      staged_imu_hist_.push_back(s.back());
    }
    if (!s.empty())
    {
      staged_last_sample_ = s.back();
      staged_have_last_ = true;
      while (!staged_imu_hist_.empty() &&
             staged_imu_hist_.front().t < s.back().t - (dyn_window_ + static_window_ + 1.0))
        staged_imu_hist_.pop_front();
    }
    *feats_undistort = *DepthMeasures.pcl_proc_cur;  // depth frames: no intra-scan motion
  }
  else
  {
    // VIO bundles carry no IMU: read, without consuming, the buffered samples
    // up to the image time (the next depth frame consumes them as usual).
    std::lock_guard<std::mutex> lock(mtx_buffer);
    for (const auto &m : imu_buffer)
    {
      if (m->header.stamp.toSec() > meas.vio_time) break;
      s.push_back(sample(m));
    }
  }

  {
    const double t_end = lio ? meas.lio_time : meas.vio_time;
    if (staged_last_prop_ >= 0.0 && t_end > staged_last_prop_)
    {
      inertial::NavState x;
      x.R = _state.rot_end;  x.p = _state.pos_end;  x.v = _state.vel_end;
      x.bg = _state.bias_g;  x.ba = _state.bias_a;  x.g = _state.gravity;
      x.P = _state.cov;
      const auto segs = inertial::segments(s, staged_last_prop_, t_end, imu_noise_.gap_thresh);
      if (staged_inited_)
        inertial::propagate(x, segs, imu_noise_, ba_bg_est_en, gravity_est_en);
      else
        inertial::propagateWarmup(x, segs, imu_noise_.gyr_nd, warmup_vel_rw_);
      _state.rot_end = x.R;  _state.pos_end = x.p;  _state.vel_end = x.v;
      _state.bias_g = x.bg;  _state.bias_a = x.ba;  _state.gravity = x.g;
      _state.cov = x.P;

      if (staged_inited_ && !staged_bias_inflated_ &&
          t_end - staged_init_time_ > bias_inflate_after_)
      {
        // Let the biases re-converge once the window contains real motion.
        _state.cov.block<3, 3>(10, 10) += M3D::Identity() * bias_inflate_bg_ * bias_inflate_bg_;
        _state.cov.block<3, 3>(13, 13) += M3D::Identity() * bias_inflate_ba_ * bias_inflate_ba_;
        staged_bias_inflated_ = true;
        ROS_INFO("[StagedIMU] bias covariance inflated %.1f s after initialisation",
                 t_end - staged_init_time_);
      }
    }
    if (t_end > staged_last_prop_) staged_last_prop_ = t_end;
  }

  state_propagat                     = _state;
  voxelmap_manager->state_           = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;
}

void RGBDMapper::stagedAfterLIO()
{
  const double t = DepthMeasures.last_lio_update_time;
  if (staged_inited_) return;

  staged_frames_.push_back({t, _state.rot_end, _state.pos_end});

  if (legacy_init_)
  {
    // FAST-LIVO2-style: gravity from the mean specific force of the first samples, bias from the mean
    // gyroscope, zero velocity, no stationarity or excitation test; every later accelerometer sample is
    // rescaled by g0 / |mean acc|.  Only the initialization differs from the staged route.
    if (staged_imu_hist_.size() < 20) return;
    V3D ma = V3D::Zero(), mg = V3D::Zero();
    for (const auto &smp : staged_imu_hist_) { ma += smp.acc; mg += smp.gyr; }
    ma /= double(staged_imu_hist_.size()); mg /= double(staged_imu_hist_.size());
    acc_scale_ *= gravity_norm_ / std::max(ma.norm(), 1e-6);
    stagedFinishInit(_state.rot_end * (-ma.normalized() * gravity_norm_), V3D::Zero(), mg,
                     init_sigma_v_static_, init_sigma_g_static_, "legacy-style");
    return;
  }
  while (staged_frames_.size() > 2 && staged_frames_.front().t < t - dyn_window_ - 1e-6)
    staged_frames_.pop_front();

  // 1) Stationary: quiet IMU and negligible depth-ICP displacement.
  std::vector<inertial::ImuSample> win;
  for (const auto &smp : staged_imu_hist_)
    if (smp.t >= t - static_window_ && smp.t <= t) win.push_back(smp);
  const inertial::Frame *f0 = nullptr;
  for (const auto &f : staged_frames_)
    if (f.t >= t - static_window_) { f0 = &f; break; }
  if (f0 && win.size() >= 10 && win.back().t - win.front().t >= 0.9 * static_window_ &&
      t - f0->t >= 0.8 * static_window_)
  {
    const auto st = inertial::initStatic(win, gravity_norm_, static_max_gyr_, static_max_acc_std_);
    if (st.ok && (_state.pos_end - f0->p).norm() < static_max_motion_)
    {
      stagedFinishInit(_state.rot_end * st.g_imu, V3D::Zero(), st.bg, init_sigma_v_static_,
                       init_sigma_g_static_, "static");
      return;
    }
  }

  // 2) Moving: visual-inertial alignment over the frame window.
  if (static_cast<int>(staged_frames_.size()) >= dyn_min_frames_ &&
      staged_frames_.back().t - staged_frames_.front().t >= 0.9 * dyn_window_)
  {
    const std::vector<inertial::Frame> fr(staged_frames_.begin(), staged_frames_.end());
    const std::vector<inertial::ImuSample> imu(staged_imu_hist_.begin(), staged_imu_hist_.end());
    const auto dy = inertial::initDynamic(fr, imu, gravity_norm_, dyn_min_excitation_,
                                          dyn_max_g_error_);
    if (dy.ok)
    {
      stagedFinishInit(dy.g_w, dy.v.back(), dy.bg, init_sigma_v_dyn_, init_sigma_g_dyn_,
                       "dynamic");
      return;
    }
    ROS_INFO_THROTTLE(1.0, "[StagedIMU] waiting: %s (excitation %.3f m/s^2, |g| %.3f)",
                      dy.why.c_str(), dy.excitation, dy.g_norm_free);
  }
}

void RGBDMapper::stagedFinishInit(const V3D &g_w, const V3D &v, const V3D &bg,
                                  double sigma_v, double sigma_g, const char *how)
{
  _state.gravity = g_w;
  _state.vel_end = v;
  _state.bias_g  = bg;
  _state.bias_a.setZero();
  // Keep the ICP attitude/position covariance; restart the inertial blocks (imu/init_cov_reset).
  if (init_cov_reset_)
  {
    auto &P = _state.cov;
    P.block<12, DIM_STATE>(7, 0).setZero();
    P.block<DIM_STATE, 12>(0, 7).setZero();
    P.block<3, 3>(7, 7)   = M3D::Identity() * sigma_v * sigma_v;
    P.block<3, 3>(10, 10) = M3D::Identity() * init_sigma_bg_ * init_sigma_bg_;
    P.block<3, 3>(13, 13) = M3D::Identity() * init_sigma_ba_ * init_sigma_ba_;
    P.block<3, 3>(16, 16) = M3D::Identity() * sigma_g * sigma_g;
  }
  staged_inited_    = true;
  staged_init_time_ = DepthMeasures.last_lio_update_time;
  staged_frames_.clear();
  ROS_INFO("[StagedIMU] %s initialisation at t=%.2f s: g=(%.3f %.3f %.3f) |v|=%.3f "
           "bg=(%.4f %.4f %.4f)", how, staged_init_time_ - _first_depth_time,
           g_w.x(), g_w.y(), g_w.z(), v.norm(), bg.x(), bg.y(), bg.z());
}

// ============================================================================
//  Degeneracy-aware ICP update (lio/dtf_mode: plain | trans | full)
//
//  VoxelMapManager::StateEstimation (FAST-LIVO2) verbatim, except that each
//  iteration's registration information (H^T R^-1 H, H^T R^-1 z) goes through
//  dtf::reduce() before the IEKF solve.  dtf_mode: plain reproduces upstream.
// ============================================================================

void RGBDMapper::dtfStateEstimation(StatesGroup &state_propagat)
{
  VoxelMapManager &vm = *voxelmap_manager;
  vm.cross_mat_list_.clear();
  vm.cross_mat_list_.reserve(vm.feats_down_size_);
  vm.body_cov_list_.clear();
  vm.body_cov_list_.reserve(vm.feats_down_size_);
  for (size_t i = 0; i < vm.feats_down_body_->size(); i++)
  {
    V3D point_this(vm.feats_down_body_->points[i].x, vm.feats_down_body_->points[i].y,
                   vm.feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    calcBodyCov(point_this, vm.config_setting_.dept_err_ + dept_err_quad_ * point_this[2] * point_this[2],
                vm.config_setting_.beam_err_, var);
    vm.body_cov_list_.push_back(var);
    point_this = vm.extR_ * point_this + vm.extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    vm.cross_mat_list_.push_back(point_crossmat);
  }
  std::vector<pointWithVar>().swap(vm.pv_list_);
  vm.pv_list_.resize(vm.feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();
  dtf::Result dr;

  bool flg_EKF_converged, EKF_stop_flg = false;
  for (int iterCount = 0; iterCount < vm.config_setting_.max_iterations_; iterCount++)
  {
    double total_residual = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    vm.TransformLidar(vm.state_.rot_end, vm.state_.pos_end, vm.feats_down_body_, world_lidar);
    M3D rot_var = vm.state_.cov.block<3, 3>(0, 0);
    M3D t_var = vm.state_.cov.block<3, 3>(3, 3);
    for (size_t i = 0; i < vm.feats_down_body_->size(); i++)
    {
      pointWithVar &pv = vm.pv_list_[i];
      pv.point_b << vm.feats_down_body_->points[i].x, vm.feats_down_body_->points[i].y,
          vm.feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
      M3D cov = vm.body_cov_list_[i];
      M3D point_crossmat = vm.cross_mat_list_[i];
      cov = vm.state_.rot_end * cov * vm.state_.rot_end.transpose() +
            (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = vm.body_cov_list_[i];
    }
    vm.ptpl_list_.clear();
    vm.BuildResidualListOMP(vm.pv_list_, vm.ptpl_list_);
    for (size_t i = 0; i < vm.ptpl_list_.size(); i++) total_residual += fabs(vm.ptpl_list_[i].dis_to_plane_);
    vm.effct_feat_num_ = vm.ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << vm.feats_undistort_->size()
         << ", downsampled feature num:" << vm.feats_down_size_
         << " effective feature num: " << vm.effct_feat_num_
         << " average residual: " << total_residual / vm.effct_feat_num_ << endl;

    MatrixXd Hsub(vm.effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, vm.effct_feat_num_);
    VectorXd R_inv(vm.effct_feat_num_);
    VectorXd meas_vec(vm.effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < vm.effct_feat_num_; i++)
    {
      const PointToPlane &ptpl = vm.ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = vm.extR_ * point_this + vm.extT_;
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);
      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl.center_;
      J_nq.block<1, 3>(0, 3) = -ptpl.normal_;
      M3D var = state_propagat.rot_end * vm.extR_ * ptpl.body_cov_ *
                (state_propagat.rot_end * vm.extR_).transpose();
      double sigma_l = J_nq * ptpl.plane_var_ * J_nq.transpose();
      R_inv(i) = 1.0 / (meas_var_scale_ * (0.001 + sigma_l + ptpl.normal_.transpose() * var * ptpl.normal_));
      V3D A(point_crossmat * vm.state_.rot_end.transpose() * ptpl.normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl.normal_[0], ptpl.normal_[1], ptpl.normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
          ptpl.normal_[0] * R_inv(i), ptpl.normal_[1] * R_inv(i), ptpl.normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl.dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;

    dtf::Mat6 HTH = Hsub_T_R_inv * Hsub;
    dtf::Vec6 HTz = Hsub_T_R_inv * meas_vec;
    dr = dtf::reduce(HTH, HTz, dtf_mode_, dtf_thresh_);
    H_T_H.block<6, 6>(0, 0) = HTH;
    MD(DIM_STATE, DIM_STATE) K_1 = (H_T_H + vm.state_.cov.inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - vm.state_;
    VD(DIM_STATE) solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) -
                             G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    vm.state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = vm.state_.rot_end.eulerAngles(2, 1, 0);

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (vm.config_setting_.max_iterations_ - 2))))
      rematch_num++;
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == vm.config_setting_.max_iterations_ - 1)))
    {
      vm.state_.cov = (I_STATE - G) * vm.state_.cov;
      vm.position_last_ = vm.state_.pos_end;
      vm.geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // Optional: forget position knowledge faster along the removed directions.
  if (dtf_inflate_ > 1.0 && dr.kept_trans < 3)
  {
    MD(DIM_STATE, DIM_STATE) T = I_STATE;
    T.block<3, 3>(3, 3) += (std::sqrt(dtf_inflate_) - 1.0) * dr.weak_trans;
    vm.state_.cov = T * vm.state_.cov * T.transpose();
  }

  // Diagnostics used by the VIO frame selector and the logs.
  last_sigma_min_ = dr.ratio_trans;
  is_degenerate_  = dr.kept_trans < 3 || dr.kept_rot < 3;
  oc_observable_dims_ = dr.kept_trans + dr.kept_rot;
  if (dr.kept_trans < 3)
    degen_dir_ = Eigen::SelfAdjointEigenSolver<M3D>(dr.weak_trans).eigenvectors().col(2);
  ROS_INFO("[DTF] t=%.2f ratio_trans=%.5f ratio_rot=%.5f kept=%d+%d",
           DepthMeasures.last_lio_update_time - _first_depth_time, dr.ratio_trans, dr.ratio_rot,
           dr.kept_rot, dr.kept_trans);
  // E11 diagnostics: the correction applied by this depth update (updated minus propagated state)
  printf("[UPD] t %.3f n %d dth %.4e dp %.4e dv %.4e dbg %.4e dba %.4e\n",
         DepthMeasures.last_lio_update_time - _first_depth_time, vm.effct_feat_num_,
         Eigen::AngleAxisd(state_propagat.rot_end.transpose() * vm.state_.rot_end).angle(),
         (vm.state_.pos_end - state_propagat.pos_end).norm(), (vm.state_.vel_end - state_propagat.vel_end).norm(),
         (vm.state_.bias_g - state_propagat.bias_g).norm(), (vm.state_.bias_a - state_propagat.bias_a).norm());
}

void RGBDMapper::stateEstimationAndMapping()
{
  switch (DepthMeasures.lio_vio_flg)
  {
  case VIO:
  {
    // Sprint 3 #11: Merge long-term visual map points into _pv_list so VIO
    // can initialise features in regions the local voxel map has slid away from.
    mergeLVMapToCurrentFrame();

    // Sprint 2 #10: Degeneration-aware adaptive VIO frame selector.
    // When LIO is well-constrained (σ_min > skip_vio_sigma_thresh_) and the
    // pose trajectory is not near-linear (linearity < 0.7), skip every other
    // VIO frame — saves ~30 % CPU/power on Jetson Orin NX.
    // When LIO is degenerate, always run VIO for compensation.
    const bool lio_well_constrained =
        (!is_degenerate_ && last_sigma_min_ > skip_vio_sigma_thresh_ &&
         linearity_score_ < 0.7);

    if (lio_well_constrained && (++vio_skip_counter_ % 2 == 0))
    {
      ROS_DEBUG("[VIO] Frame skipped (σ_min=%.3f, ℓ=%.2f) — LIO well-constrained.",
                last_sigma_min_, linearity_score_);
      break;
    }
    handleVIO();
    break;
  }
  case LIO:
  case LO:
    if (lio_en)
      handleLIO();        // full depth-as-LiDAR: ICP + voxel map + EKF
    else
      handleDepthInit();  // depth-aided VIO: normals only, no ICP
    break;
  default:
    break;
  }
}

// ============================================================================
//  VIO step (unchanged from FAST-LIVO2 logic)
// ============================================================================

void RGBDMapper::handleVIO()
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20)
           << DepthMeasures.last_lio_update_time - _first_depth_time << " "
           << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " "
           << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " "
           << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;

  // As in FAST-LIVO2 the update uses _pv_list and the voxel map of the last
  // depth frame; pcl_w_wait_pub only feeds the published cloud.
  if (!lidar_map_inited || _pv_list.empty())
  {
    ROS_WARN("[VIO] No depth point cloud available yet.");
    return;
  }

  ROS_INFO("[VIO] Raw feature num: %zu", _pv_list.size());

  vio_manager->plot_flag =
      (std::fabs((DepthMeasures.last_lio_update_time - _first_depth_time) - plot_time)
       < (frame_cnt / 2 * 0.1));

  // Sprint 1 #6: Apply adaptive covariance scaling.
  // vio_cov_scale_ is driven by the pose linearity score (updateLinearityScore):
  // 1× for rich 3-D trajectory, up to 3× for near-straight-line flight where
  // depth-scale observability is reduced and VIO patches are less reliable.
  vio_manager->img_point_cov = IMG_POINT_COV * vio_cov_scale_;

  vio_manager->processFrame(
      DepthMeasures.measures.back().img,
      _pv_list,
      voxelmap_manager->voxel_map_,
      DepthMeasures.last_lio_update_time - _first_depth_time);

  if (imu_prop_enable)
  {
    ekf_finish_once  = true;
    latest_ekf_state = _state;
    latest_ekf_time  = DepthMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out  << std::setw(20)
            << DepthMeasures.last_lio_update_time - _first_depth_time << " "
            << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " "
            << _state.vel_end.transpose() << " "
            << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " "
            << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
            << feats_undistort->points.size() << std::endl;
}

// ============================================================================
//  Depth-aided VIO initialisation  (lio_en = false)
//
//  Replaces the ICP-based LIO step.  Instead of running scan-to-map ICP,
//  we directly convert the (already IMU-undistorted) depth point cloud into
//  a vector<pointWithVar> with:
//    • 3-D world-frame positions  (from depth + current IMU-propagated pose)
//    • Per-point ray-direction normals   ← the key addition that allows
//      generateVisualMapPoints() to initialise new visual features
//    • Per-point covariances based on the D455 depth noise model
//
//  Why ray-direction normals work:
//    VIOManager::generateVisualMapPoints() skips every point whose normal is
//    zero (line 811 of vio.cpp).  Using the inward ray direction ( =-p_b / |p_b| )
//    as an approximate surface normal ensures:
//      • No points are skipped    →  VIO can initialise new features
//      • The cos-θ grazing check  →  (commented out in vio.cpp, so irrelevant)
//      • Patch warp quality       →  approximate, but VIO refines it later
//    The true surface normals are updated by updateReferencePatch() after the
//    first few frames once sufficient parallax has been observed.
//
//  The EKF state is NOT updated here (no ICP residuals).
//  The voxel map is NOT written (remains empty).
//  State is updated ONLY by the subsequent VIO photometric residuals + IMU.
// ============================================================================

void RGBDMapper::handleDepthInit()
{
  if (!feats_undistort || feats_undistort->empty())
  {
    ROS_WARN("[DepthInit] No depth points after IMU undistortion.");
    return;
  }

  _pv_list.clear();
  _pv_list.reserve(feats_undistort->size());

  // Combined rotation: depth-camera body → world
  // (extR maps depth body → IMU body; rot_end maps IMU body → world)
  const M3D R_world_depth = _state.rot_end * extR;

  for (const auto &pt : feats_undistort->points)
  {
    const double x = pt.x, y = pt.y, z = pt.z;
    if (z <= 0.0) continue;  // invalid depth point

    pointWithVar pv;

    // ---- Positions in each frame ----------------------------------------
    pv.point_b = V3D(x, y, z);                          // depth-camera frame
    pv.point_i = extR * pv.point_b + extT;               // IMU frame
    pv.point_w = _state.rot_end * pv.point_i + _state.pos_end; // world frame

    // ---- Skew-symmetric matrix of world-frame position ------------------
    //  Used in the EKF Jacobian (LIO); harmless to fill for completeness.
    const V3D &pw = pv.point_w;
    pv.point_crossmat <<  0,     -pw[2],  pw[1],
                          pw[2],  0,     -pw[0],
                         -pw[1],  pw[0],  0;

    // ---- D455 depth noise model -----------------------------------------
    //   Axial (range) noise:   σ_z  ≈  noise_frac × Z  (D455: ~1 % of range)
    //   Lateral noise:         σ_xy ≈  σ_z × tan(pixel_angle) + lateral_floor
    //                                ≈  0.5 σ_z  + lateral_noise_m
    const double sigma_z  = depth_noise_frac * z;
    const double sigma_xy = 0.5 * sigma_z + lateral_noise_m;

    pv.body_var  = Eigen::Matrix3d::Zero();
    pv.body_var(0, 0) = sigma_xy * sigma_xy;
    pv.body_var(1, 1) = sigma_xy * sigma_xy;
    pv.body_var(2, 2) = sigma_z  * sigma_z;

    // var_nostate = body_var rotated to world frame
    pv.var_nostate = R_world_depth * pv.body_var * R_world_depth.transpose();

    // Full world-frame variance = sensor noise + position state covariance
    pv.var = pv.var_nostate + _state.cov.block<3, 3>(3, 3);

    // ---- Approximate surface normal  (ray direction pointing at camera) --
    //
    //  generateVisualMapPoints() (vio.cpp line 811) skips points whose
    //  normal is zero.  We therefore MUST supply a non-zero normal.
    //
    //  Best first-order approximation: the inward ray direction.
    //  For a point p_b in depth-camera frame, the ray from camera to point
    //  is p_b/|p_b|, so the surface faces the camera at -p_b/|p_b|.
    //
    //  This is exact for surfaces perpendicular to the ray (floor, walls)
    //  and a reasonable approximation elsewhere.  VIO refines the normal
    //  after sufficient inter-frame parallax via updateReferencePatch().
    const V3D n_depth = -pv.point_b.normalized();  // inward ray (depth frame)
    pv.normal = R_world_depth * n_depth;             // rotate to world frame

    _pv_list.push_back(pv);
  }

  // ---- Build pcl_w_wait_pub for RViz visualisation ----------------------
  pcl_w_wait_pub->clear();
  pcl_w_wait_pub->reserve(_pv_list.size());
  for (const auto &pv : _pv_list)
  {
    PointType p;
    p.x = pv.point_w[0];
    p.y = pv.point_w[1];
    p.z = pv.point_w[2];
    pcl_w_wait_pub->push_back(p);
  }

  ROS_INFO("[DepthInit] %zu depth-aided VIO points built (no ICP, normals = ray dirs)",
           _pv_list.size());
}

// ============================================================================
//  LIO step — same as original handleLIO() but with depth cloud as input
// ============================================================================

void RGBDMapper::handleLIO()
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20)
           << DepthMeasures.last_lio_update_time - _first_depth_time << " "
           << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " "
           << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " "
           << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;

  if (!feats_undistort || feats_undistort->empty())
  {
    ROS_WARN("[LIO] No depth points after IMU undistortion.");
    return;
  }

  double t0 = omp_get_wtime();

  // Voxel-grid downsampling (depth clouds are dense: ~20k pts per frame)
  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_  = feats_down_body;
  transformDepthCloud(_state.rot_end, _state.pos_end,
                      feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_  = feats_down_size;

  ROS_INFO_THROTTLE(2.0, "[LIO] raw=%zu  ds=%d  state=(%.3f,%.3f,%.3f)",
                    feats_undistort->size(), feats_down_size,
                    _state.pos_end[0], _state.pos_end[1], _state.pos_end[2]);

  // Build voxel map on first depth frame
  if (!lidar_map_inited)
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
    ROS_INFO("[LIO] Voxel map initialised: %zu voxels, %d downsampled points.",
             voxelmap_manager->voxel_map_.size(), feats_down_size);
  }

  double t1 = omp_get_wtime();

  // ICP state estimation against voxel map
  if (dtf_ours_)
    dtfStateEstimation(state_propagat);
  else
    voxelmap_manager->StateEstimation(state_propagat);
  _state   = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;
  {
    // Same statistic as the photometric gate (vio.cpp), logged for the depth update.
    Eigen::Matrix<double, 6, 1> d6 = (_state - state_propagat).head<6>();
    Eigen::Matrix<double, 6, 6> P6 = state_propagat.cov.topLeftCorner<6, 6>();
    printf("[LIOGATE] t %.3f d2 %.3f drot %.5f dpos %.5f srot %.5f spos %.5f\n",
           DepthMeasures.last_lio_update_time - _first_depth_time, d6.dot(P6.ldlt().solve(d6)),
           d6.head<3>().norm(), d6.tail<3>().norm(), std::sqrt(P6.topLeftCorner<3, 3>().trace()),
           std::sqrt(P6.bottomRightCorner<3, 3>().trace()));
  }

  // Degeneracy detection: use ITPS FIM when available, else ICP normals
  // (the dtf modes handle degeneracy inside the update; lio/oc_after_plain projects after the plain one)
  if (!dtf_ours_ || (oc_after_plain_ && dtf_mode_ == dtf::Mode::Plain))
  {
    if (depth_proc->info_select_en && last_selection_info_.n_candidates > 0)   // 0 = depth-bin fallback
      applyITPSDegeneracy(last_selection_info_);
    else
      computeDegeneracy(_pv_list);
  }
  updateLinearityScore();
  if (staged_imu_ && imu_en) stagedAfterLIO();

  double t2 = omp_get_wtime();

  if (imu_prop_enable)
  {
    ekf_finish_once  = true;
    latest_ekf_state = _state;
    latest_ekf_time  = DepthMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // Pose output for EVO evaluation
  if (pose_output_en)
  {
    static bool opened = false;
    std::ofstream evoFile;
    evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt",
                 opened ? std::ios::app : std::ios::out);
    opened = true;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed
            << DepthMeasures.last_lio_update_time << " "
            << _state.pos_end[0] << " " << _state.pos_end[1] << " "
            << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << std::endl;
    if (cov_output_en)
    {
      static bool cov_opened = false;
      std::ofstream covFile(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".cov",
                            cov_opened ? std::ios::app : std::ios::out);
      cov_opened = true;
      const Eigen::Matrix3d Pp = _state.cov.block<3, 3>(3, 3);   // position block, world frame
      const Eigen::Matrix3d Pr = _state.cov.block<3, 3>(0, 0);   // rotation block, body-frame error
      const Eigen::Matrix3d Prp = _state.cov.block<3, 3>(0, 3);  // rotation-position cross block
      covFile << std::fixed << DepthMeasures.last_lio_update_time << std::scientific << std::setprecision(9)
              << " " << Pp(0, 0) << " " << Pp(0, 1) << " " << Pp(0, 2)
              << " " << Pp(1, 1) << " " << Pp(1, 2) << " " << Pp(2, 2)
              << " " << Pr(0, 0) << " " << Pr(0, 1) << " " << Pr(0, 2)
              << " " << Pr(1, 1) << " " << Pr(1, 2) << " " << Pr(2, 2);
      for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k) covFile << " " << Prp(r, k);
      covFile << std::endl;
    }
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat   = tf::createQuaternionMsgFromRollPitchYaw(
      euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  // Transform down-sampled depth cloud to world frame for map update
  PointCloudXYZI::Ptr world_depth(new PointCloudXYZI());
  transformDepthCloud(_state.rot_end, _state.pos_end,
                      feats_down_body, world_depth);
  for (size_t i = 0; i < world_depth->points.size(); i++)
  {
    voxelmap_manager->pv_list_[i].point_w
        << world_depth->points[i].x,
           world_depth->points[i].y,
           world_depth->points[i].z;

    M3D cross = voxelmap_manager->cross_mat_list_[i];
    M3D var   = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose()
        + (-cross) * _state.cov.block<3,3>(0,0) * (-cross).transpose()
        + _state.cov.block<3,3>(3,3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  _pv_list = voxelmap_manager->pv_list_;

  double t4 = omp_get_wtime();

  if (voxelmap_manager->config_setting_.map_sliding_en)
    voxelmap_manager->mapSliding();

  // Sprint 3 #11: Snapshot current depth points into the long-term visual map
  updateLVMap();

  // Publish coloured depth cloud
  PointCloudXYZI::Ptr srcCloud =
      dense_map_en ? feats_undistort : feats_down_body;
  int sz = srcCloud->points.size();
  PointCloudXYZI::Ptr worldCloud(new PointCloudXYZI(sz, 1));
  for (int i = 0; i < sz; i++)
    RGBpointBodyToWorld(&srcCloud->points[i], &worldCloud->points[i]);
  *pcl_w_wait_pub = *worldCloud;

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);  // else after VIO
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num
                  + (t4 - t0) / frame_num;

  printf("\033[1;34m+------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|               RGB-D LIO Mapping Time                       |\033[0m\n");
  printf("\033[1;34m+------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-28s | %-27f |\033[0m\n", "DownSample",        t1 - t0);
  printf("\033[1;36m| %-28s | %-27f |\033[0m\n", "ICP",               t2 - t1);
  printf("\033[1;36m| %-28s | %-27f |\033[0m\n", "UpdateVoxelMap",    t4 - t3);
  printf("\033[1;36m| %-28s | %-27f |\033[0m\n", "Total (current)",   t4 - t0);
  printf("\033[1;36m| %-28s | %-27f |\033[0m\n", "Total (average)",   aver_time_consu);
  printf("\033[1;34m+------------------------------------------------------------+\033[0m\n");

  fout_out << std::setw(20)
           << DepthMeasures.last_lio_update_time - _first_depth_time << " "
           << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " "
           << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " "
           << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
           << feats_undistort->points.size() << std::endl;
}

// ============================================================================
//  Sprint 1 #9 — Pose linearity check
//
//  Keeps a sliding window of world-frame positions.  PCA on the translation
//  history yields a linearity score:
//    ℓ = 1 − (λ₂+λ₃)/λ₁
//  where λ₁≥λ₂≥λ₃ are eigenvalues of the position covariance matrix.
//  ℓ→1 means near-linear motion (straight crop row / corridor).
//  When ℓ>0.85 the depth scale is poorly observable from the pose history
//  → warn and inflate VIO covariance; when ℓ<0.5 reset to baseline.
// ============================================================================

void RGBDMapper::updateLinearityScore()
{
  if (linearity_window_ <= 0)          // off: no linearity-based VIO covariance scaling
  {
    linearity_score_ = 0.0;
    vio_cov_scale_   = 1.0;
    return;
  }
  pose_history_.push_back(_state.pos_end);
  if ((int)pose_history_.size() > linearity_window_)
    pose_history_.pop_front();

  if ((int)pose_history_.size() < 5) { linearity_score_ = 0.0; return; }

  // Compute centred covariance of positions
  V3D mean = V3D::Zero();
  for (const auto &p : pose_history_) mean += p;
  mean /= static_cast<double>(pose_history_.size());

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto &p : pose_history_)
  {
    V3D d = p - mean;
    cov  += d * d.transpose();
  }
  cov /= static_cast<double>(pose_history_.size());

  // Eigenvalues (ascending order from SelfAdjointEigenSolver)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
  const auto &ev = eig.eigenvalues();          // ev[0] ≤ ev[1] ≤ ev[2]
  double lambda1 = ev[2], lambda2 = ev[1], lambda3 = ev[0];

  if (lambda1 > 1e-6)
    linearity_score_ = 1.0 - (lambda2 + lambda3) / lambda1;
  else
    linearity_score_ = 0.0;

  linearity_score_ = std::max(0.0, std::min(1.0, linearity_score_));

  // Scale VIO covariance: near-linear → less parallax → depth less observable
  // Scale range: 1× (rich trajectory) to 3× (perfectly straight)
  vio_cov_scale_ = 1.0 + 2.0 * linearity_score_;

  if (linearity_score_ > 0.85)
    ROS_WARN_THROTTLE(2.0,
      "[Linearity] Near-linear motion (ℓ=%.2f) — depth scale may drift; "
      "VIO cov scaled ×%.1f", linearity_score_, vio_cov_scale_);
}

// ============================================================================
//  Observability-Constrained EKF (OC-EKF) — ITPS path
//
//  Given the ITPS translational FIM F (3×3, camera frame):
//
//    1. Transform F to world frame:  F_w = R · F · R^T   where R = rot_end · extR
//    2. Eigendecompose F_w = V Λ V^T
//    3. Classify each eigenvector as observable (λ_i/λ_max ≥ thresh) or not
//    4. Build the 6×6 observability projection Π₆ for (rotation, translation):
//         - Rotation: assumed fully observable from ICP (3 DOFs from normals)
//         - Translation: project to span of observable eigenvectors only
//    5. Embed Π₆ into DIM_STATE×DIM_STATE as Π (identity for vel/bias/gravity)
//    6. Apply constrained update:
//         δx ← Π · (x_ekf - x_prop) + (x_prop - x_pre_icp)
//         P  ← Π · P_ekf · Π^T + (I-Π) · P_prop · (I-Π)^T
//
//  This is equivalent to a Kalman gain that is zero along unobservable
//  directions — the optimal estimator under partial observability.
//
//  Theorem (optimality):
//    The OC-EKF update minimises E[‖x - x_true‖²_W] subject to the constraint
//    that the update lies in the column space of the observability Gramian.
//    See Zhang et al. ICRA 2016 §III-B; Tagliabue et al. RAL 2021 §IV.
// ============================================================================

// E11 diagnostics: the two covariances a projected update can store (the Joseph form of the projected gain and the
// splice of our earlier system), and one log line per projected update with the discarded correction and the
// difference between the two forms.  Both paths below store the configured one, as before.
typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatState;

static void projectedCovariances(const MatState &Pm, const MatState &Pp, const MatState &Pi, MatState &PL, MatState &PS)
{
  const MatState I_minus_Pi = MatState::Identity() - Pi;
  // Joseph form with L = Pi K, written without H:
  //   I - K H = P+ (P-)^-1 =: M,   K R K^T = P+ - M P+,
  //   P_L = (I - Pi K H) P- (I - Pi K H)^T + Pi K R K^T Pi^T,  I - Pi K H = (I - Pi) + Pi M.
  const MatState M = Pm.ldlt().solve(Pp).transpose();   // P+ (P-)^-1
  const MatState A = I_minus_Pi + Pi * M;
  PL = A * Pm * A.transpose() + Pi * (Pp - M * Pp) * Pi.transpose();
  PL = 0.5 * (PL + PL.transpose());
  PS = Pi * Pp * Pi.transpose() + I_minus_Pi * Pm * I_minus_Pi.transpose();
}

static void logCovDiag(const char *path, double t, int n_obs, double isotropy, const V3D &dp, const V3D &dropped,
                       const StatesGroup &post, const StatesGroup &prop, const MatState &PL, const MatState &PS)
{
  MatState D = PS - PL;
  D = 0.5 * (D + D.transpose());
  Eigen::SelfAdjointEigenSolver<MatState> es(D);
  printf("[COVDIAG] t %.3f path %s obs %d iso %.5f dp %.4e dropped %.4e dv %.4e dbg %.4e dba %.4e "
         "dPF %.4e dPmin %.4e dPmax %.4e PLF %.4e\n",
         t, path, n_obs, isotropy, dp.norm(), dropped.norm(), (post.vel_end - prop.vel_end).norm(),
         (post.bias_g - prop.bias_g).norm(), (post.bias_a - prop.bias_a).norm(), D.norm(),
         es.eigenvalues().minCoeff(), es.eigenvalues().maxCoeff(), PL.norm());
}

void RGBDMapper::applyConstrainedEKF_ITPS(const SelectionInfo &info,
                                           const StatesGroup &state_prop)
{
  // ── 1. Transform FIM from camera frame to world frame ──────────────────
  M3D R_wc = _state.rot_end * extR;  // camera → world rotation
  Eigen::Matrix3d F_world = R_wc * info.fim * R_wc.transpose();

  // ── 2. Eigendecompose ──────────────────────────────────────────────────
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(F_world);
  Eigen::Vector3d eigenvalues  = es.eigenvalues();   // ascending order
  Eigen::Matrix3d eigenvectors = es.eigenvectors();

  double lambda_max = eigenvalues.maxCoeff();
  if (lambda_max < 1e-12)
  {
    // No information at all — revert to propagated state entirely
    is_degenerate_ = true;
    ++consecutive_degenerate_frames_;
    _state = state_prop;
    ROS_WARN_THROTTLE(1.0, "[OC-EKF] Fully degenerate (no FIM info), reverting to IMU.");
    return;
  }

  // ── 3. Classify observable/unobservable translation DOFs ───────────────
  //
  // Adaptive threshold: interpolate between oc_lambda_lo_ (well-conditioned,
  // accept more DOFs) and oc_lambda_hi_ (ill-conditioned, reject aggressively)
  // using a sigmoid on the condition number κ = λ_max / λ_min.
  //
  // This fixes the RPE spike on desk-scale scenes (fr2/desk) where the fixed
  // threshold (0.05) was too aggressive — clipping a marginally observable
  // direction that carried useful local frame-to-frame information.
  //
  //   thresh(κ) = lo + (hi - lo) · σ( (κ - κ_knee) / (0.25·κ_knee) )
  //   where σ(x) = 1/(1+e^{-x}) is the logistic sigmoid
  //
  double eff_thresh = oc_lambda_thresh_;  // fallback: fixed threshold
  if (oc_adaptive_thresh_en_)
  {
    double kappa = (eigenvalues[0] > 1e-12)
                 ? lambda_max / eigenvalues[0] : 1e6;
    double x = (kappa - oc_adapt_kappa_knee_) / (0.25 * oc_adapt_kappa_knee_);
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    eff_thresh = oc_lambda_lo_ + (oc_lambda_hi_ - oc_lambda_lo_) * sigmoid;
  }

  // Construct translation projection matrix Π_t (3×3):
  //   Π_t = Σ_{i: λ_i/λ_max ≥ thresh} v_i v_i^T
  //
  Eigen::Matrix3d Pi_trans = Eigen::Matrix3d::Zero();
  int n_obs = 0;
  for (int i = 0; i < 3; ++i)
  {
    double ratio = eigenvalues[i] / lambda_max;
    if (ratio >= eff_thresh)
    {
      Pi_trans += eigenvectors.col(i) * eigenvectors.col(i).transpose();
      ++n_obs;
    }
  }
  oc_observable_dims_ = n_obs + 3;  // 3 rotation + n_obs translation

  // Update degeneracy diagnostics (with oc_sched_from_ratio the VIO scheduling signal set by the plain
  // depth update, rho_t, is kept so that the projection ablation differs only in the projection)
  double isotropy = (lambda_max > 1e-12) ? eigenvalues[0] / lambda_max : 0.0;
  const bool degenerate_now = (n_obs < 3);
  printf("[OCPROJ] t %.3f path itps obs %d ratio %.5f thresh %.4f\n",
         DepthMeasures.last_lio_update_time - _first_depth_time, n_obs, isotropy, eff_thresh);
  const bool was_degenerate = oc_sched_from_ratio_ ? (consecutive_degenerate_frames_ > 0) : is_degenerate_;
  if (!oc_sched_from_ratio_)
  {
    last_sigma_min_ = isotropy;
    is_degenerate_ = degenerate_now;
  }
  degen_dir_      = R_wc * info.degen_dir;  // weakest direction in world frame

  if (degenerate_now)
  {
    ++consecutive_degenerate_frames_;
    ROS_WARN_THROTTLE(1.0,
        "[OC-EKF] Degenerate: %d/3 trans DOFs observable (ι=%.4f, κ=%.1f, "
        "consec=%d)",
        n_obs, isotropy, info.condition, consecutive_degenerate_frames_);
  }
  else
  {
    if (was_degenerate)
      ROS_INFO("[OC-EKF] Resolved: 3/3 DOFs observable (ι=%.4f)", isotropy);
    consecutive_degenerate_frames_ = 0;
  }

  // ── 3b. Temporal persistence guard ────────────────────────────────────
  //
  // When degeneracy persists for many consecutive frames (e.g. outdoor crop
  // rows, long corridors), the OC-EKF projection can accumulate error by
  // repeatedly discarding DOFs that carry partial information.  The guard
  // disables the projection (Π_t → I) after oc_persist_frames_ consecutive
  // degenerate frames, letting the full EKF update through.
  //
  if (oc_persist_guard_en_ && degenerate_now &&
      consecutive_degenerate_frames_ > oc_persist_frames_)
  {
    Pi_trans = Eigen::Matrix3d::Identity();
    n_obs = 3;
    ROS_WARN_THROTTLE(2.0,
        "[OC-EKF] Persistence guard: %d consec degenerate frames > %d, "
        "bypassing projection",
        consecutive_degenerate_frames_, oc_persist_frames_);
  }

  // ── 4. Build full-state projection Π ∈ R^{DIM_STATE × DIM_STATE} ──────
  //
  // State layout: [rot(3) | pos(3) | inv_expo(1) | vel(3) | bg(3) | ba(3) | g(3)]
  //               0..2     3..5     6             7..9     10..12  13..15   16..18
  //
  // ICP only observes rotation and translation (indices 0..5).
  // Rotation: fully observable → identity block (ICP constrains all 3 rot DOFs)
  // Translation: project using Π_t
  // All other states: identity (propagated by IMU, not touched by ICP)
  //
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> Pi_full =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();

  // Replace the translation block [3:5, 3:5] with the observability projection
  Pi_full.block<3, 3>(3, 3) = Pi_trans;

  // ── 5. Apply constrained state update ─────────────────────────────────
  //
  // The EKF has already updated _state. We decompose the update:
  //   δx_full = x_ekf - x_prop
  //   δx_constrained = Π · δx_full
  //   x_final = x_prop + δx_constrained
  //
  // Equivalently for position (the main target):
  //   pos_final = pos_prop + Π_t · (pos_ekf - pos_prop)
  //
  V3D dp = _state.pos_end - state_prop.pos_end;
  V3D dp_constrained = Pi_trans * dp;
  _state.pos_end = state_prop.pos_end + dp_constrained;

  // For rotation: keep the full ICP rotation update (all 3 DOFs observable)
  // For velocity/bias/gravity: keep unchanged (propagated by IMU)

  // ── 6. Constrained covariance update ──────────────────────────────────
  //
  //   P = Π · P_ekf · Π^T + (I - Π) · P_prop · (I - Π)^T
  //
  // Observable subspace: use EKF posterior covariance (tighter)
  // Unobservable subspace: preserve propagated covariance (honest)
  //
  MatState PL, PS;
  projectedCovariances(state_prop.cov, _state.cov, Pi_full, PL, PS);
  logCovDiag("itps", DepthMeasures.last_lio_update_time - _first_depth_time, n_obs, isotropy, dp, dp - dp_constrained,
             _state, state_prop, PL, PS);
  _state.cov = (oc_cov_mode_ == "joseph") ? PL : PS;

  // Optional: inflate covariance along unobservable translation directions
  // to signal uncertainty to the VIO layer
  if (degenerate_now && oc_cov_inflate_ > 1.0)
  {
    Eigen::Matrix3d inflate = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i)
    {
      if (eigenvalues[i] / lambda_max < eff_thresh)
      {
        inflate += (oc_cov_inflate_ - 1.0)
                 * eigenvectors.col(i) * eigenvectors.col(i).transpose();
      }
    }
    _state.cov.block<3, 3>(3, 3) =
        inflate * _state.cov.block<3, 3>(3, 3) * inflate.transpose();
  }

  ROS_DEBUG_THROTTLE(2.0, "[OC-EKF] κ=%.1f thresh=%.4f obs=%d/3",
      (eigenvalues[0] > 1e-12) ? lambda_max/eigenvalues[0] : 999.0,
      eff_thresh, n_obs);
}

// ============================================================================
//  Observability-Constrained EKF — ICP-normal fallback path
//
//  When ITPS is disabled, we build the translational FIM from ICP plane
//  normals post-hoc: F = Σ_i n_i n_i^T (world frame, already transformed).
//  Then apply the same observability projection as the ITPS path.
// ============================================================================

void RGBDMapper::applyConstrainedEKF_ICP(
    const std::vector<pointWithVar> &pv_list,
    const StatesGroup &state_prop)
{
  if (pv_list.empty()) { if (!oc_sched_from_ratio_) is_degenerate_ = false; return; }

  // Build FIM from ICP plane normals (already in world frame)
  Eigen::Matrix3d F_world = Eigen::Matrix3d::Zero();
  int valid = 0;
  for (const auto &pv : pv_list)
  {
    if (pv.normal.squaredNorm() < 1e-8) continue;
    V3D n = pv.normal.normalized();
    F_world += n * n.transpose();
    ++valid;
  }
  if (valid == 0) { if (!oc_sched_from_ratio_) is_degenerate_ = false; return; }

  // Wrap into a SelectionInfo for unified handling
  SelectionInfo synth_info;
  synth_info.fim = F_world;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(F_world);
  synth_info.lambda_min = es.eigenvalues()[0];
  synth_info.lambda_max = es.eigenvalues()[2];
  synth_info.degen_dir  = es.eigenvectors().col(0);
  synth_info.isotropy   = (synth_info.lambda_max > 1e-12)
                        ? synth_info.lambda_min / synth_info.lambda_max : 0.0;
  synth_info.condition  = (synth_info.lambda_min > 1e-12)
                        ? synth_info.lambda_max / synth_info.lambda_min : 1e9;

  // Reuse the ITPS constrained-EKF path.
  // Note: the FIM is already in world frame, so we temporarily set extR to
  // identity and rot_end to identity for the transform step.  Instead, we
  // call the projection logic directly.

  // ── Eigendecompose (already done above) ────────────────────────────────
  Eigen::Vector3d eigenvalues  = es.eigenvalues();
  Eigen::Matrix3d eigenvectors = es.eigenvectors();
  double lambda_max = eigenvalues.maxCoeff();

  if (lambda_max < 1e-12)
  {
    if (!oc_sched_from_ratio_) is_degenerate_ = true;
    ++consecutive_degenerate_frames_;
    _state = state_prop;
    return;
  }

  // Build translation projection (with adaptive threshold)
  double eff_thresh = oc_lambda_thresh_;
  if (oc_adaptive_thresh_en_)
  {
    double kappa = (eigenvalues[0] > 1e-12)
                 ? lambda_max / eigenvalues[0] : 1e6;
    double x = (kappa - oc_adapt_kappa_knee_) / (0.25 * oc_adapt_kappa_knee_);
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    eff_thresh = oc_lambda_lo_ + (oc_lambda_hi_ - oc_lambda_lo_) * sigmoid;
  }

  Eigen::Matrix3d Pi_trans = Eigen::Matrix3d::Zero();
  int n_obs = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (eigenvalues[i] / lambda_max >= eff_thresh)
    {
      Pi_trans += eigenvectors.col(i) * eigenvectors.col(i).transpose();
      ++n_obs;
    }
  }
  oc_observable_dims_ = n_obs + 3;
  degen_dir_      = eigenvectors.col(0);

  // Diagnostics (with oc_sched_from_ratio the VIO scheduling signal of the plain update is kept, as in the ITPS path)
  const double sigma_min = eigenvalues[0] / lambda_max;
  const bool degenerate_now = (n_obs < 3);
  const bool was_degenerate = oc_sched_from_ratio_ ? (consecutive_degenerate_frames_ > 0) : is_degenerate_;
  if (!oc_sched_from_ratio_)
  {
    last_sigma_min_ = sigma_min;
    is_degenerate_ = degenerate_now;
  }
  printf("[OCPROJ] t %.3f path icp obs %d ratio %.5f thresh %.4f\n",
         DepthMeasures.last_lio_update_time - _first_depth_time, n_obs, sigma_min, eff_thresh);
  if (degenerate_now)
    ++consecutive_degenerate_frames_;
  else
    consecutive_degenerate_frames_ = 0;

  if (degenerate_now)
    ROS_WARN_THROTTLE(1.0, "[OC-EKF-ICP] %d/3 trans DOFs observable (σ_min=%.4f, consec=%d)",
                      n_obs, sigma_min, consecutive_degenerate_frames_);
  else if (was_degenerate)
    ROS_INFO("[OC-EKF-ICP] Resolved (σ_min=%.4f)", sigma_min);

  // Temporal persistence guard (same logic as ITPS path)
  if (oc_persist_guard_en_ && degenerate_now &&
      consecutive_degenerate_frames_ > oc_persist_frames_)
  {
    Pi_trans = Eigen::Matrix3d::Identity();
    n_obs = 3;
    ROS_WARN_THROTTLE(2.0,
        "[OC-EKF-ICP] Persistence guard: %d consec > %d, bypassing projection",
        consecutive_degenerate_frames_, oc_persist_frames_);
  }

  // Apply constrained position update
  V3D dp = _state.pos_end - state_prop.pos_end;
  _state.pos_end = state_prop.pos_end + Pi_trans * dp;

  // Constrained covariance
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> Pi_full =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  Pi_full.block<3, 3>(3, 3) = Pi_trans;

  // Covariance as in the ITPS path (lio/oc_cov_mode, lio/oc_cov_inflate) with lio/oc_icp_cov_from_config;
  // otherwise the splice without inflation, as our earlier system stored on this path
  MatState PL, PS;
  projectedCovariances(state_prop.cov, _state.cov, Pi_full, PL, PS);
  logCovDiag("icp", DepthMeasures.last_lio_update_time - _first_depth_time, n_obs, sigma_min, dp, dp - Pi_trans * dp,
             _state, state_prop, PL, PS);
  _state.cov = (oc_icp_cov_from_config_ && oc_cov_mode_ == "joseph") ? PL : PS;

  if (oc_icp_cov_from_config_ && degenerate_now && oc_cov_inflate_ > 1.0)
  {
    Eigen::Matrix3d inflate = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i)
    {
      if (eigenvalues[i] / lambda_max < eff_thresh)
      {
        inflate += (oc_cov_inflate_ - 1.0)
                 * eigenvectors.col(i) * eigenvectors.col(i).transpose();
      }
    }
    _state.cov.block<3, 3>(3, 3) =
        inflate * _state.cov.block<3, 3>(3, 3) * inflate.transpose();
  }
}

// ============================================================================
//  Legacy wrappers — delegate to OC-EKF or fall back to Tikhonov suppression
// ============================================================================

void RGBDMapper::applyITPSDegeneracy(const SelectionInfo &info)
{
  if (oc_ekf_en_)
  {
    applyConstrainedEKF_ITPS(info, state_propagat);
    return;
  }

  // Legacy Tikhonov path (kept for ablation)
  V3D v_cam  = info.degen_dir;
  V3D v_world = _state.rot_end * extR * v_cam;

  last_sigma_min_ = info.isotropy;
  degen_dir_      = v_world;

  const bool was_degenerate = is_degenerate_;
  is_degenerate_ = (info.isotropy < deg_isotropy_thresh_);

  if (is_degenerate_)
  {
    ++consecutive_degenerate_frames_;
    if (consecutive_degenerate_frames_ > 5)
    {
      V3D dp   = _state.pos_end - state_propagat.pos_end;
      double proj = dp.dot(degen_dir_);
      _state.pos_end -= 0.8 * proj * degen_dir_;
    }
  }
  else
    consecutive_degenerate_frames_ = 0;
}

void RGBDMapper::computeDegeneracy(const std::vector<pointWithVar> &pv_list)
{
  if (oc_ekf_en_)
  {
    applyConstrainedEKF_ICP(pv_list, state_propagat);
    return;
  }

  // Legacy Tikhonov path
  if (pv_list.empty()) { is_degenerate_ = false; return; }

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  int valid = 0;
  for (const auto &pv : pv_list)
  {
    if (pv.normal.squaredNorm() < 1e-8) continue;
    V3D n = pv.normal.normalized();
    H    += n * n.transpose();
    ++valid;
  }
  if (valid == 0) { is_degenerate_ = false; return; }
  H /= static_cast<double>(valid);

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU);
  Eigen::Vector3d sv = svd.singularValues();
  if (sv[0] > 1e-10) sv /= sv[0];

  last_sigma_min_ = sv[2];
  degen_dir_      = svd.matrixU().col(2);

  const bool was_degenerate = is_degenerate_;
  is_degenerate_ = (last_sigma_min_ < deg_threshold_);

  if (is_degenerate_)
  {
    ++consecutive_degenerate_frames_;
    if (consecutive_degenerate_frames_ > 5)
    {
      V3D dp   = _state.pos_end - state_propagat.pos_end;
      double proj = dp.dot(degen_dir_);
      _state.pos_end -= 0.8 * proj * degen_dir_;
    }
  }
  else
    consecutive_degenerate_frames_ = 0;
}

// ============================================================================
//  Sprint 3 #11 — Long-term visual map: update
//
//  Called every lv_update_interval_ LIO frames.  Samples current _pv_list
//  at 1/10 density into a coarse spatial hash (cell size lv_voxel_size_).
//  Only the first point falling into each cell is stored, so the LV map is
//  sparse and geographically unique.  When the map exceeds lv_map_max_pts_
//  the oldest half is evicted (approximated by clearing and re-seeding from
//  the current frame — keeps the map anchored to the live neighbourhood).
// ============================================================================

void RGBDMapper::updateLVMap()
{
  if (++lv_frame_counter_ % lv_update_interval_ != 0) return;
  if (_pv_list.empty()) return;

  const double inv_vs = 1.0 / lv_voxel_size_;

  // Sample _pv_list at 1-in-10 and insert into the spatial hash
  for (int i = 0; i < (int)_pv_list.size(); i += 10)
  {
    const V3D &pw = _pv_list[i].point_w;
    int vx = static_cast<int>(std::floor(pw[0] * inv_vs));
    int vy = static_cast<int>(std::floor(pw[1] * inv_vs));
    int vz = static_cast<int>(std::floor(pw[2] * inv_vs));

    // Mix 16-bit integers into a 64-bit key (±32 767 cells per axis = ±65 km at 2 m)
    std::size_t key =
        (std::size_t)((vx + 32768) & 0xFFFF) << 32 |
        (std::size_t)((vy + 32768) & 0xFFFF) << 16 |
        (std::size_t)((vz + 32768) & 0xFFFF);

    lv_map_.emplace(key, _pv_list[i]); // emplace: no-op if key already exists
  }

  // Evict: if over limit, clear and re-seed from current frame only
  if ((int)lv_map_.size() > lv_map_max_pts_)
  {
    ROS_INFO("[LVMap] Size %zu > %d — evicting; re-seeding from current frame.",
             lv_map_.size(), lv_map_max_pts_);
    lv_map_.clear();
    for (int i = 0; i < (int)_pv_list.size(); i += 10)
    {
      const V3D &pw = _pv_list[i].point_w;
      int vx = static_cast<int>(std::floor(pw[0] * inv_vs));
      int vy = static_cast<int>(std::floor(pw[1] * inv_vs));
      int vz = static_cast<int>(std::floor(pw[2] * inv_vs));
      std::size_t key =
          (std::size_t)((vx + 32768) & 0xFFFF) << 32 |
          (std::size_t)((vy + 32768) & 0xFFFF) << 16 |
          (std::size_t)((vz + 32768) & 0xFFFF);
      lv_map_.emplace(key, _pv_list[i]);
    }
  }

  ROS_DEBUG("[LVMap] Updated: %zu cells total.", lv_map_.size());
}

// ============================================================================
//  Sprint 3 #11 — Long-term visual map: merge into current frame
//
//  Before VIO, finds LV map points that are within the depth-camera frustum
//  (distance ≤ 3×max_depth, cosine to forward axis > 0.3) and appends them
//  to _pv_list.  This provides long-term visual anchors even after the local
//  voxel map has slid away from a previously visited region.
//
//  At most 2000 points are added per frame to keep VIO latency bounded.
// ============================================================================

void RGBDMapper::mergeLVMapToCurrentFrame()
{
  if (lv_map_.empty()) return;

  // Depth-camera forward axis in world frame:
  //   z-axis of depth camera body → rotate by extR → IMU body → rotate by rot_end → world
  const V3D forward_world = _state.rot_end * extR * V3D(0, 0, 1);
  const V3D &pos          = _state.pos_end;
  const double max_dist   = depth_proc->max_depth * 3.0; // generous frustum depth

  int added = 0;
  const int max_add = 2000;

  for (auto &kv : lv_map_)
  {
    const V3D &pw = kv.second.point_w;
    V3D  dp   = pw - pos;
    double d  = dp.norm();
    if (d < 0.1 || d > max_dist) continue;
    // Cosine of angle between dp and forward_world; points behind camera skipped
    if (dp.dot(forward_world) / d < 0.3) continue;

    _pv_list.push_back(kv.second);
    if (++added >= max_add) break;
  }

  if (added > 0)
    ROS_DEBUG("[LVMap] Merged %d long-term visual points into current frame.", added);
}

// ============================================================================
//  Main run loop
// ============================================================================

void RGBDMapper::run()
{
  ros::Rate rate(5000);
  while (ros::ok())
  {
    ros::spinOnce();
    if (!sync_packages(DepthMeasures))
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();
    processImu();
    stateEstimationAndMapping();
  }
  savePCD();
}

// ============================================================================
//  PCD saving
// ============================================================================

void RGBDMapper::savePCD()
{
  if (!pcd_save_en) return;
  if (pcl_wait_save->points.empty() &&
      pcl_wait_save_intensity->points.empty()) return;
  if (pcd_save_interval >= 0) return; // interval mode handled per-frame

  pcl::PCDWriter writer;
  if (img_en)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ds(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(pcl_wait_save);
    vg.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
    vg.filter(*ds);
    writer.writeBinary(std::string(ROOT_DIR) + "Log/pcd/all_points.pcd", *ds);
    ROS_INFO("[RGBDMapper] PCD saved: %zu points.", ds->size());
  }
  else
  {
    writer.writeBinary(std::string(ROOT_DIR) + "Log/pcd/all_points_intensity.pcd",
                       *pcl_wait_save_intensity);
  }
}

// ============================================================================
//  Point-cloud transform helpers
// ============================================================================

void RGBDMapper::transformDepthCloud(const Eigen::Matrix3d &rot,
                                      const Eigen::Vector3d &t,
                                      const PointCloudXYZI::Ptr &in,
                                      PointCloudXYZI::Ptr &out)
{
  PointCloudXYZI().swap(*out);
  out->reserve(in->size());
  for (const auto &p : in->points)
  {
    Eigen::Vector3d pt(p.x, p.y, p.z);
    pt = rot * (extR * pt + extT) + t;
    PointType po;
    po.x         = pt(0);
    po.y         = pt(1);
    po.z         = pt(2);
    po.intensity = p.intensity;
    out->push_back(po);
  }
}

void RGBDMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p(_state.rot_end * (extR * V3D(pi.x, pi.y, pi.z) + extT)
        + _state.pos_end);
  po.x = p(0); po.y = p(1); po.z = p(2);
  po.intensity = pi.intensity;
}

void RGBDMapper::RGBpointBodyToWorld(PointType const *pi, PointType *po)
{
  V3D p(_state.rot_end * (extR * V3D(pi->x, pi->y, pi->z) + extT)
        + _state.pos_end);
  po->x = p(0); po->y = p(1); po->z = p(2);
  po->intensity  = pi->intensity;
  po->normal_x   = pi->normal_x;
  po->normal_y   = pi->normal_y;
  po->normal_z   = pi->normal_z;
}

void RGBDMapper::RGBpointBodyDepthToIMU(PointType const *pi, PointType *po)
{
  V3D p(extR * V3D(pi->x, pi->y, pi->z) + extT);
  po->x = p(0); po->y = p(1); po->z = p(2);
}

// ============================================================================
//  Publishing helpers
// ============================================================================

template <typename T>
void RGBDMapper::set_posestamp(T &out)
{
  out.pose.position.x    = _state.pos_end(0);
  out.pose.position.y    = _state.pos_end(1);
  out.pose.position.z    = _state.pos_end(2);
  out.pose.orientation   = geoQuat;
}

void RGBDMapper::publish_odometry(const ros::Publisher &pub)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id  = "body";
  odomAftMapped.header.stamp    =
      ros::Time().fromSec(DepthMeasures.last_lio_update_time);
  set_posestamp(odomAftMapped.pose);
  pub.publish(odomAftMapped);
}

void RGBDMapper::publish_mavros(const ros::Publisher &pub)
{
  msg_body_pose.header.stamp    =
      ros::Time().fromSec(DepthMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose);
  pub.publish(msg_body_pose);
}

void RGBDMapper::publish_path(const ros::Publisher &pub)
{
  msg_body_pose.header.stamp    =
      ros::Time().fromSec(DepthMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose);
  path.poses.push_back(msg_body_pose);
  pub.publish(path);
}

void RGBDMapper::publish_frame_world(const ros::Publisher &pub, VIOManagerPtr vio)
{
  sensor_msgs::PointCloud2 laserCloudMsg;
  pcl::toROSMsg(*pcl_w_wait_pub, laserCloudMsg);
  laserCloudMsg.header.stamp    =
      ros::Time().fromSec(DepthMeasures.last_lio_update_time);
  laserCloudMsg.header.frame_id = "camera_init";
  pub.publish(laserCloudMsg);
  pcl_w_wait_pub->clear();
}

void RGBDMapper::publish_img_rgb(const image_transport::Publisher &pub,
                                  VIOManagerPtr vio)
{
  if (vio->img_rgb.empty()) return;
  sensor_msgs::ImagePtr msg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", vio->img_rgb).toImageMsg();
  msg->header.stamp =
      ros::Time().fromSec(DepthMeasures.last_lio_update_time);
  pub.publish(msg);
}

// ============================================================================
//  IMU propagation timer (high-rate odometry for flight controller)
// ============================================================================

void RGBDMapper::prop_imu_once(StatesGroup &s, double dt, V3D acc, V3D angvel)
{
  double norm = p_imu->IMU_mean_acc_norm;
  acc         = acc * G_m_s2 / norm - s.bias_a;
  angvel     -= s.bias_g;

  s.rot_end = s.rot_end * Exp(angvel, dt);
  V3D a_global = s.rot_end * acc
               + V3D(s.gravity[0], s.gravity[1], s.gravity[2]);
  s.pos_end = s.pos_end + s.vel_end * dt + 0.5 * a_global * dt * dt;
  s.vel_end = s.vel_end + a_global * dt;
}

void RGBDMapper::imu_prop_callback(const ros::TimerEvent &)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) return;

  mtx_buffer_imu_prop.lock();
  new_imu = false;

  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_dt = 0.0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      while (!prop_imu_buffer.empty() &&
             prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time)
        prop_imu_buffer.pop_front();
      last_dt = 0.0;
      for (const auto &imu : prop_imu_buffer)
      {
        double t = imu.header.stamp.toSec() - latest_ekf_time;
        double dt = t - last_dt;
        V3D a(imu.linear_acceleration.x, imu.linear_acceleration.y,
              imu.linear_acceleration.z);
        V3D w(imu.angular_velocity.x, imu.angular_velocity.y,
              imu.angular_velocity.z);
        prop_imu_once(imu_propagate, dt, a, w);
        last_dt = t;
      }
      state_update_flg = false;
    }
    else
    {
      double t  = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t - last_dt;
      V3D a(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y,
            newest_imu.linear_acceleration.z);
      V3D w(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y,
            newest_imu.angular_velocity.z);
      prop_imu_once(imu_propagate, dt, a, w);
      last_dt = t;
    }

    Eigen::Quaterniond q(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id        = "world";
    imu_prop_odom.header.stamp           = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x   = imu_propagate.pos_end.x();
    imu_prop_odom.pose.pose.position.y   = imu_propagate.pos_end.y();
    imu_prop_odom.pose.pose.position.z   = imu_propagate.pos_end.z();
    imu_prop_odom.pose.pose.orientation.w= q.w();
    imu_prop_odom.pose.pose.orientation.x= q.x();
    imu_prop_odom.pose.pose.orientation.y= q.y();
    imu_prop_odom.pose.pose.orientation.z= q.z();
    imu_prop_odom.twist.twist.linear.x   = imu_propagate.vel_end.x();
    imu_prop_odom.twist.twist.linear.y   = imu_propagate.vel_end.y();
    imu_prop_odom.twist.twist.linear.z   = imu_propagate.vel_end.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}
