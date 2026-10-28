# Configurations

A run loads three files, in this order, each overriding the previous one (`rgbd_vio/launch/mapping_d455.launch`):

1. a **dataset configuration** (`BASE_CFG`): topics, depth range, IMU noise, camera-IMU extrinsics;
2. a **camera configuration** (`CAM_CFG`): intrinsics in the vikit format;
3. an **experiment configuration** (the variant passed to `scripts/run_sequences.sh`): the switches that define a method.

| Dataset | `BASE_CFG` | `CAM_CFG` |
|---|---|---|
| UT-MM | `utmm.yaml` | `camera_utmm.yaml` |
| OpenLORIS-Scene office, corridor, home, cafe | `openloris_tf.yaml` | `camera_openloris.yaml` |
| OpenLORIS-Scene market | `openloris_market_tf.yaml` | `camera_openloris_market.yaml` |
| FAST-LIVO2 `Retail_Street`, `CBD_Building_01` | `fastlivo2_retail.yaml` | `camera_fl2_retail.yaml` |
| FAST-LIVO2 `CBD_Building_02` | `fastlivo2_hku.yaml` | `camera_fl2_hku.yaml` |
| Rosario v2 | `rosario.yaml` | `camera_rosario.yaml` |
| ROVER | `rover.yaml` | `camera_rover.yaml` |
| TUM RGB-D | `tum_fr1.yaml`, `tum_fr2.yaml`, `tum_fr3.yaml` (chosen from the sequence name when unset) | `camera_tum_fr*.yaml` |
| ICL-NUIM simulation | `icl_nuim.yaml` | `camera_icl_nuim.yaml` |
| Field recordings, Site A | `field_site_a.yaml` | `camera_field_d455.yaml` |
| Field recordings, Site B | `field_site_b.yaml` | `camera_field_d455.yaml` |

## Methods of the paper

| Paper | Experiment configuration |
|---|---|
| **Ours** | `s_ours_lin.yaml` |
| Baseline (our earlier system) | `B.yaml` |
| Ours without the photometric update ("− VIO"; on TUM RGB-D: `s_P2novio.yaml`) | `s_ours_novio.yaml` |
| − ITPS selection (depth-stratified sampler); capped at 20,000 points | `lin_noitps.yaml`; `lin_noitps_eq.yaml` |
| + projection (projection, covariance splice, and inflation of the earlier system) | `s_lin_oc.yaml` |
| legacy IMU; with − ITPS; with − VIO | `s_lin_legacy.yaml`; `lin_legacy_noitps.yaml`; `s_lin_legacy_novio.yaml` |
| legacy initialization only; legacy noise only | `s_lin_legacyinit.yaml`; `s_lin_legacynoise.yaml` |
| ITPS with the row-order fill; uniform sample of 20,000 points | `ours_lin.yaml`; `u_ours_lin.yaml` |
| + projection (adaptive trigger) with Joseph form, splice, splice with $\alpha=10$ | `s_lin_proja_joseph.yaml`, `s_lin_proja_splice.yaml`, `s_lin_proja_full.yaml` |
| + projection at 0.10 with Joseph form, splice | `s_lin_proj10_joseph.yaml`, `s_lin_proj10_splice.yaml` |
| truncated information, 0.10 | `s_lin_info10.yaml` |
| − linearity variance; − frame scheduling | `s_ours.yaml`; `s_lin_noskip.yaml` |
| local photometric bundle adjustment (Supplementary S8) | `s_lin_lba.yaml` |

`u_*` repeats a variant on the uniform-sample base, and `P2*.yaml`, `ours.yaml`, and `ours_novio.yaml`
are the files the `s_*` and `u_*` configurations were derived from (each file's header says which).

## Controlled experiments

| Files | Experiment | Paper |
|---|---|---|
| `e11_*.yaml` | the projected update with Joseph-form, spliced, and inflated covariance at matched triggers | Section 5.5, Figure 4, Supplementary S2 |
| `e1_*.yaml` | inertial noise form and magnitudes, gravity random walk | Supplementary S10 |
| `e2_*.yaml` | point budget | Supplementary S10 |
| `e5_*.yaml`, `e10_*.yaml` | covariance output; scaled residual variances | Supplementary S10 |
| `e12_*.yaml` | ICL-NUIM consistency simulation | Supplementary S10 |
| `e7_*.yaml` | lateral-normal candidates removed (degeneracy injection) | Supplementary S9 |
| `e9_*.yaml` | raised depth range, with and without the linearity-dependent variance | Supplementary S11 |
| `gate_g*.yaml` | photometric consistency gate (`vio/gate_chi2`) | Supplementary S5 |

The header comments name the configuration each file derives from; some also cite development notes that are not part
of this release.
