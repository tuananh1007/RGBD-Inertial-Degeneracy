#!/bin/bash
# Download the TUM RGB-D archives used in the evaluation of the paper and convert them
# with convert_tum_rgbd.py (per-image stamps, synthetic IMU from GT).
#
# Usage: make_tum_bags.sh RAW_DIR OUT_DIR   (archives are kept in RAW_DIR;
#        sequences whose OUT_DIR/<seq>_d455.bag exists are skipped)
set -u
RAW=$1; OUT=$2
CONV=$(cd "$(dirname "$0")" && pwd)/convert_tum_rgbd.py
BASE=https://cvg.cit.tum.de/rgbd/dataset
mkdir -p $RAW $OUT
SEQS=(fr1_360:1:360 fr1_desk:1:desk fr1_desk2:1:desk2 fr1_floor:1:floor fr1_plant:1:plant fr1_room:1:room
      fr1_rpy:1:rpy fr1_xyz:1:xyz fr2_360_kidnap:2:360_kidnap fr2_coke:2:coke fr2_desk:2:desk
      fr2_flower:2:flowerbouquet fr2_flower_brown:2:flowerbouquet_brownbackground fr2_metallic2:2:metallic_sphere2
      fr2_pioneer_360:2:pioneer_360 fr2_pioneer_slam3:2:pioneer_slam3 fr2_rpy:2:rpy
      fr3_cabinet:3:cabinet fr3_large_cabinet:3:large_cabinet fr3_ns_nt_near:3:nostructure_notexture_near_withloop
      fr3_ns_t_far:3:nostructure_texture_far fr3_ns_t_near:3:nostructure_texture_near_withloop
      fr3_office:3:long_office_household fr3_s_nt_far:3:structure_notexture_far fr3_s_nt_near:3:structure_notexture_near
      fr3_s_t_far:3:structure_texture_far fr3_s_t_near:3:structure_texture_near fr3_sit_half:3:sitting_halfsphere
      fr3_sit_rpy:3:sitting_rpy fr3_sit_static:3:sitting_static fr3_sit_xyz:3:sitting_xyz fr3_teddy:3:teddy
      fr3_walk_half:3:walking_halfsphere fr3_walk_rpy:3:walking_rpy fr3_walk_static:3:walking_static fr3_walk_xyz:3:walking_xyz)
for e in "${SEQS[@]}"; do
  IFS=: read -r S F N <<< "$e"; D=rgbd_dataset_freiburg${F}_${N}
  [ -s $OUT/${S}_d455.bag ] && { echo "[skip] $S"; continue; }
  if [ ! -s $RAW/$D.tgz ]; then
    wget -q $BASE/freiburg$F/$D.tgz -O $RAW/$D.tgz.part && mv $RAW/$D.tgz.part $RAW/$D.tgz
  fi
  tar -xzf $RAW/$D.tgz -C $RAW && \
  python3 $CONV $RAW/$D --output $OUT/${S}_d455.bag --freiburg $F 2>&1 | tail -1
  rm -rf $RAW/$D
  echo "[tum] $S $(ls -la $OUT/${S}_d455.bag 2>/dev/null | awk '{print $5}')"
done
echo "[tum] all done"
