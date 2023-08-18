# #!/bin/bash

# Lidar2dKeypointFusion
python ~/master_thesis/3D_Pose_Estimation/main.py --suffix Fusion_3D_projections --wandb True --model_type Lidar2dKeypointFusion --config /lhome/petbau/master_thesis/3D_Pose_Estimation/configs/supervised_baselines/config_2d_projections_with_lidar.gin
python ~/master_thesis/3D_Pose_Estimation/main.py --suffix Fusion_2D_labels --wandb True --model_type Lidar2dKeypointFusion --config /lhome/petbau/master_thesis/3D_Pose_Estimation/configs/supervised_baselines/config_2d_labels_with_lidar.gin

# Lidar only
python ~/master_thesis/3D_Pose_Estimation/main.py --suffix Lidar --wandb True --model_type PointNet --config /lhome/petbau/master_thesis/3D_Pose_Estimation/configs/supervised_baselines/config_lidar.gin

# keypoints
python ~/master_thesis/3D_Pose_Estimation/main.py --suffix 2D_lables --wandb True --model_type SimpleLiftingModel --config /lhome/petbau/master_thesis/3D_Pose_Estimation/configs/supervised_baselines/config_2d_labels.gin
python ~/master_thesis/3D_Pose_Estimation/main.py --suffix 3D_projections --wandb True --model_type SimpleLiftingModel --config /lhome/petbau/master_thesis/3D_Pose_Estimation/configs/supervised_baselines/config_2d_projections.gin
