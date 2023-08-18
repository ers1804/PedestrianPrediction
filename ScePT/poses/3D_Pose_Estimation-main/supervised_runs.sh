#!/bin/bash

# LIFTING
python main.py --wandb True --suffix  SimpleLiftingModel_2D_labels_supervised --dataset waymo_2d_labels_supervised --model_type SimpleLiftingModel --config ./configs/supervised_baselines/lifting_waymo_2d_labels_supervised.gin --wandb_project SUPERVISED_FINAL
python main.py --wandb True --suffix  SimpleLiftingModel_waymo_3d_2d_projections_supervised --dataset waymo_3d_2d_projections_supervised --model_type SimpleLiftingModel --config ./configs/supervised_baselines/lifting_waymo_3d_2d_projections_supervised.gin --wandb_project SUPERVISED_FINAL

# Point Net
python main.py --wandb True --suffix  PointNet --dataset waymo_2d_labels_supervised --model_type PointNet --config ./configs/supervised_baselines/point_net.gin --wandb_project SUPERVISED_FINAL

# FUSION
python main.py --wandb True --suffix  Lidar2dKeypointFusion_waymo_3d_2d_projections_supervised --dataset waymo_3d_2d_projections_supervised --model_type Lidar2dKeypointFusion --config ./configs/supervised_baselines/fusion_waymo_3d_2d_projections_supervised.gin --wandb_project SUPERVISED_FINAL
python main.py --wandb True --suffix  Lidar2dKeypointFusion_waymo_2d_labels_supervised --dataset waymo_2d_labels_supervised --model_type Lidar2dKeypointFusion --config ./configs/supervised_baselines/fusion_waymo_2d_labels_supervised.gin --wandb_project SUPERVISED_FINAL


