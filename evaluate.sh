#!/bin/bash

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=68 \
#     --log_dir=/home/erik/NAS/cluster/default-dl-training-pvc-pvc-d366543f-80c2-4cf8-9e47-8de49d960aee/scept-poses-gt-training-paper/ \
#     --trained_model_dir=poses-gt-w-augment-hidden-64-64_27_Jan_2024_15_15_38 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed_pose_gt1_augment \
#     --mode=poses-gt \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_pose_gt1_config_aug.json \
#     --map_encoding \
#     --batch_size=1024

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=71 \
#     --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
#     --trained_model_dir=base-retraining-complete30_Jan_2024_14_27_44 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed \
#     --mode=base \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_config.json \
#     --map_encoding \
#     --batch_size=1024

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=71 \
#     --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
#     --trained_model_dir=poses-gt-w-augment_27_Jan_2024_15_15_33 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed_pose_gt1_augment \
#     --mode=poses-gt \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_pose_gt1_config_aug.json \
#     --map_encoding \
#     --batch_size=1024

python ./ScePT/evaluate.py \
    --eval_data_dict=nuScenes_val.pkl \
    --iter_num=69 \
    --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
    --trained_model_dir=poses-gt-wo-augment-implicit_27_Jan_2024_15_15_44 \
    --eval_task=eval_statistics \
    --data_dir=./experiments/processed_pose_gt1 \
    --mode=poses-gt \
    --num_workers=8 \
    --model_id=run_2023-11-29T07-48-39-851756 \
    --use_processed_data \
    --conf=./config/clique_nusc_pose_gt1_config.json \
    --map_encoding \
    --batch_size=1024 \
    --implicit

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=71 \
#     --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
#     --trained_model_dir=poses-gt-w-augment-norm-batch_27_Jan_2024_15_15_54 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed_pose_gt1_augment \
#     --mode=poses-gt \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_pose_gt1_config_aug.json \
#     --map_encoding \
#     --batch_size=1024

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=69 \
#     --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
#     --trained_model_dir=poses-gt-w-augment-hidden-128-64_27_Jan_2024_15_16_05 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed_pose_gt1_augment \
#     --mode=poses-gt \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_pose_gt1_config_aug.json \
#     --map_encoding \
#     --batch_size=1024

# python ./ScePT/evaluate.py \
#     --eval_data_dict=nuScenes_val.pkl \
#     --iter_num=71 \
#     --log_dir=/home/erik/ScePT/scept-poses-gt-training-paper/ \
#     --trained_model_dir=poses-gt-trial-wo-augment28_Jan_2024_23_17_48 \
#     --eval_task=eval_statistics \
#     --data_dir=./experiments/processed_pose_gt1 \
#     --mode=poses-gt \
#     --num_workers=8 \
#     --model_id=run_2023-11-29T07-48-39-851756 \
#     --use_processed_data \
#     --conf=./config/clique_nusc_pose_gt1_config.json \
#     --map_encoding \
#     --batch_size=1024