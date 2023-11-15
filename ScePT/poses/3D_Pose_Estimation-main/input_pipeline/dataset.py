import gin
import logging
import sys
import torch

from torch.utils.data import DataLoader
from torch.utils import data as torch_data
from input_pipeline.waymo_dataloader import WaymoOpenDataset
from input_pipeline.transforms import NormalizeKeypoints2D, NormalizeKeypoints3D, NormalizePointCloud


def load_main_3D_data(data_src, name, train_factor, test_factor, val_factor, alpha_pose=False):
    # get 3D_2D data available
    waymo_csv = data_src + "3D_2D/" + "image_segment_relations.csv"
    waymo_labels = data_src + "3D_2D/" + "labels.pkl"
    if alpha_pose:
        logging.info("Using 2D Alpha Pose Labels...")
        waymo_labels = data_src + "alpha_pose/" + "3D_2D_ap.pkl"
        logging.info("Overwriting gin.config arguements with standard ones: pc_min_size=20, min_2D_keypoints=7, lidar_projection_ratio=0.5")
        waymo_data = WaymoOpenDataset(waymo_csv, waymo_labels, name, pc_min_size=20, min_2D_keypoints=7, lidar_projection_ratio=0.5,
                                      transform_kp_2D=NormalizeKeypoints2D(), )  # , transform_kp_3D=NormalizeKeypoints3D())

    else:
        waymo_data = WaymoOpenDataset(waymo_csv, waymo_labels, name, transform_kp_2D=NormalizeKeypoints2D())  # , transform_kp_3D=NormalizeKeypoints3D())

    train_size = round(train_factor * len(waymo_data))
    test_size = round(test_factor * len(waymo_data))
    val_size = round(val_factor * len(waymo_data))

    if len(waymo_data) != (train_size + test_size + val_size):
        train_size += len(waymo_data) - (train_size + test_size + val_size)

    train_data, test_data, val_data = torch_data.random_split(waymo_data, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_data, test_data, val_data


@gin.configurable
def load(data_src, batch_size, name):
    """
    Load the dataset from pickle files
    Args:
        name (str): name of the dataset that should be loaded
        data_src (str): path to the root directory of the data
    Returns:
          train, test and val dataset ready to be used by pytorch
    """

    if name != 'all_3D_samples':

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        train_factor = 0.6
        test_factor = 0.25
        val_factor = 0.15

        if name not in ["waymo_alphapose_weakly_supervised", "waymo_2d_labels_supervised", "waymo_3d_2d_projections_supervised", "waymo_weakly_supervised", "waymo_alphapose_weakly_supervised"]:
            logging.error(
                'Currently only "waymo_2d_labels_supervised", "waymo_alphapose_weakly_supervised" "waymo_3d_2d_projections_supervised" and "waymo_weakly_supervised" dataset available. Exiting now!')
            sys.exit(0)

        logging.info(f"Preparing dataset {name}...")
        ### SUPERVISED ###
        if name == "waymo_2d_labels_supervised" or name == "waymo_3d_2d_projections_supervised":
            train_data, test_data, val_data = load_main_3D_data(data_src, name, train_factor, test_factor, val_factor)
        ### WEAKLY-SUPERVISED ###
        else:
            logging.info('### LOADING TRAINING DATA ###')
            if name == "waymo_alphapose_weakly_supervised":
                waymo_csv = data_src + "alpha_pose/" + "image_segment_relations_wo_3D_2D.csv"
                waymo_labels = data_src + "alpha_pose/" + "labels_wo_3D_2D.pkl"
            elif name == "waymo_weakly_supervised":
                waymo_csv = data_src + "2D/" + "image_segment_relations.csv"
                waymo_labels = data_src + "2D/" + "labels.pkl"
            train_data = WaymoOpenDataset(waymo_csv, waymo_labels, name, weakly_supervised=True, transform_kp_2D=NormalizeKeypoints2D())
            logging.info('### LOADING TEST AND VALIDATION DATA ###')
            # change name to load val and test data from 3D samples
            name = "weakly_supervised_testing"
            _, test_data, val_data = load_main_3D_data(data_src, name, train_factor, test_factor, val_factor, alpha_pose=True)

        logging.info(f' Train dataset size: {len(train_data)}')
        logging.info(f' Val dataset size: {len(val_data)}')
        logging.info(f' Test dataset size: {len(test_data)}')

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader, val_dataloader

    else:
        logging.info('')
        logging.info('')
        logging.info('')
        logging.info('### LOADING TEST AND VALIDATION DATA - ALL 3D SAMPLES AVAILABLE ###')

        name = "weakly_supervised_testing"
        waymo_csv_val_test = data_src + "3D_2D/" + "image_segment_relations.csv"
        waymo_labels_val_test = data_src + "3D_2D/" + "labels.pkl"
        logging.info("Overwriting gin.config arguements with standard ones: pc_min_size=20, min_2D_keypoints=7, lidar_projection_ratio=0.5")
        waymo_data_all = WaymoOpenDataset(waymo_csv_val_test, waymo_labels_val_test, name, pc_min_size=20, min_2D_keypoints=7, lidar_projection_ratio=0.5, transform_kp_2D=NormalizeKeypoints2D())
        test_all_dataloader = DataLoader(waymo_data_all,  batch_size=len(waymo_data_all), shuffle=False)
        logging.info(f' Test_all dataset size: {len(waymo_data_all)}')

        return test_all_dataloader
