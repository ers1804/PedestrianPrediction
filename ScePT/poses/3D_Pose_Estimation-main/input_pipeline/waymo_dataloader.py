
import pandas as pd
import numpy as np
import pickle
import logging
import cv2
import random
import gin
import sys

sys.path.append("../../../../../ScePT")

from torch.utils.data import Dataset
#from waymo_open_dataset import v2
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from configs.constants import JOINT_KEYS


@gin.configurable
class WaymoOpenDataset(Dataset):
    """Custom Waymo Dataset"""

    def __init__(self, csv, labels, name, pc_min_size, min_2D_keypoints, alpha_pose_confidence_score=0.6, num_closest_cp=4, lidar_projection_ratio=0.8,
                 weakly_supervised=False, transform_kp_2D=None, transform_kp_3D=None, transform_pc=None, rm_no_hips=True):
        """
        Args:
            csv (df): pandas df that stores information about the sample (id, frame, tfr-file, etc.)
            labels (dict): labels from the dataset
            name (str): name of the data to load
            pc_min_size (int): minimum number of lidar points in sample
            min_2D_keypoints (int): minimum number of labeled joints in sample
            num_closest_cp (int): defines how many lidar points are returned to infer pseudo 3D label
            transform_kp_2D (python class): transformation for 2d keypoints
            transform_kp_3D (python class): transformation for 3d keypoints
            transform_kp_pc (python class): transformation for point cloud data
            rm_no_hips (bool): remove 3D samples without both hips labeled
        """

        self.pc_min_size = pc_min_size
        self.min_2D_keypoints = min_2D_keypoints
        self.lidar_projection_ratio = lidar_projection_ratio
        self.num_closest_cp = num_closest_cp
        self.transform_kp_2D = transform_kp_2D
        self.transform_kp_3D = transform_kp_3D
        self.transform_pc = transform_pc
        self.name = name
        self.no_hips = []
        self.csv = pd.read_csv(csv)
        self.image_path = "/".join(csv.split('/')[:-1]) + "/images/"
        self.weakly_supervised = weakly_supervised
        self.alpha_pose_confidence_score = alpha_pose_confidence_score

        # if self.name == "waymo_alphapose_weakly_supervised":
        #     logging.info("CURRENTLY ONLY USING LABELS THAT ARE ALSO LABELeD IN 2D AND NOT ALL DETECTIONS AVAILABALE IN THE DATASET")
        #     self.csv = pd.read_csv("/media/petbau/data/waymo/v0.10/alpha_pose/image_segment_relations_compressed.csv")
        #     labels = "/media/petbau/data/waymo/v0.10/alpha_pose/labels_compressed.pkl"

        logging.info(f'Loading labels from {labels}...')
        with open(labels, 'rb') as pickle_file:
            self.labels = pickle.load(pickle_file)

        self.total_count = 0
        self.initial_number_of_saples = len(self.labels)

        self.rm_cams(del_cams=set([]))

        #     self.labels = dict(filter(lambda sub: sub[1], self".labels.items()))
        # remove samples that do not contain hips
        # if rm_no_hips and not self.weakly_supervised:
        #     self.rm_no_hips()

        # only use samples that contain more than self.pc_min_size points
        # if self.weakly_supervised:
        self.rm_sparse_lidar_and_projections()

        # remove samples that contain less than self.min_2D_keypoints 2D keypoint
        # if name == "waymo_2d_labels_supervised" or name == "weakly_supervised_testing" or self.weakly_supervised:
        self.rm_sparse_keypoints()

        logging.info(f"In total removed {self.total_count} samples from dataset due to data-cleaning ({round(self.total_count/self.initial_number_of_saples*100,1)}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # get item from file
        key = self.csv.iloc[idx].image_id
        data = self.labels[key]

        bb_2d = np.array(list(data['bb_2d'].values()))
        bb_3d = np.array(list(data['bb_3d'].values()))

        keypoints_2D, occlusions_2D, mask_2D, keypoints_2D_unnormalized, root, keypoints_3D, occlusions_3D, mask_3D = self.process_kp_data(data, key)

        # remove forehead/head_center from 2D/3D (to be more consistent with waymo paper)
        # https://arxiv.org/pdf/2112.12141.pdf
        if not self.name == "waymo_alphapose_weakly_supervised":
            keypoints_2D = keypoints_2D[:-2]
            occlusions_2D = occlusions_2D[:-2]
            mask_2D = mask_2D[:-2]
            keypoints_2D_unnormalized = keypoints_2D_unnormalized[:-2]
            if not isinstance(keypoints_3D, bool):
                keypoints_3D = keypoints_3D[:-2]
                occlusions_3D = occlusions_3D[:-2]
                mask_3D = mask_3D[:-2]
        intrinsics, ks, ps = self.get_intrinsics(np.squeeze(data['intrinsic']))

        if self.weakly_supervised or self.name == "weakly_supervised_testing":

            
            ### TODO: REMOVE ONCE EXTRACTION SCRIPT IS UPDATED
            # if self.name == "waymo_alphapose_weakly_supervised":
            #     margin = 0.3
            #     img_2d_width = (1 + margin) * data['bb_2d']['width']
            #     img_2D_height = (1 + margin) * data['bb_2d']['height']
            #     # resize and store camera projections
            #     min_x = max(0, int(data['bb_2d']['center_x'] - img_2d_width / 2))
            #     min_y = max(0, int(data['bb_2d']['center_y'] - img_2D_height / 2))
            #     data['cp_points'][:, 1] = data['cp_points'][:, 1] - min_x
            #     data['cp_points'][:, 2] = data['cp_points'][:, 2] - min_y
            #     data['cp_points'][data['cp_points'][:, 0] == 0] = 0
            ####
            closest_cp_idx, closest_cp_dist = self.get_closest_cp(keypoints_2D_unnormalized, data['cp_points'][:, 1:], mask_2D)
            pc, closest_cp_idx_resampled = self.get_sampled_pc(data, root, mask_2D, closest_cp_idx)
            # set unnormalized keypoints to zero
            keypoints_2D_unnormalized[:, 0] = keypoints_2D_unnormalized[:, 0] - np.min(keypoints_2D_unnormalized[:, 0][mask_2D[:, 0]])
            keypoints_2D_unnormalized[:, 1] = keypoints_2D_unnormalized[:, 1] - np.min(keypoints_2D_unnormalized[:, 1][mask_2D[:, 1]])

            keypoints_2D_unnormalized = keypoints_2D_unnormalized * mask_2D
            # keypoints_2D = keypoints_2D_unnormalized/root[0]

            return {'keypoints_2D': keypoints_2D.astype('float32'), 'mask_2D': mask_2D, 'pc': pc.astype('float32'), 'occlusions_2D': occlusions_2D,
                    'keypoints_3D': keypoints_3D, 'occlusions_3D': occlusions_3D, 'mask_3D': mask_3D, 'keypoints_2D_unnormalized': keypoints_2D_unnormalized,
                    'closest_cp_idx': closest_cp_idx_resampled, 'closest_cp_dist': closest_cp_dist, 'closest_cp_idx_before_resampling': closest_cp_idx, 'root': root,
                    'intrinsics': intrinsics, 'metadata': data['metadata'], 'bb_2d': bb_2d, 'bb_3d': bb_3d, 'idx': idx, 'key': key}
        else:
            pc, closest_cp_idx = self.get_sampled_pc(data, root, mask_2D)

            return {'keypoints_2D': keypoints_2D.astype('float32'), 'keypoints_3D': keypoints_3D.astype('float32'), 'pc': pc.astype('float32'), 'occlusions_2D': occlusions_2D,  'intrinsics': intrinsics,
                    'occlusions_3D': occlusions_3D, 'mask_2D': mask_2D, 'mask_3D': mask_3D, 'bb_2d': bb_2d, 'bb_3d': bb_3d, 'idx': idx, 'metadata': data['metadata'], 'root': root, }

    def process_kp_data(self, data, key):

        if self.name == "waymo_weakly_supervised" or self.name == "waymo_alphapose_weakly_supervised":
            keypoints_2D, occlusions_2D, mask_2D, keypoints_2D_unnormalized = self.get_keypoints_2D(data)
            keypoints_3D, occlusions_3D, mask_3D = False, False, False
            # root is 3D_bb center since hips are not available for self supervised approach
            inv_extrinsics = np.linalg.inv(data['extrinsic'])
            root_vehicle = np.array([data['bb_3d']['center_x'], data['bb_3d']['center_y'], data['bb_3d']['center_z'], 1])
            # transform to sensor coordinate system
            root = inv_extrinsics @ root_vehicle
            # remove homogenous coordinates
            root = root[:-1] / root[-1]

        if self.name == "waymo_2d_labels_supervised" or self.name == "weakly_supervised_testing":

            keypoints_2D, occlusions_2D, mask_2D, keypoints_2D_unnormalized = self.get_keypoints_2D(data)
            keypoints_3D, occlusions_3D, root = self.get_keypoints_3D(data, key)

            # create usable mask
            mask_3D = np.ones(keypoints_3D.shape, dtype=np.bool)
            mask_3D[data['mask_3d']] = False

        elif self.name == "waymo_3d_2d_projections_supervised":

            # get 3D keypoints
            keypoints_3D, occlusions_3D, root = self.get_keypoints_3D(data, key)
            # project 3D labels to 2D
            keypoints_2D, occlusions_2D, keypoints_2D_unnormalized = self.project_3D(data, transform=self.transform_kp_2D)

            # create usable mask
            mask_3D = np.ones(keypoints_3D.shape, dtype=np.bool)
            mask_3D[data['mask_3d']] = False

            mask_2D = mask_3D[:, :2].copy()
            # same switch as in self.project_3D
            mask_2D[[13, 14]] = mask_2D[[14, 13]]

        return keypoints_2D, occlusions_2D, mask_2D, keypoints_2D_unnormalized, root, keypoints_3D, occlusions_3D, mask_3D

    def project_3D(self, data, transform=True):
        """Project the 3D Lidar labels to the corresponding 2D image plane"""

        keypoints = data['keypoints_3d_arr'][:, :3]
        occlusions = data['keypoints_3d_arr'][:, -1]
        extrinsic = data['extrinsic']
        intrinsic = np.squeeze(data['intrinsic'])
        metadata = data['metadata']
        camera_image_metadata = data['camera_image_metadata']
        frame_pose_transform = data['frame_pose_transform']

        keypoints_world = np.einsum('ij,nj->ni', frame_pose_transform[:3, :3], keypoints) + frame_pose_transform[:3, 3]

        cp_keypoints = py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata, camera_image_metadata, keypoints_world)

        # get max and min of x and y for normalization
        valid_cp_mask = np.array(cp_keypoints[:, -1], dtype=bool)

        cp_keypoints_masked = cp_keypoints[valid_cp_mask]

        if transform:
            min_x = np.min(cp_keypoints_masked[:, 0])
            max_x = np.max(cp_keypoints_masked[:, 0])
            min_y = np.min(cp_keypoints_masked[:, 1])
            max_y = np.max(cp_keypoints_masked[:, 1])

            keypoints_2D = np.empty(cp_keypoints[:, :2].shape)
            keypoints_2D[:, 0] = cp_keypoints[:, 0] - min_x
            keypoints_2D[:, 1] = cp_keypoints[:, 1] - min_y
            height = max_y - min_y
            width = max_x - min_x
            keypoints_2D = self.transform_kp_2D(keypoints_2D, height, width)

            keypoints_2D[data['mask_3d']] = np.array(np.zeros(2))

        # switch forehead and head_center since normally forehead is labeled in 2D domain
        keypoints_2D[[13, 14]] = keypoints_2D[[14, 13]]

        return keypoints_2D, occlusions, cp_keypoints  # cp_keypoints_masked

    def get_keypoints_2D(self, data):
        # remove occlusion status
        keypoints_2D = data['keypoints_2d_arr'][:, :2]
        occlusions_2D = data['keypoints_2d_arr'][:, -1]

        if self.name == "waymo_alphapose_weakly_supervised":
            mask_2D_tmp = occlusions_2D > self.alpha_pose_confidence_score
            mask_2D = np.expand_dims(mask_2D_tmp, 1).repeat(2, 1)
            mask_indices = np.where(mask_2D_tmp == False)

        else:
            mask_2D = np.ones(keypoints_2D.shape, dtype=np.bool)
            mask_2D[data['mask_2d']] = False
            mask_2D_tmp = np.all(mask_2D, axis=-1)
            mask_indices = data['mask_2d']
            # mask forehead and head center again since not used anymore
            # --> Final prediction version only uses 13 keypoints
            if mask_2D_tmp.shape[0] == 15:
                mask_2D_tmp[-2:] = False

        keypoints_masked = keypoints_2D[mask_2D_tmp]
        if self.transform_kp_2D:
            min_x = np.min(keypoints_masked[:, 0])
            max_x = np.max(keypoints_masked[:, 0])
            min_y = np.min(keypoints_masked[:, 1])
            max_y = np.max(keypoints_masked[:, 1])
            norm_keypoints_2D = np.empty(keypoints_2D[:, :2].shape)
            norm_keypoints_2D[:, 0] = keypoints_2D[:, 0] - min_x
            norm_keypoints_2D[:, 1] = keypoints_2D[:, 1] - min_y
            height = max_y - min_y
            width = max_x - min_x
            norm_keypoints_2D = self.transform_kp_2D(norm_keypoints_2D, height, width)
        else:
            norm_keypoints_2D = keypoints_2D

        norm_keypoints_2D[mask_indices] = np.array(np.zeros(2))
        # again mask FOREHEAD and HEAD_CENTER
        if norm_keypoints_2D.shape[0] == 15:
            norm_keypoints_2D[-2:, :] = np.array(np.zeros(2))

        return norm_keypoints_2D, occlusions_2D, mask_2D, keypoints_2D

    def get_keypoints_3D(self, data, key):

        # A Sensor frame is defined for each sensor. It is denoted
        # as a 4x4 transformation matrix that maps data from sensor
        # frame to vehicle frame. This is also known as the ”extrinsics”
        # matrix.

        inv_extrinsics = np.linalg.inv(data['extrinsic'])
        keypoints = data['keypoints_3d_arr']
        # change occlusion dimension to homogeneous dimension
        keypoints[:, -1] = 1

        # apply inverse extrinsic matrix to each row in the sensor
        sensor_keypoints = np.einsum('ij,kj->ki', inv_extrinsics, keypoints)
        sensor_keypoints = sensor_keypoints[:, :3] / sensor_keypoints[:, -1].reshape(-1, 1)

        # # create a root joint between the hips or in the center of the bb
        # # First check if both hips are labeled
        # if 8 in data['keypoints_3d'] and 16 in data['keypoints_3d']:
        #     l_hip = JOINT_KEYS[8]
        #     r_hip = JOINT_KEYS[16]
        #     root = (sensor_keypoints[l_hip, :3] + sensor_keypoints[r_hip, :3]) / 2

        # else:
        #     logging.warning('All samples with no hips should be removed, using 3d_bb center now! Please fix...')
        #     if key not in self.no_hips:
        #         self.no_hips.append(key)
        root_vehicle = np.array([data['bb_3d']['center_x'], data['bb_3d']['center_y'], data['bb_3d']['center_z'], 1])
        # transform to sensor coordinate system
        root = inv_extrinsics @ root_vehicle
        # remove homogenous coordinates
        root = root[:-1] / root[-1]

        # switch from global to bb specific coordinate system (only if kp is available -> != (0, 0, 0) )
        tmp_keypoints_3D = sensor_keypoints  # data['keypoints_3d_arr'][:, :3]
        keypoints_3D = np.subtract(tmp_keypoints_3D, root)

        # reset unlabeled keypoints to zero
        keypoints_3D[data['mask_3d']] = np.array(np.zeros(3))
        occlusions_3D = data['keypoints_3d_arr'][:, -1]

        # if self.transform_kp_3D:
        #     bb_length = data['bb_3d']['length']
        #     bb_height = data['bb_3d']['height']
        #     bb_width = data['bb_3d']['width']
        #     keypoints_3D = self.transform_kp_3D(keypoints_3D, bb_length, bb_width, bb_height)

        return keypoints_3D.astype('float32'), occlusions_3D, root

    def get_sampled_pc(self, data, root, mask_2D, anchor_indices=np.empty(0), sample_to=512):

        # A Sensor frame is defined for each sensor. It is denoted
        # as a 4x4 transformation matrix that maps data from sensor
        # frame to vehicle frame. This is also known as the ”extrinsics”
        # matrix. We map all lidar points into the sensor frame for better predictions.

        inv_extrinsics = np.linalg.inv(data['extrinsic'])
        pc_vehicle = data['lidar']
        # add homogeneous dimension
        pc_vehicle = np.concatenate((pc_vehicle, np.ones([pc_vehicle.shape[0], 1])), axis=1)

        pc_sensor = np.einsum('ij,kj->ki', inv_extrinsics, pc_vehicle)
        pc_sensor = pc_sensor[:, :3] / pc_sensor[:, -1].reshape(-1, 1)

        # root is already in sensor frame (see process_kp_data)
        pc = pc_sensor - root

        if self.weakly_supervised or self.name == "weakly_supervised_testing":
            if pc.shape[0] > sample_to:
                # downsample
                anchor_indices_masked = anchor_indices[mask_2D[:, 0]].flatten()
                tmp_anchor_indices = np.arange(0, len(anchor_indices_masked)).reshape(-1, 4)
                new_anchor_indices = np.ones((anchor_indices.shape[0], self.num_closest_cp)) * -1
                new_anchor_indices[mask_2D[:, 0]] = tmp_anchor_indices
                # anchor_indices are used for pseudo 3D gt and should therfore not be removed
                # 1. get anchor points
                new_pc_anchor = pc[anchor_indices_masked]
                # all indices
                indices = np.arange(pc.shape[0])
                # remove already used points
                indices = indices[~anchor_indices_masked]
                # get as many indices as needed
                num_miss_points = sample_to-len(anchor_indices_masked)
                indices = random.choices(indices, k=num_miss_points)
                new_pc_rest = pc[indices]
                pc = np.concatenate([new_pc_anchor, new_pc_rest])
                updated_anchors = new_anchor_indices
            elif pc.shape[0] < sample_to:
                # upsample
                diff = sample_to-pc.shape[0]
                indices = random.choices(range(pc.shape[0]), k=diff)
                double_points = pc[indices]
                pc = np.concatenate([pc, double_points])
                updated_anchors = anchor_indices
            else:
                # point cloud contains already exactly "sample_to" points
                updated_anchors = anchor_indices

            return pc, updated_anchors.astype('float32')

        # supervised
        else:
            if pc.shape[0] > sample_to:
                # downsample
                indices = random.sample(range(pc.shape[0]), sample_to)
                pc = pc[indices]
            elif pc.shape[0] < sample_to:
                # upsample
                diff = sample_to-pc.shape[0]
                indices = random.choices(range(pc.shape[0]), k=diff)
                double_points = pc[indices]
                pc = np.concatenate([pc, double_points])

            return pc, 0

    def get_closest_cp(self,  keypoints_2D_unnormalized, cp_points, mask_2D):
        """
        Find the closest lidar points in the image for each keypoint
        """

        closest_cp_values = np.zeros((keypoints_2D_unnormalized.shape[0], self.num_closest_cp))
        closest_cp_indices = np.zeros((keypoints_2D_unnormalized.shape[0], self.num_closest_cp), dtype=int)
        for i in range(keypoints_2D_unnormalized.shape[0]):
            dist = np.linalg.norm((cp_points-keypoints_2D_unnormalized[i]), axis=-1)
            closest_cp_idx = np.argpartition(dist, kth=self.num_closest_cp-1)[:self.num_closest_cp]
            closest_cp_indices[i] = closest_cp_idx
            closest_cp_values[i] = dist[closest_cp_idx]

        # mask final output again to reset unlabeled joints
        closest_cp_indices[~mask_2D[:, 0]] = np.array(np.ones(self.num_closest_cp)*-1, dtype=int)
        closest_cp_values[~mask_2D[:, 0]] = np.array(np.ones(self.num_closest_cp)*-1)

        return closest_cp_indices, closest_cp_values

    def get_intrinsics(self, waymo_intrinsics):

        # Camera model:
        # | fx  0 cx 0 |
        # |  0 fy cy 0 |
        # |  0  0  1 0 |
        camera_intrinsics = np.array([
            [waymo_intrinsics[0], 0,                   waymo_intrinsics[2], 0],
            [0,                   waymo_intrinsics[1], waymo_intrinsics[3], 0],
            [0,                   0,                   1,                   0]])

        ks = (waymo_intrinsics[4], waymo_intrinsics[5], waymo_intrinsics[8])
        ps = (waymo_intrinsics[6], waymo_intrinsics[7])

        return camera_intrinsics, ks, ps

    def get_id(self, idx):
        return self.csv.iloc[int(idx)].image_id

    def get_complete_sample(self, idx):
        key = self.csv.iloc[idx].image_id

        img = cv2.imread(self.image_path + key + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.labels[key], img

    def get_no_hips_ids_3D(self):
        return self.no_hips

    def rm_no_hips(self):
        del_keys = []
        count = 0
        for key in self.labels.keys():
            if not (8 in self.labels[key]['keypoints_3d'] and 16 in self.labels[key]['keypoints_3d']):
                del_keys.append(key)
                count += 1
        for key in del_keys:
            del self.labels[key]
            self.csv.drop(self.csv.index[self.csv['image_id'] == key], inplace=True)
        logging.info(f'Removing {count} samples from a dataset due to missing 3D hips.')
        self.total_count += count

    def rm_cams(self, del_cams=set({})):
        """_summary_

        Args:
            del_cams (set, optional): Cameras to delete. Defaults to [].
        """

        del_keys = []
        count = 0
        for key in self.labels.keys():
            cam = int(key.split("_")[1])
            if cam in del_cams:
                del_keys.append(key)
                count += 1
        for key in del_keys:
            del self.labels[key]
        self.csv = self.csv[~self.csv['image_id'].isin(del_keys)]
        logging.info(f'Removing {count} samples from a dataset since they are recorded with camera [{del_cams}].')
        self.total_count += count

    def rm_sparse_lidar_and_projections(self):
        del_keys = []
        count = 0
        for key in self.labels.keys():
            #print(self.labels[key]['lidar'])
            if isinstance(self.labels[key]['lidar'], float):
                del_keys.append(key)
                count += 1
            elif (not self.labels[key]['lidar'].shape[0] >= self.pc_min_size) or (self.lidar_projection_ratio >= self.labels[key]['lidar_cp_points_ratio']):
                del_keys.append(key)
                count += 1
        for key in del_keys:
            del self.labels[key]
        self.csv = self.csv[~self.csv['image_id'].isin(del_keys)]
        logging.info(f'Removing {count} samples from a dataset due to number of Lidar points being smaller than {self.pc_min_size} or lidar projection ratio smaller {self.lidar_projection_ratio}.')
        self.total_count += count

    def rm_sparse_keypoints(self):
        del_keys = []
        count = 0

        # remove based on alpha pose confidence score
        if self.name == "waymo_alphapose_weakly_supervised":
            for key in self.labels.keys():
                hits = np.sum(self.labels[key]['keypoints_2d_arr'][:, -1] > self.alpha_pose_confidence_score)
                if hits <= self.min_2D_keypoints:
                    del_keys.append(key)
                    count += 1
        # normal removement for labeled data
        else:
            for key in self.labels.keys():
                if len(self.labels[key]['keypoints_2d'].keys()) <= self.min_2D_keypoints:
                    del_keys.append(key)
                    count += 1
        for key in del_keys:
            del self.labels[key]
        self.csv = self.csv[~self.csv['image_id'].isin(del_keys)]
        logging.info(f'Removing {count} samples from a dataset due to number of labeled 2D keypoints being smaller than {self.min_2D_keypoints}.')
        self.total_count += count
