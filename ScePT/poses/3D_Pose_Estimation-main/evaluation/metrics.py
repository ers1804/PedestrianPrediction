import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

from configs.constants import JOINT_KEYS

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_data

tf.config.run_functions_eagerly(True)

class Metrics():
    """Metrics class to evaluate the performance of keypoint predictions"""

    L1 = nn.L1Loss()
    REVERSE_JOINT_KEYS = {value: key for key, value in JOINT_KEYS.items()}

    def __init__(self):
        pass

    @classmethod
    def mpjpe(cls, predictions, gt):

        mpjpe_per_joint = torch.norm(gt-predictions, p=2, dim=-1)
        total_mpjpe = mpjpe_per_joint.mean(axis=(-1, -2))
        return total_mpjpe, mpjpe_per_joint

    @classmethod
    def masked_mpjpe(cls, predictions, gt, mask):

        # DEPRECATED SINCE FOREHEAD/HEAD CENTER IS REMOVED NOW
        # # remove KEYPOINT_TYPE_FOREHEAD since mainly labeled in 2D
        # # if remove_FOREHEAD:
        # #     tmp_mask = np.ones(predictions.shape[1], dtype=np.bool)
        # #     tmp_mask[-2] = False
        # #     gt = gt[:, tmp_mask]
        # #     predictions = predictions[:, tmp_mask]
        # #     mask = mask[:, tmp_mask]

        gt = gt[mask[:, :, 0]]
        predictions = predictions[mask[:, :, 0]]

        mpjpe_per_joint = torch.norm(gt-predictions, p=2, dim=-1)
        total_mpjpe = mpjpe_per_joint.mean()
        return total_mpjpe

    @classmethod
    def maksed_jointwise_mpjpe(cls, predictions, gt, mask):

        mpjpe_per_joint = torch.empty(predictions.shape[1])

        for i in range(predictions.shape[1]):
            mpjpe_per_joint[i] = cls.masked_mpjpe(torch.unsqueeze(predictions[:, i, :], dim=1),
                                                  torch.unsqueeze(gt[:, i, :], dim=1),
                                                  torch.unsqueeze(mask[:, i, :], dim=1))

        return mpjpe_per_joint

    @staticmethod
    def masked_l1(predictions, gt, mask):

        gt = gt[mask[:, :, 0]]
        predictions = predictions[mask[:, :, 0]]

        l1_loss = Metrics.L1(predictions, gt)
        return l1_loss

    @classmethod
    def bone_length_symmetry(cls, predictions,):

        # nose_head_center = predictions[:, 0, :] - predictions[:, 14, :]
        # left_shoulder_right_shoulder = predictions[:, 1, :] - predictions[:, 7, :]
        # right_hip_left_hip = predictions[:, 10, :] - predictions[:, 4, :]

        right_shoulder_right_hip = predictions[:, 7, :] - predictions[:, 10, :]
        left_shoulder_left_hip = predictions[:, 1, :] - predictions[:, 4, :]

        right_shoulder_right_elbow = predictions[:, 7, :] - predictions[:, 8, :]
        left_shoulder_left_elbow = predictions[:, 1, :] - predictions[:, 2, :]

        right_elbow_right_wrist = predictions[:, 8, :] - predictions[:, 9, :]
        left_elbow_left_wrist = predictions[:, 2, :] - predictions[:, 3, :]

        right_hip_right_knee = predictions[:, 10, :] - predictions[:, 11, :]
        left_hip_left_knee = predictions[:, 4, :] - predictions[:, 5, :]

        right_knee_right_ankle = predictions[:, 11, :] - predictions[:, 12, :]
        left_knee_left_ankle = predictions[:, 5, :] - predictions[:, 6, :]

        len_differences = torch.tensor([
            torch.abs(torch.linalg.norm(right_shoulder_right_hip) - torch.linalg.norm(left_shoulder_left_hip)),
            torch.abs(torch.linalg.norm(right_shoulder_right_elbow) - torch.linalg.norm(left_shoulder_left_elbow)),
            torch.abs(torch.linalg.norm(right_elbow_right_wrist) - torch.linalg.norm(left_elbow_left_wrist)),
            torch.abs(torch.linalg.norm(right_hip_right_knee) - torch.linalg.norm(left_hip_left_knee)),
            torch.abs(torch.linalg.norm(right_knee_right_ankle) - torch.linalg.norm(left_knee_left_ankle)),
        ])

        return torch.sum(len_differences)

    @classmethod
    def waymo_eval(cls, all_metrics, predictions, data):
        """Evalue prediction results on the metrics provided by the waymo open dataset"""

        keypoints_3D = data['keypoints_3D']
        occlusions_3D = data['occlusions_3D']
        mask_3D = data['mask_3D']
        bb_3d = data['bb_3d']

        all_keypoints_gt = []
        all_keypoints_pred = []
        all_boxes = []
        for sample in range(predictions.shape[0]):

            box_3D = label_pb2.Label.Box(center_x=bb_3d[sample][0],
                                         center_y=bb_3d[sample][1],
                                         center_z=bb_3d[sample][2],
                                         width=bb_3d[sample][3],
                                         length=bb_3d[sample][4],
                                         height=bb_3d[sample][5],
                                         heading=bb_3d[sample][6])
            box = keypoint_data.create_laser_box_tensors(box_3D, dtype=tf.float32)

            waymo_keypoints_3D_gt = cls.transform_to_waymo_keypoint(keypoints_3D[sample], occlusions_3D[sample], mask_3D[sample])
            waymo_keypoints_3D_pred = cls.transform_to_waymo_keypoint(predictions[sample], occlusions_3D[sample], mask_3D[sample])

            keypoints_gt = keypoint_data.create_laser_keypoints_tensors(
                waymo_keypoints_3D_gt,
                default_location=tf.constant([0, 0, 0], dtype=tf.float32),
                order=keypoint_data.CANONICAL_ORDER_LASER,
                dtype=tf.float32)
            keypoints_pred = keypoint_data.create_laser_keypoints_tensors(
                waymo_keypoints_3D_pred,
                default_location=tf.constant([0, 0, 0], dtype=tf.float32),
                order=keypoint_data.CANONICAL_ORDER_LASER,
                dtype=tf.float32)

            all_keypoints_gt.append(keypoints_gt)
            all_keypoints_pred.append(keypoints_pred)
            all_boxes.append(box)

        keypoint_tensors_gt = keypoint_data.stack_keypoints(all_keypoints_gt)
        keypoint_tensors_pred = keypoint_data.stack_keypoints(all_keypoints_pred)
        box_tensors = keypoint_data.stack_boxes(all_boxes)

        all_metrics.update_state([keypoint_tensors_gt, keypoint_tensors_pred, box_tensors])

    @staticmethod
    def feature_transform_reguliarzer(trans):
        """Code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py"""
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss

    @classmethod
    def transform_to_waymo_keypoint(cls, keypoints, occlusions, mask_3D):

        keypoints_list = []
        counter = 0
        for keypoint, occlusion in zip(keypoints, occlusions):
            if mask_3D[counter][0]:
                laser_keypoint = keypoint_pb2.LaserKeypoint()
                laser_keypoint.type = cls.REVERSE_JOINT_KEYS[counter]
                laser_keypoint.keypoint_3d.location_m.x = keypoint[0]
                laser_keypoint.keypoint_3d.location_m.y = keypoint[1]
                laser_keypoint.keypoint_3d.location_m.z = keypoint[2]
                laser_keypoint.keypoint_3d.visibility.is_occluded = False if int(occlusion) == 1 else True
                keypoints_list.append(laser_keypoint)
            counter += 1

        return keypoints_list