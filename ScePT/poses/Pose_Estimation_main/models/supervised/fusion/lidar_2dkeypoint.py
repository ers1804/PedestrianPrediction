import gin
import torch
import torch.nn as nn

from poses.Pose_Estimation_main.evaluation.metrics import Metrics
from poses.Pose_Estimation_main.models.supervised.point_networks.pointnet import PointNet
from poses.Pose_Estimation_main.models.supervised.lifting_networks.simple_lifting_model import SimpleLiftingModel


@gin.configurable
class Lidar2dKeypointFusionmodel(nn.Module):
    def __init__(self, num_joints=13, channels=3):
        super(Lidar2dKeypointFusionmodel, self).__init__()

        self.num_joints = num_joints
        self.channels = channels
        self.loss_contributions = (1, 1)
        self.type = "fusion"

        # self.w1 = nn.Linear(self.num_joints*self.channels*2, hidden_size)
        self.w2 = nn.Linear(self.num_joints*self.channels*2, self.num_joints*self.channels)
        # self.activ = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)

        self.point_net = PointNet(num_joints=self.num_joints)
        self.lifting_net = SimpleLiftingModel(num_joints=self.num_joints)

    def forward(self, pc, keypoints_2D, gt=None):
        """

        Args:
            pc (tensor): Point cloud of shape [B, num_joints, channels], where channels usually is 3 (x,y,z).
            keypoints_2D (tensor): Two dimensional  keypoints [B, num_joints, 2]
            gt (tuple, optional): Ground truth data as tuple of 3D keypoint tensor and mask tensor.
                                  Only used for eval step where loss values of lifting and point cloud are compared
                                  Defaults to None.

        Returns:
            tuple: (predictions tensor, transformation features from PointNet (regularization), loss contributions of each net)
        """
        point_net_preds, trans_features = self.point_net(pc)
        lifting_net_preds = self.lifting_net(keypoints_2D.repeat(2,1,1))[:1, :, :]

        if gt:
            keypoints_3D = gt[0]
            mask = gt[1]
            lifting_mpjpe = Metrics.masked_mpjpe(point_net_preds, keypoints_3D, mask)
            point_mpjpe = Metrics.masked_mpjpe(lifting_net_preds, keypoints_3D, mask)
            self.loss_contributions = (point_mpjpe, lifting_mpjpe)

        x = torch.cat([point_net_preds, lifting_net_preds], axis=1).view(point_net_preds.shape[0], -1)
        # x = self.w1(x)
        # x = self.activ(x)
        # x = self.dropout(x)
        x = self.w2(x)

        return x.reshape(x.shape[0], -1, self.channels), trans_features, self.loss_contributions
