"""Code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_cls.py"""
import torch.nn as nn
import gin
import torch.utils.data
import torch.nn.functional as F
from poses.Pose_Estimation_main.models.supervised.point_networks.pointnet_utils import PointNetEncoder


@gin.configurable
class PointNet(nn.Module):
    def __init__(self, dropout, num_joints=13):
        super(PointNet, self).__init__()

        channel = 3
        self.num_joints = num_joints

        self.type = "point_cloud"
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_joints*3)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1) - remove softmax since we apply PointNet for a regression task
        return torch.reshape(x, (-1, self.num_joints, 3)), trans_feat
