#from symbol import yield_stmt
import gin
import torch.nn as nn
import torch
from models.supervised.lifting_networks.simple_lifting_model import SimpleLiftingModel
from models.supervised.fusion.lidar_2dkeypoint import Lidar2dKeypointFusionmodel


@gin.configurable
class Generator(nn.Module):
    def __init__(self, num_joints=13, activation=nn.ReLU):
        super(Generator, self).__init__()
        self.type = "generator"
        self.num_joints = num_joints
        self.second_block = True

        self.lifting_net = SimpleLiftingModel(num_joints=self.num_joints)
        self.lidar_2dkeypoint_fusion_net = Lidar2dKeypointFusionmodel(num_joints=self.num_joints)
        # self.scale = nn.Sequential(nn.Linear(3, 1), nn.Dropout(p=0.2), nn.Sigmoid())

    def forward(self, keypoints_2D, pc=None):

        pose_3D = self.lifting_net(keypoints_2D)
        pose_3D, trans_features, loss_contributions = self.lidar_2dkeypoint_fusion_net(pc, keypoints_2D)
        pose_3D = torch.clamp(pose_3D, min=-2, max=2)

        # if bb_3D is not None:
        #     max_values = torch.max(torch.abs(pose_3D), dim=-2)[0]
        #     scale = self.scale(max_values).squeeze()
        #     pose_3D_perm = pose_3D.permute(-1, -2, -3)
        #     pose_3D = (pose_3D_perm * scale).permute(-1, -2, -3)
        #     scale_penalty = torch.norm(bb_3D-max_values)
        #     return pose_3D, scale_penalty
        # return pose_3D, torch.zeros(1)
        return pose_3D, trans_features
