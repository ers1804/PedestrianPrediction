
import gin
import torch.nn as nn
from models.supervised.lifting_networks.model_utils import Encoder2D, Decoder3D


@gin.configurable
class SimpleLiftingModel(nn.Module):
    def __init__(self, latent_dim, num_joints=13, activation=nn.ReLU):
        super(SimpleLiftingModel, self).__init__()

        self.type = "keypoints"

        self.num_joints = num_joints
        self.encoder_2d = Encoder2D(latent_dim, num_joints, activation)
        self.decoder_3d = Decoder3D(latent_dim, num_joints, activation)

    def forward(self, poses_2d):
        enc_2d = self.encoder_2d(poses_2d)
        pose_3d_est = self.decoder_3d(enc_2d).reshape(-1, self.num_joints, 3)

        return pose_3d_est
