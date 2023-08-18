
import gin
import torch.nn as nn
from models.supervised.lifting_networks.model_utils import LBAD


@gin.configurable
class Discriminator(nn.Module):
    def __init__(self, num_joints=13, activation=nn.ReLU):
        super(Discriminator, self).__init__()

        self.type = "discriminator"
        self.num_joints = num_joints
        self.second_block = False
        self.activation = activation
        self.drop_out_p = 0.5
        self.neurons = 1024

        self.__build_model()

    def __build_model(self):

        self.input_block = nn.Sequential(
            nn.Linear(2*self.num_joints, self.neurons),  # expand features
            nn.BatchNorm1d(self.neurons),
            self.activation(),
            nn.Dropout(p=self.drop_out_p)
        )
        self.LBAD_1 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_2 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_3 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_3 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_4 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.compression_block = nn.Sequential(
            nn.Linear(self.neurons, int(self.neurons/2)),
            self.activation(),
            nn.Linear(int(self.neurons/2), 1),
            self.activation(),
            nn.Dropout(p=self.drop_out_p),
            nn.Sigmoid()
        )

    def forward(self, poses_2d):

        x = poses_2d.reshape(-1, 2*self.num_joints).float()
        x = self.input_block(x)

        residual = x
        x = self.LBAD_1(x)
        x = self.LBAD_2(x) + residual

        if self.second_block:
            residual = x
            x = self.LBAD_3(x)
            x = self.LBAD_4(x) + residual

        return self.compression_block(x).view(-1)
