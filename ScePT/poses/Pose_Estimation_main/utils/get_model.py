from models.supervised.lifting_networks.simple_lifting_model import SimpleLiftingModel
from models.supervised.point_networks.pointnet import PointNet
from models.supervised.fusion.lidar_2dkeypoint import Lidar2dKeypointFusionmodel

from models.weakly_supervised.discriminator import Discriminator
from models.weakly_supervised.generator import Generator

import torch.nn as nn


def get_model(FLAGS, supervised=False):
    if supervised:
        if FLAGS.model_type.lower() == "SimpleLiftingModel".lower():
            model = SimpleLiftingModel()
        elif FLAGS.model_type.lower() == "PointNet".lower():
            model = PointNet()
        elif FLAGS.model_type.lower() == "Lidar2dKeypointFusion".lower():
            model = Lidar2dKeypointFusionmodel()
        return model
    else:
        discriminator = Discriminator(activation=nn.LeakyReLU)
        discriminator.apply(weights_init)
        generator = Generator(activation=nn.LeakyReLU)
        generator.apply(weights_init)

        return generator, discriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
