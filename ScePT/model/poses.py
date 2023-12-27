import sys
from poses.Pose_Estimation_main.models.supervised.fusion.lidar_2dkeypoint import Lidar2dKeypointFusionmodel
import torch
import logging

# AlphaPose
sys.path.append('/home/erik/gitproject/AlphaPose')
from alphapose.utils.config import update_config
from alphapose.models import builder
from trackers.tracker_cfg import cfg as tcfg
from alphapose.utils.detector import DetectionLoader
from detector.apis import get_detector
from alphapose.utils.writer import DataWriter


def get_pose_model():
    return Lidar2dKeypointFusionmodel()


# TODO: Write a class that combines AlphaPose and the Fusion Network
class PoseEstimator:
    def __init__(self, model_id, args):
        self.args = args

        # Load Fusion Network
        self.model = get_pose_model()
        logging.info(f"Loading model weights from {model_id}/ckpts/best_model")
        self.model.load_state_dict(torch.load("/home/erik/ScePT/ScePT/poses/runs/" + model_id + "/ckpts/best_model"))
        self.model.to(self.args.device)
        self.model.eval()
        # Load AlphaPose
        self.alpha_cfg = update_config(self.args.alpha_cfg)
        model_config = self.alpha_cfg.MODEL
        data_preset = self.alpha_cfg.DATA_PRESET
        self.alpha = builder.build_sppe(model_config, preset_cfg=data_preset)
        self.alpha.load_state_dict(torch.load(self.args.alpha_checkpoint, map_location=self.args.device))
        self.alpha.to(self.args.device)
        self.alpha.eval()
    

    def forward():
        pass