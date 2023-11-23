import sys
import gin
import time
import logging
import torch
import wandb
import torch.optim as optim
from evaluation.metrics import Metrics
from train import Trainer
from configs.constants import *
import numpy as np

# AlphaPose
################
sys.path.append('/home/erik/gitproject/AlphaPose')

from alphapose.models import builder
from alphapose.utils.config import update_config
from trackers.tracker_cfg import cfg as tcfg
from alphapose.utils.detector import DetectionLoader
from detector.apis import get_detector
from tqdm import tqdm

################


@gin.configurable
class SupervisedTrainer(Trainer):
    """
    Class to train the implemented supervised models
    """

    def __init__(self, model, train_set, val_set, test_set, run_paths, epochs,
                 lr, lr_decay_factor, lr_step, loss_types={'masked_mpjpe': True},
                 device="cpu", wandb=False, waymo_evaluation=True, flags=None):

        super().__init__(train_set, val_set, test_set, run_paths, epochs,
                         device, wandb, waymo_evaluation,  type='supervised')

        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_step = lr_step

        self.generator = model
        self.loss_types = loss_types
        self.optimiser = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.wandb = wandb
        self.generator.to(self.device)
        self.use_alpha = flags.use_alpha
        self.flags = flags
        if self.use_alpha:
            self.cfg = update_config(flags.alpha_cfg)
            model_config = self.cfg.MODEL
            data_preset = self.cfg.DATA_PRESET
            self.alpha = builder.build_sppe(model_config, preset_cfg=data_preset)
            self.alpha.load_state_dict(torch.load(flags.alpha_checkpoint, map_location=self.device))
            self.alpha.to(self.device)
            self.alpha.eval()
        else:
            self.alpha = None
    def map_keypoints(self, keypoints_coco):
        """
        Method to map keypoints from COCO format to Waymo format
        keypoints_coco: [x1, y1, c1, ..., xk, yk, ck]
        keypoints_waymo: array [B, num_joints, 2]
        """
        batch_size = gin.query_parameter('load.batch_size')
        keypoints_waymo = torch.empty((batch_size, self.generator.num_joints, 2), dtype=torch.float32)
        num_keypoints = len(keypoints_coco) // 3
        arr_keypoints_coco = torch.Tensor(keypoints_coco)
        arr_keypoints_coco = arr_keypoints_coco.reshape(-1, num_keypoints, 3)
        sequence_coco = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
        for point in sequence_coco:
            if point in JOINT_NAMES:
                keypoints_waymo[:, JOINT_NAMES.index(point), :] = arr_keypoints_coco[:, sequence_coco.index(point), :2]
            else:
                continue
        return keypoints_waymo
    def train(self):
        """
        Training routine
        """

        self.print_train_start()

        self.generator.train()

        for self.epoch in range(self.epochs):
            self.epoch_start_time = time.time()
            logging.info(f'Epoch:{self.epoch+1}')

            self.train_loss = 0.0
            self.samples_counter = 0

            # start training loop
            for data in self.train_set:

                # clear gradients
                self.optimiser.zero_grad()
                if self.use_alpha:
                    det_loader = DetectionLoader([numpy_array.numpy() for numpy_array in data['img']],
                                                 get_detector(self.flags),
                                                 self.cfg,
                                                 self.flags,
                                                 batchSize=self.flags.detbatch,
                                                 mode='loaded_image',
                                                 queueSize=self.flags.qsize)
                    det_worker = det_loader.start()
                    data_len = det_loader.length
                    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
                    batchSize = self.flags.posebatch
                    for _ in im_names_desc:
                        with torch.no_grad():
                            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                            if orig_img is None:
                                # TODO: Decide, what to do in this case!
                                break
                            if boxes is None or boxes.nelement() == 0:
                                print('No Human Detected')
                                # TODO: Decide, what to do in this case!
                                continue
                            # Pose Estimation
                            inps = inps.to(self.device)
                            datalen = inps.size(0)
                            leftover = 0
                            if (datalen) % batchSize:
                                leftover = 1
                            num_batches = datalen // batchSize + leftover
                            hm = []
                            for j in range(num_batches):
                                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                                hm_j = self.alpha(inps_j)
                                hm.append(hm_j)
                            hm = torch.cat(hm)
                    print("2D Keypoints estimated")

                else:
                    keypoints_2D = data['keypoints_2D'].to(self.device)
                keypoints_3D = data['keypoints_3D'].to(self.device)
                pc = data['pc'].to(self.device).transpose(2, 1)

                # forward pass
                if self.generator.type == "point_cloud":
                    predictions, trans_features = self.generator(pc)
                elif self.generator.type == "keypoints":
                    predictions = self.generator(keypoints_2D)
                    trans_features = None
                elif self.generator.type == "fusion":
                    predictions, trans_features, loss_contributions = self.generator(pc, keypoints_2D)

                else:
                    logging.error('Model input not defined properly.')
                    sys.exit(1)

                step_loss = self.calculate_loss_supervised(predictions, keypoints_3D, data['mask_3D'], trans_features)

                # Backpropagation
                step_loss.backward()
                self.optimiser.step()

                self.train_loss += step_loss.item() * keypoints_2D.shape[0]
                self.samples_counter += keypoints_2D.shape[0]

                if self.current_step % 10 == 0:
                    logging.info("step {0:04d}; Train loss avg: {1:.4f}".format(
                        self.current_step, self.train_loss/self.samples_counter))

                self.current_step += 1

                # lr decay
                if self.current_step % self.lr_step == 0:
                    self.lr = self.lr*self.lr_decay_factor
                    for g in self.optimiser.param_groups:
                        g['lr'] = self.lr

                    logging.info("Decay learning rate. New value at " + str(self.lr))
            self.epoch_end_time = time.time()
            # save checkpoint and run evaluation on val data after each epoch
            self.validate()
        logging.info(
            f'Finsihed with training. Logs can be found at {self.run_paths["path_model_id"]} ')
        return self.generator

    def validate(self):
        """Validation procedure of the supervised training routine"""

        step_losses = []

        with torch.no_grad():
            self.generator.eval()
            for data in self.val_set:
                batch_dim = data['keypoints_2D'].shape[0]
                keypoints_2D = data['keypoints_2D'].to(self.device)
                keypoints_3D = data['keypoints_3D'].to(self.device)
                pc = data['pc'].to(self.device).transpose(2, 1)

                if self.generator.type == "point_cloud":
                    tmp_preds_keypoints3D, trans_features = self.generator(pc)
                elif self.generator.type == "keypoints":
                    tmp_preds_keypoints3D = self.generator(keypoints_2D)
                    trans_features = None
                elif self.generator.type == "fusion":
                    tmp_preds_keypoints3D, trans_features, loss_contributions = self.generator(pc, keypoints_2D, gt=(keypoints_3D, data['mask_3D']))
                else:
                    logging.error('Model input not defined properly.')
                    sys.exit(1)

                step_loss = Metrics.masked_mpjpe(tmp_preds_keypoints3D, keypoints_3D, data['mask_3D'])
                step_losses.append(step_loss * batch_dim)

            self.val_loss = sum(step_losses)/len(self.val_set.dataset)
            self.print_val_stats_and_save_model()
            if self.wandb:
                self.push_wandb()
            self.previous_losses.append(self.train_loss)
            sys.stdout.flush()
            self.generator.train()

    def push_wandb(self):
        """Push all data to Weights and Biases."""
        wandb_dict = {
            'val_loss': self.val_loss,
            'train_loss': self.train_loss/self.samples_counter,
            'learning_rate': self.lr,
            'step': self.current_step,
            'epoch': self.epoch
        }
        wandb.log(
            wandb_dict
        )

    def calculate_loss_supervised(self, predictions, keypoints_3D, mask_3D, trans_features=None):
        step_loss = 0
        if self.loss_types["masked_mpjpe"]:
            step_loss += self.loss_types["masked_mpjpe"] * Metrics.masked_mpjpe(predictions, keypoints_3D, mask_3D)
        if self.loss_types["l1"]:
            step_loss += self.loss_types["l1"] * Metrics.masked_l1(predictions, keypoints_3D, mask_3D)
        if self.loss_types["bone_length"]:
            step_loss += self.loss_types["bone_length"] * Metrics.bone_length_symmetry(predictions)
        if (self.generator.type == "point_cloud" or self.generator.type == "fusion") and self.loss_types["feature_transform_reguliarzer"]:
            step_loss += self.loss_types["feature_transform_reguliarzer"] * Metrics.feature_transform_reguliarzer(trans_features)

        return step_loss
