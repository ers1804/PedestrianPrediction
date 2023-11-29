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
#from tqdm import tqdm
from alphapose.utils.writer import DataWriter
#from input_pipeline.transforms import NormalizeKeypoints2D
from train import normalize_keypoints2D_batch


# def normalize_keypoints2D_batch(keypoint_batch, epsilon=1e-6):
#     max_values = torch.max(keypoint_batch, dim=1).values
#     min_values = torch.min(keypoint_batch, dim=1).values

#     height = max_values[:, 1] - min_values[:, 1]
#     width = max_values[:, 0] - min_values[:, 0]

#     keypoint_batch_zz = keypoint_batch.transpose(0, 1) - min_values

#     keypoint_batch_trans = keypoint_batch_zz.transpose(1, 2)

#     # norm height [-1,1]
#     norm_kp = (keypoint_batch_trans/(height+epsilon))*2
#     norm_kp[:, 0, :] = norm_kp[:, 0, :] - width/(height+epsilon)
#     norm_kp[:, 1, :] = norm_kp[:, 1, :] - 1

#     return norm_kp.permute(2, 0, 1)

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
                         device, wandb, waymo_evaluation,  flags, type='supervised')

        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_step = lr_step

        self.generator = model
        self.loss_types = loss_types
        self.optimiser = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.wandb = wandb
        self.generator.to(self.device)
        # if flags is not None:
        #     self.use_alpha = flags.use_alpha
        # else:
        #     self.use_alpha = False
        # self.flags = flags
        # if self.use_alpha:
        #     self.cfg = update_config(flags.alpha_cfg)
        #     model_config = self.cfg.MODEL
        #     data_preset = self.cfg.DATA_PRESET
        #     self.alpha = builder.build_sppe(model_config, preset_cfg=data_preset)
        #     self.alpha.load_state_dict(torch.load(flags.alpha_checkpoint, map_location=self.device))
        #     self.alpha.to(self.device)
        #     self.alpha.eval()
        # else:
        #     self.alpha = None
    def map_keypoints(self, keypoints_coco):
        """
        Method to map keypoints from COCO format to Waymo format
        keypoints_coco: [x1, y1, c1, ..., xk, yk, ck]
        keypoints_waymo: array [B, num_joints, 2]
        """
        batch_size = gin.query_parameter('load.batch_size')
        keypoints_waymo = torch.zeros((keypoints_coco.shape[0], self.generator.num_joints, 2), dtype=torch.float32)
        #num_keypoints = len(keypoints_coco) // 3
        #arr_keypoints_coco = torch.Tensor(keypoints_coco)
        #arr_keypoints_coco = arr_keypoints_coco.reshape(-1, num_keypoints, 3)
        sequence_coco = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
        for point in sequence_coco:
            if point in JOINT_NAMES:
                keypoints_waymo[:, JOINT_NAMES.index(point), :] = keypoints_coco[:, sequence_coco.index(point), :2]
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
                    with torch.no_grad():
                        det_loader = DetectionLoader([numpy_array.numpy() for numpy_array in data['img']],
                                                    get_detector(self.flags),
                                                    self.cfg,
                                                    self.flags,
                                                    batchSize=self.flags.detbatch,
                                                    mode='loaded_image',
                                                    queueSize=self.flags.qsize)
                        det_worker = det_loader.start()
                        data_len = det_loader.length
                        #im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
                        batchSize = self.flags.posebatch
                        writer = DataWriter(self.cfg, self.flags, save_video=False, queueSize=self.flags.qsize).start()
                        # If AlphaPose cannot find any humans in the image, use groundtruth keypoints
                        indices_wo_detection = []
                        indices_w_detection = []
                        for i in range(data_len):
                            with torch.no_grad():
                                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                                if orig_img is None:
                                    # TODO: Decide, what to do in this case!
                                    break
                                if boxes is None or boxes.nelement() == 0:
                                    print('No Human Detected')
                                    # Save indices and add groundtruth keypoints later
                                    indices_wo_detection.append(i)
                                    continue
                                # Pose Estimation
                                indices_w_detection.append(i)
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
                                hm = hm.cpu()
                                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                        results = writer.results() # List of dictionaries containing: {'imgname': 'x.jpg', 'result': dict_with_results}
                        writer.clear_queues()
                        #print("2D Keypoints estimated")
                        writer.stop()
                        det_loader.stop()
                        det_loader.terminate()
                        # dict_with_results is a list containing dictionaries (length of one as we only have one person in the image)
                        # with {'keypoints': [68,2], 'kp_score': [68,1], 'proposal_score': [1,], 'idx': [1,], box': [4]}
                        # First filter keypoints to the ones you want then normalize using the re-implemented function
                        # Output should be a tensor of shape [batch_size, num_joints, 2]
                        # Batch the keypoints to tensor
                        parsed_keypoints = torch.zeros((data['keypoints_2D'].shape[0], self.cfg.DATA_PRESET.NUM_JOINTS, 2), dtype=torch.float32)
                        for i, sample in enumerate(results):
                            index = indices_w_detection[i]
                            if len(sample['result']) == 0:
                                indices_wo_detection.append(index)
                            else:
                                parsed_keypoints[index] = sample['result'][0]['keypoints']
                        # Map Keypoints
                        keypoints_2D = self.map_keypoints(parsed_keypoints)
                        # Normalize Keypoints
                        keypoints_2D = normalize_keypoints2D_batch(keypoints_2D).to(self.device)
                        #print("Keypoints Normalized")
                        # Add groundtruth keypoints for samples without detection
                        for i in indices_wo_detection:
                            keypoints_2D[i] = data['keypoints_2D'][i]
                else:
                    keypoints_2D = data['keypoints_2D'].to(self.device)
                keypoints_3D = data['keypoints_3D'].to(self.device)
                pc = data['pc'].to(self.device).transpose(2, 1)

                if pc.shape[0] != keypoints_2D.shape[0]:
                    print("Different batch sizes for pc and keypoints_2D.")

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
        torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary())

        with torch.no_grad():
            self.generator.eval()
            for data in self.val_set:
                batch_dim = data['keypoints_2D'].shape[0]

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
                    #im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
                    batchSize = self.flags.posebatch
                    writer = DataWriter(self.cfg, self.flags, save_video=False, queueSize=self.flags.qsize).start()
                    # If AlphaPose cannot find any humans in the image, use groundtruth keypoints
                    indices_wo_detection = []
                    indices_w_detection = []
                    for i in range(data_len):
                        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                        if orig_img is None:
                            # TODO: Decide, what to do in this case!
                            break
                        if boxes is None or boxes.nelement() == 0:
                            print('No Human Detected')
                            # Save indices and add groundtruth keypoints later
                            indices_wo_detection.append(i)
                            continue
                        # Pose Estimation
                        indices_w_detection.append(i)
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
                        hm = hm.cpu()
                        writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    results = writer.results() # List of dictionaries containing: {'imgname': 'x.jpg', 'result': dict_with_results}
                    #print("2D Keypoints estimated")
                    #writer.terminate()
                    writer.clear_queues()
                    writer.stop()
                    det_loader.stop()
                    det_loader.terminate()
                    # dict_with_results is a list containing dictionaries (length of one as we only have one person in the image)
                    # with {'keypoints': [68,2], 'kp_score': [68,1], 'proposal_score': [1,], 'idx': [1,], box': [4]}
                    # First filter keypoints to the ones you want then normalize using the re-implemented function
                    # Output should be a tensor of shape [batch_size, num_joints, 2]
                    # Batch the keypoints to tensor
                    parsed_keypoints = torch.zeros((data['keypoints_2D'].shape[0], self.cfg.DATA_PRESET.NUM_JOINTS, 2), dtype=torch.float32)
                    for i, sample in enumerate(results):
                        index = indices_w_detection[i]
                        if len(sample['result']) == 0:
                            indices_wo_detection.append(index)
                        else:
                            parsed_keypoints[index] = sample['result'][0]['keypoints']
                    # Map Keypoints
                    keypoints_2D = self.map_keypoints(parsed_keypoints)
                    # Normalize Keypoints
                    keypoints_2D = normalize_keypoints2D_batch(keypoints_2D).to(self.device)
                    #print("Keypoints Normalized")
                    # Add groundtruth keypoints for samples without detection
                    for i in indices_wo_detection:
                        keypoints_2D[i] = data['keypoints_2D'][i]
                else:
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
