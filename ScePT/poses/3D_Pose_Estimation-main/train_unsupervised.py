import sys
import gin
import random
import logging
from numpy import angle
import torch
import wandb
import logging
import torch.optim as optim
import math
import time
import torch.nn as nn

from scipy.spatial.transform import Rotation as Rot
from evaluation.metrics import Metrics
from input_pipeline.transforms import normalize_keypoints2D_batch   # normalize_keypoints2D_batch_by_depth
# from adaption_net.train_adaption import AdaptionNet

from configs.constants import JOINT_NAMES
from train import Trainer
from vis import Visualizer


@gin.configurable
class SelfSupervisedTrainer(Trainer):
    """
    Class to train the implemented self-supervised models
    """

    train_set = None
    test_set = None
    val_set = None

    JOINT_NAMES = JOINT_NAMES

    def __init__(self, generator, discriminator, train_set, val_set, test_set, run_paths, epochs,
                 lr, lr_decay_factor, loss_types, pseudo_weights_3D,
                 pseudo_weight, direct_reprojection_weight,
                 log_grads=True, device="cpu", wandb=False, waymo_evaluation=True):

        super().__init__(train_set, val_set, test_set, run_paths, epochs,
                         device, wandb, waymo_evaluation, type='weakly_supervised')

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        # self.update_generator = 5

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()
        self.pseudo_weights_3D = pseudo_weights_3D

        self.lr = lr
        self.lr_decay_factor = lr_decay_factor

        self.direct_reprojection_weight = direct_reprojection_weight
        self.pseudo_weight = pseudo_weight

        # Setup Adam optimizer
        beta1 = 0.5
        self.optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(beta1, 0.999), weight_decay=1e-4)
        self.log_grads = log_grads

        self.loss_types = loss_types

        # TODO: NOT USED ANYMORE - MAYBE REMOVE LATER
        self.setup_cam_params()
        self.extrinsics = self.extrinsics.to(self.device)
        self.homogeneous_extension = self.homogeneous_extension.to(self.device)

        if self.wandb and self.log_grads:
            self.log_grads_wandb(self.generator)
            self.log_grads_wandb(self.discriminator)

        self.visualizer = Visualizer(self, dataset="waymo_weakly_supervised")

        # DEPRECATED SINCE ADAPTION DOES NOT IMPROVE RESULTS...
        # self.adaption_net = AdaptionNet(latent_dim=128)
        # self.adaption_net.eval()
        # self.adaption_net.to(self.device)
        # self.adaption_net.load_state_dict(torch.load("path_to_model"))

    def train(self):
        """
        Training routine for self-supervised approaches.
        """

        self.losses = []

        self.print_train_start()
        self.generator.train()
        self.discriminator.train()

        for self.epoch in range(self.epochs):
            self.epoch_start_time = time.time()
            self.losses_tmp = []
            logging.info(f'Epoch:{self.epoch+1}')

            for i, data in enumerate(self.train_set):

                # assign data
                keypoints_2D = data['keypoints_2D'].to(self.device)
                batch_size = keypoints_2D.shape[0]
                mask = data['mask_2D'].to(self.device)
                pc = data['pc'].to(self.device)
                intrinsics = data['intrinsics'].to(self.device)
                origin = data['root'].to(self.device)
                closest_cp_idx = data['closest_cp_idx'].to(self.device)
                closest_cp_dist = data['closest_cp_dist'].to(self.device)
                # velocity = data['velocity'].to(self.device)
                # angular_velocity = data['angular_velocity'].to(self.device)
                # cam = data['cam'].to(self.device)

                pc = pc.transpose(2, 1)

                # clear gradients
                self.generator.zero_grad()

                # # scale pc between [0.9, 1.1]
                # scale_factors = (torch.rand((batch_size, 1), device=self.device) * (1.1 - 0.9)) + 0.9
                # pc = pc * scale_factors.view(-1, 1, 1)
                
                # # randomly rotate pc
                # angles = (torch.rand((batch_size, 1)) * 20) - 10
                # for j in range(batch_size):
                #     random_rot_mat_z = torch.tensor(Rot.from_euler(seq="z", angles=angles[j], degrees=True).as_matrix(), dtype=torch.float32).to(self.device)
                #     pc[j] = torch.matmul(random_rot_mat_z, pc[j]).squeeze()

                predictions_3D, trans_features = self.generator(keypoints_2D, pc)

                # direct re-projection without normalization
                # direct_projection, _, pc_test = self.get_proj_rot_preds(predictions_3D, mask, pc, intrinsics, origin, deterministic_angle=360, norm=True)
                # direct_projection = self.adaption_net(direct_projection, cam, velocity, angular_velocity)
                # err_direct_reprojection =  Metrics.masked_mpjpe(direct_projection, keypoints_2D, mask)

                pseudo_gt_3D = self.compute_pseudo_gt(pc.transpose(2, 1), closest_cp_dist, closest_cp_idx, intrinsics)

                err_pseudo = self.calculate_loss_pseudo_gt(predictions_3D, pseudo_gt_3D, self.lift_mask(mask), trans_features)

                err_pseudo_w = err_pseudo * self.pseudo_weight
                # err_direct_reprojection_w = err_direct_reprojection * self.direct_reprojection_weight

                err = err_pseudo_w  # + err_direct_reprojection_w
                err.backward()
                self.optimizer.step()

                # Output training stats
                if i % 75 == 0:
                    # logging.info(f"[{self.epoch+1}/{self.epochs}][{i}/{len(self.train_set)}]\terr_pseudo: {round(err_pseudo.item(),4)} | err_direct_reprojection: {round(err_direct_reprojection.item(),4)}")
                    logging.info(f"[{self.epoch+1}/{self.epochs}][{i}/{len(self.train_set)}]\terr_pseudo: {round(err_pseudo.item(),4)}")
                self.current_step += 1

            # save checkpoint and run evaluation
                if i % (2*75) == 0:
                    self.generator.eval()
                    self.epoch_end_time = time.time()
                    self.validate()
                    self.losses.extend(self.losses_tmp)
                    self.generator.train()

            # vis some samples
            # if (self.epoch+1) % 5 == 0:
            #     self.generator.to('cpu')
            #     self.visualizer.vis_samples(num_samples=3, training=str(self.epoch))
            #     self.visualizer.vis_predictions_from_training(keypoints_2D, predictions_3D, self.epoch, keys, self.lift_mask(mask), pc, intrinsics, origin, num_samples=5)
            #    self.generator.to(self.device)

            # lr decay generator
            self.lr = self.lr*self.lr_decay_factor
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

        save_path = self.run_paths['path_ckpts_train'] + "/final_model"
        torch.save(self.generator.state_dict(), save_path)
        logging.info(
            f'Finsihed with training. Logs can be found at {self.run_paths["path_model_id"]} ')
        return self.generator, self.discriminator

    def validate(self):
        """Validation procedure of the self-supervised training routine"""

        self.train_loss = 0.0
        step_losses = []

        with torch.no_grad():
            self.generator.eval()
            for data in self.val_set:
                batch_dim = data['keypoints_2D'].shape[0]
                keypoints_2D = data['keypoints_2D'].to(self.device)
                keypoints_3D = data['keypoints_3D'].to(self.device)
                pc = data['pc'].to(self.device).transpose(2, 1)

                preds_3D, _ = self.generator(keypoints_2D, pc)

                step_loss = Metrics.masked_mpjpe(preds_3D, keypoints_3D, data['mask_3D'])
                step_losses.append(step_loss * batch_dim)
            self.val_loss = sum(step_losses)/len(self.val_set.dataset)
            self.print_val_stats_and_save_model()
            if self.wandb:
                self.push_wandb_val()
            self.previous_losses.append(self.train_loss)
            sys.stdout.flush()
        self.generator.train()

    def push_wandb_train(self,):
        wandb_dict = {
            'step': self.current_step,
            'epoch': self.epoch
        }
        wandb.log(
            wandb_dict
        )

    def push_wandb_val(self):
        """Push all data to Weights and Biases."""
        wandb_dict = {
            'best_val_loss': self.best_val_loss.detach().item(),
            'Validation MPJPE': self.val_loss.detach().item(),
            'step': self.current_step,
            'epoch': self.epoch
        }
        wandb.log(
            wandb_dict
        )

    def compute_pseudo_gt(self, pc, closest_cp_dist, closest_cp_idx, intrinsics):
        """
        Compute pseudo ground truth (using weighted 2D projection) for 3D keypoint estimation

        Args:
            pc (torch tensor): Point cloud.
            closest_cp_dist (torch tensor): 2D distances from the nearby_points to the actual 2D keypoint annotation/detection.
            closest_cp_idx (torch tensor): Indices of the pc points projections that are closest to the keypoint.
            intrinsics (torch.tensor): Camera intrinsics from the Waymo Open Dataset Camera.
        """
        batch_size, num_joints, num_close_points = closest_cp_dist.shape
        nearby_points_batch = torch.zeros((batch_size, num_joints, num_close_points, 3)).to(self.device)

        # get nearby points
        for i in range(batch_size):
            sample_pc = pc[i]
            sample_id = closest_cp_idx[i]
            nearby_points = sample_pc[sample_id.flatten().long()].reshape(closest_cp_idx.shape[1], -1, 3)
            nearby_points_batch[i] = nearby_points

        pseudo_weights = self.compute_pseudo_weights(closest_cp_dist, closest_cp_idx, nearby_points_batch)

        pseudo_gt = torch.sum(pseudo_weights*nearby_points_batch.permute(-1, 0, 1, 2), dim=-1)

        return pseudo_gt.permute(1, 2, 0)

    def calculate_loss_pseudo_gt(self, predictions, pseudo_3D, mask_3D, trans_features=None):
        loss = 0
        if self.loss_types["masked_mpjpe"]:
            loss += Metrics.masked_mpjpe(predictions, pseudo_3D, mask_3D)
        if self.loss_types["l1"]:
            loss += self.loss_types["l1"] * Metrics.masked_l1(predictions, pseudo_3D, mask_3D)
        if self.loss_types["bone_length"]:
            loss += self.loss_types["bone_length"] * Metrics.bone_length_symmetry(predictions)
        if self.loss_types["feature_transform_reguliarzer"]:
            loss += self.loss_types["feature_transform_reguliarzer"] * Metrics.feature_transform_reguliarzer(trans_features)

        return loss

    def get_proj_rot_preds(self, predictions, mask_3D, pc, intrinsics, origin=None, norm=True, deterministic_angle=0):
        """
        Rotate 3D prediction and project it to an image plane

        Args:
            predictions (torch tensor): 3D predictions of the model
            mask (torch tensor): Mask of not labeled keypoints - maybe not needed once all joints are predicted
            pc (torch.tensor): Sampled point cloud data
            norm (bool): Normalize results after projection
            intrinsics (torch.tensor): Camera intrinsics from waymo
            origin (torch.tensor): Origin of the sensor frame. If None, class extrinsics are used
            deterministic (int, optional): Degrees for a deterministic rotation angle (0,360] - default is False (stochastic)
                                           NOTE: For 0 Rotation use 360 degree (0 will create random rotation)

        Returns:
            tuple of torch.tensors: projection of rotated keypoints, rotated keypoints,  rotated point cloud (pc)
        """
        device = predictions.device

        # check if homogeneous extension matches (needed for last batch which contains less elements and visualization)
        if predictions.shape[0] != self.homogeneous_extension.shape[0]:
            self.adapt_hom_extension(predictions.shape[0])

        rot_pc, predictions_rot = self.random_rotation_3D(predictions, pc, deterministic_angle=deterministic_angle, device=predictions.device)

        if origin is not None:
            origin_unsqueezed = origin.unsqueeze(1).repeat(1, self.generator.num_joints, 1)
            predictions_rot = torch.add(predictions_rot, origin_unsqueezed)

            f_u = intrinsics[:, 0, 0]
            f_v = intrinsics[:, 1, 1]
            c_u = intrinsics[:, 0, 2]
            c_v = intrinsics[:, 1, 2]

            u_d = - predictions_rot[:, :, 1]/predictions_rot[:, :, 0]
            v_d = - predictions_rot[:, :, 2]/predictions_rot[:, :, 0]

            u_d = u_d * f_u.unsqueeze(dim=-1) + c_u.unsqueeze(dim=-1)
            v_d = v_d * f_v.unsqueeze(dim=-1) + c_v.unsqueeze(dim=-1)

            projected_predictions = torch.cat((u_d.unsqueeze(dim=0), v_d.unsqueeze(dim=0))).permute(1, 2, 0).to(torch.float32)

        else:
            homogeneous_predictions = torch.cat((predictions_rot, self.homogeneous_extension), dim=-1).to(device)
            full_projection = torch.matmul(intrinsics.to(torch.float32), self.extrinsics.to(device))
            hom_projected_predictions = torch.matmul(full_projection, homogeneous_predictions.transpose(1, 2)).transpose(1, 2)
            projected_predictions = hom_projected_predictions[:, :, :2] / torch.unsqueeze(hom_projected_predictions[:, :, -1], -1)

        # normalize_keypoints2D_batch(projected_predictions, mask_3D[:, :, :2], self.batch_size)
        if norm:
            # if origin is not None:
            #     projected_predictions = normalize_keypoints2D_batch_by_depth(projected_predictions, mask_3D[:, :, :2], origin[:, 0])
            # else:
            projected_predictions = normalize_keypoints2D_batch(projected_predictions, mask_3D[:, :, :2])
        projected_predictions[~mask_3D[:, :, 0]] = torch.tensor([0, 0], dtype=torch.float32).to(device)

        return projected_predictions, predictions_rot, rot_pc

    def random_rotation_3D(self, keypoints, pc, deterministic_angle=0, z_rot=(0, +360), device="cpu"):
        """
        Rotation of 3D Points in space

        Args:
            pc (torch tensor): 3D point cloud to be rotated [batch x 3].
            keypoints (torch tensor): 3D keypoints to be rotated.
            z_rot, (tuple, optional): Rotation boundaries for specific axis.
            device (str): Device
            deterministic (int, optional): Degrees for a deterministic rotation angle (0,360] - default is False (stochastic).
                                           NOTE: For 0 Rotation use 360 degree (0 will create random rotation).

        Returns:
            tuple: (rotated point cloud tensor, rotated keypoints tensor)
        """
        rot_keypoints = None
        rot_pc = None

        if deterministic_angle:
            self.current_random_angle = deterministic_angle
        else:
            self.current_random_angle = random.randint(z_rot[0], z_rot[1])
        
        if rot_keypoints is not None:
            device = rot_keypoints.device
        elif rot_pc is not None:
            device = rot_pc.device
        else:
            device = "cpu"
            
        random_rot_mat_z = torch.tensor(Rot.from_euler(seq="z", angles=self.current_random_angle, degrees=True).as_matrix(), dtype=torch.float32).to(device)

        if pc is not None:
            rot_pc = torch.matmul(random_rot_mat_z, pc)
        if keypoints is not None:
            rot_keypoints = torch.matmul(random_rot_mat_z, keypoints.transpose(1, 2))

        return rot_pc, rot_keypoints.transpose(1, 2),

    def setup_cam_params(self):
        """
        Setup projection matrices for self-supervised learning
        """

        # DEPRECATED (USE WAYMO INTRINSICS NOW)
        # self.intrinsic = torch.tensor([[250, 0, 1920], [0, 250, 1080], [0, 0, 1]], dtype=torch.float32)
        # self.projection_matrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)

        # rotate -90 degree around x-axis, then -90 degree around y-axis
        # -> needed for correct projection to sensor coordinate system (x to the right, y down)

        angle = math.pi/2
        rot_x = torch.tensor([
            [1,         0,               0],
            [0, math.cos(-angle), -math.sin(-angle)],
            [0, math.sin(-angle), math.cos(-angle)]])
        rot_y = torch.tensor([
            [math.cos(-angle),  0, math.sin(-angle)],
            [0,                 1,       0],
            [-math.sin(-angle), 0, math.cos(angle)]])

        r = rot_y @ rot_x
        t = torch.tensor([[15], [0], [0]], dtype=torch.float32)
        hom = torch.tensor([[0, 0, 0, 1]])
        r_t = -r @ t
        trans = torch.cat((r, r_t), dim=1)

        self.extrinsics = torch.cat((trans, hom), dim=0)
        # DEPRECATED (USE WAYMO INTRINSICS NOW)s
        # proj = self.projection_matrix @ self.extrinsics
        # full_projection = self.intrinsic  @ proj
        # self.full_projection = torch.cat(self.batch_size*[torch.flatten(full_projection)]).reshape(self.batch_size, 3, 4)
        self.basic_homogeneous_extension = torch.ones(self.generator.num_joints, 1)
        self.homogeneous_extension = torch.ones(self.batch_size, self.generator.num_joints, 1)

    def adapt_hom_extension(self, batch_size):
        self.homogeneous_extension = self.basic_homogeneous_extension.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)

    def compute_pseudo_weights(self, distances, indices, nearby_points, temperature=1):

        # find mean of the close points
        mean_points = torch.mean(nearby_points, dim=2)
        dist_from_mean = torch.norm((nearby_points.permute(2, 0, 1, 3) - mean_points).permute(1, 2, 0, 3), dim=-1)

        weights_3D = torch.nn.functional.softmax(dist_from_mean/temperature/2, dim=-1)
        weights_2D = torch.nn.functional.softmax(distances/temperature, dim=-1)

        return self.pseudo_weights_3D*weights_3D + (1-self.pseudo_weights_3D)*weights_2D

    def bb_error(self, predictions, bb_3d_expansion):
        """
        Penalty for generation of to large values.

        Args:
            predictions (torch.tensor): 3D predictions from generator.
            bb_3d_expansion (torch.tensor): Max expansion of 3D-bb.

        Returns:
            error: Scalar penalty value.
        """
        diff = abs(predictions.transpose(0, -1)) - bb_3d_expansion/2
        return torch.sum(diff[diff > 0])

    @staticmethod
    def lift_mask(mask_2D):
        """
        Lift mask from 2D -> 3D.
        """
        return torch.cat((mask_2D, mask_2D), dim=-1)[:, :, :-1]

    @staticmethod
    def get_splitted_batch_sizes(full_batch_size):
        if (full_batch_size % 2) == 0:
            return int(full_batch_size/2), int(full_batch_size/2)
        else:
            return int(full_batch_size/2 + 0.5), int(full_batch_size/2 - 0.5)
