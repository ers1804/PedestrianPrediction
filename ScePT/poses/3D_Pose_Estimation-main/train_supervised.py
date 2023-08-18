import sys
import gin
import time
import logging
import torch
import wandb
import torch.optim as optim
from evaluation.metrics import Metrics
from train import Trainer


@gin.configurable
class SupervisedTrainer(Trainer):
    """
    Class to train the implemented supervised models
    """

    def __init__(self, model, train_set, val_set, test_set, run_paths, epochs,
                 lr, lr_decay_factor, lr_step, loss_types={'masked_mpjpe': True},
                 device="cpu", wandb=False, waymo_evaluation=True):

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
