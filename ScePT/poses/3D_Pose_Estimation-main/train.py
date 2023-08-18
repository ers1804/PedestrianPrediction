import sys
import gin
import logging
import torch
import wandb
import numpy as np

from abc import ABC, abstractmethod
from evaluation.metrics import Metrics
import evaluation.metrics_waymo as metrics_waymo

from configs.constants import JOINT_NAMES


class Trainer(ABC):
    """
    Parent class of SupervisedTrainer and SelfSupervisedTrainer
    """

    train_set = None
    test_set = None
    val_set = None

    JOINT_NAMES = JOINT_NAMES

    def __init__(self, train_set, val_set, test_set, run_paths, epochs,
                 device, wandb, waymo_evaluation, type):

        # type should be either 'supervised' or 'weakly_supervised'
        self.type = type.lower()

        self.batch_size = gin.query_parameter('load.batch_size')
        self.val_set = val_set
        self.train_set = train_set
        self.test_set = test_set
        self.run_paths = run_paths
        self.waymo_evaluation = waymo_evaluation

        self.loss, self.val_loss = 0.0, 0.0
        self.current_step = 0
        self.previous_losses = []
        self.best_val_loss = float("Inf")
        self.epochs = epochs

        self.wandb = wandb
        self.device = self.get_cuda_device(device)

        self.val_loss = np.inf
        self.train_loss = np.inf
        self.test_loss = np.inf

    @abstractmethod
    def train(self):
        """
        Training routine

        Every trainer object needs a specific training routine.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Training routine

        Every trainer object needs a specific validation routine.
        """
        pass

    def eval_on_all_3D_poses(self, all_data):
        self.test_set = all_data
        logging.info('CHANGING FROM NORMAL TEST SET TO ALL AVAILABLE 3D SAMPLES!')
        self.eval(reload_best_model=True)

    def eval(self, reload_best_model=False):
        """
        Test trained models.

        Evaluation routine is the same for all models and trainings.
        """

        logging.info('#'*39)
        logging.info('#' * 7 + "   Starting Evaluation   " + '#'*7)
        logging.info('#'*39)
        logging.info('')

        self.device = 'cpu'
        self.generator.to(self.device)

        logging.info(f'Testing data is stored in {self.run_paths["path_logs_eval"]}')
        load_dir = self.run_paths['path_ckpts_train'] + "/best_model"
        if reload_best_model:
            logging.info(f'Loading best model from: {load_dir}')
            self.generator.load_state_dict(torch.load(load_dir))
        else:
            logging.info(f"Proceeding with current model. NOTE: Best model is stored in {load_dir}")
        if self.waymo_evaluation:
            all_metrics = metrics_waymo.create_combined_metric(metrics_waymo.DEFAULT_CONFIG_LASER)

        with torch.no_grad():
            self.generator.eval()

            # Reset loss
            self.loss = 0
            step_losses = []
            mpjpes_per_joint = []

            for data in self.test_set:
                batch_dim = data['keypoints_2D'].shape[0]
                keypoints_2D = data['keypoints_2D'].to(self.device)
                keypoints_3D = data['keypoints_3D'].to(self.device)

                pc = data['pc'].to(self.device).transpose(2, 1)
                # weakly_supervised
                if self.type == "weakly_supervised":
                    tmp_preds_keypoints3D, _ = self.generator(keypoints_2D, pc)
                # supervised
                else:
                    if self.generator.type == "point_cloud":
                        tmp_preds_keypoints3D, trans_features = self.generator(pc)
                    elif self.generator.type == "keypoints":
                        tmp_preds_keypoints3D = self.generator(keypoints_2D)
                    elif self.generator.type == "fusion":
                        tmp_preds_keypoints3D, _, _ = self.generator(pc, keypoints_2D, gt=(keypoints_3D, data['mask_3D']))
                    else:
                        logging.error('Model input not defined properly.')
                        sys.exit(1)

                step_loss = Metrics.masked_mpjpe(tmp_preds_keypoints3D, keypoints_3D, data['mask_3D'])
                mpjpe_per_joint = Metrics.maksed_jointwise_mpjpe(tmp_preds_keypoints3D, keypoints_3D, data['mask_3D'])
                if self.waymo_evaluation:
                    Metrics.waymo_eval(all_metrics, tmp_preds_keypoints3D, data)
                step_losses.append(step_loss * batch_dim)
                mpjpes_per_joint.append(mpjpe_per_joint * batch_dim)

            self.test_loss = sum(step_losses)/len(self.test_set.dataset)
            self.test_mpjpe_per_joint = sum(mpjpes_per_joint)/len(self.test_set.dataset)
            self.waymo_eval_results = all_metrics.result()

            for index, name in enumerate(self.JOINT_NAMES):
                logging.info(f"Test MPJPE {name}: {round(self.test_mpjpe_per_joint[index].item()*100,3)}cm")

            logging.info(f'Test MPJPE avg: {round(self.test_loss.item()*100,3)}cm.')
            if self.waymo_evaluation:
                logging.info('')
                logging.info('Waymo Metrics:')
                results = sorted(self.waymo_eval_results.items(), key=lambda e: e[0])
                for name, tensor in results:
                    logging.info(f'{name:20s}: {tensor.numpy():.5f}')

                # create latex output
                groups = {
                    #   "All": {"MPJPE": results[0],
                    #             "OKS_AP": results[8],
                    #             "PCK@30":  results[99]
                    #         },

                    "Ankles": {"MPJPE": results[1],
                               "OKS_AP": results[19],
                               "PCK@30":  results[105]
                               },

                    "Elbows": {"MPJPE": results[2],
                               "OKS_AP": results[30],
                               "PCK@30":  results[111]
                               },
                    "Head": {"MPJPE": results[3],
                             "OKS_AP": results[41],
                             "PCK@30":  results[117]
                             },

                    "Hips": {"MPJPE": results[4],
                             "OKS_AP": results[52],
                             "PCK@30":  results[123]
                             },

                    "Knees": {"MPJPE": results[5],
                              "OKS_AP": results[63],
                              "PCK@30":  results[129]
                              },
                    "Shoulders": {"MPJPE": results[6],
                                  "OKS_AP": results[74],
                                  "PCK@30":  results[135]
                                  },

                    "Wrists": {"MPJPE": results[7],
                               "OKS_AP": results[85],
                               "PCK@30":  results[141]
                               },
                }
                logging.info("")
                logging.info("")
                logging.info('LATEX TEMPLATE:')
                logging.info("\\begin{table}")
                logging.info("\\centering")
                logging.info("\\begin{tabular}{ |c|c|c|c|  }")
                logging.info("\\hline")
                logging.info("\\textbf{Keypoint} & \\gls{mpjpe}  [\\unit[]{cm}] $\\downarrow$ & \\gls{oks}/ACC  [\\unit[]{\%}] $\\uparrow$ & \\ gls{pck}@30cm  [\\unit[]{\%}] $\\uparrow$  \\\\")
                logging.info("\\hline")
                for group, values in groups.items():
                    logging.info(f'{group}  & {values["MPJPE"][1].numpy():.4f} &   {values["OKS_AP"][1].numpy():.4f}   &  {values["PCK@30"][1].numpy():.4f} \\\\')
                logging.info("\\hline")
                logging.info(f"All   & {results[0][1].numpy():.4f} &   {results[8][1].numpy():.4f}   &   {results[99][1].numpy():.4f} \\\\")
                logging.info("\\hline")
                logging.info("\\end{tabular}")
                logging.info("\\caption{ CAPTION}")
                logging.info("\\label{tab:TODO}")
                logging.info("\\end{table}")
                logging.info("")
                logging.info("")

            if self.wandb:
                wandb.save(self.run_paths['path_logs_train'])
                wandb.save(self.run_paths['path_logs_eval'])
                wandb.save(self.run_paths['path_flags'])
                wandb.save(self.run_paths['path_gin'])

            sys.stdout.flush()

            return self.test_loss.item()*100

    def print_train_start(self):
        """Print short message whenever training is started."""
        logging.info('#'*39)
        logging.info('#' * 8 + "   Starting Training   " + '#'*8)
        logging.info('#'*39)
        logging.info('')
        logging.info(
            f'All relevant data from this run is stored in:{self.run_paths["path_model_id"]}')
        logging.info("")

    def print_val_stats_and_save_model(self):
        """Print statistics from validation run and save model if better than previous one."""
        logging.info("")
        logging.info("====== Validation ======")
        logging.info("")
        logging.info(f"Global step:         {self.current_step}")
        logging.info(f"Epoch Training Time: {int(self.epoch_end_time-self.epoch_start_time)} sec.")
        if self.type == 'supervised':
            logging.info(f"Learning rate:      {self.lr}")
            logging.info(f"Train loss epoch avg:     {round(self.train_loss/self.samples_counter,4)}")
        else:
            # TODO:
            pass
        logging.info("--------------------------")
        logging.info(f"Val loss:           {round(float(self.val_loss),4)}")
        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            save_path = self.run_paths['path_ckpts_train'] + \
                "/best_model"
            logging.info(f"Saving new best model to: {save_path}")
            torch.save(self.generator.state_dict(), save_path)
        logging.info("========================")
        logging.info('')

    @ staticmethod
    def log_grads_wandb(model):
        """Display gradients of model in wandb"""
        wandb.watch(model, log='all')

    @ staticmethod
    def get_cuda_device(device):
        "Get Cuda device if available."
        if device == -1 or device == "cpu":
            device = torch.device("cpu")
            logging.warning("Running on cpu...")
        else:
            device = torch.device("cuda:" + f"{device}" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(device)
            logging.info(f"Running on device {device.type}:{torch.cuda.current_device()} - {torch.cuda.get_device_name(device.index)} ")
        return device
