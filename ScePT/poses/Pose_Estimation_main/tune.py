import wandb
import logging
import sys
import gin
import os
import datetime
from models.weakly_supervised.discriminator import Discriminator

from train_supervised import SupervisedTrainer
from train_unsupervised import SelfSupervisedTrainer
from vis import Visualizer
from input_pipeline import dataset
from configs.sweeps.supervised_sweeps import waymo_3d_2d_projections_supervised, waymo_3d_projections_fusion, waymo_3d_lidar_supervised, waymo_2d_labels_supervised, waymo_2d_labels_fusion
from configs.sweeps.weakly_supervised_sweeps import waymo_weakly_supervised, waymo_alphapose_weakly_supervised
from utils import utils_params, utils_misc, get_model
from pandas import DataFrame as df


class Tuner():
    """Class for hyperparameter optimization"""

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.logs = df(columns=['run_id', 'Avg Test MPJPE'])
        self.counter = 0

    def train_func(self):

        # always log to wandb if tuning
        self.FLAGS.wandb = True

        # check supervision
        weakly_supervised = True if "weakly_supervised" in self.FLAGS.tune.lower() else False
        trainer_type = "SelfSupervisedTrainer" if weakly_supervised else "SupervisedTrainer"

        with wandb.init() as run:
            gin.clear_config()
            gin.parse_config_file(self.FLAGS.config)
            loss_types = gin.query_parameter(f'{trainer_type}.loss_types')
            gin.clear_config()

            # Hyperparameters
            bindings = []

            for key, value in run.config.items():
                if f"{trainer_type}.loss_types".lower() in key.lower():
                    loss_types[key.split(".")[-1]] = value
                else:
                    if isinstance(value, str):
                        bindings.append(f"{key}='{value}'")
                    else:
                        bindings.append(f"{key}={value}")

            bindings.append(f'{trainer_type}.loss_types={loss_types}')

            # setup run folders and name
            run_paths, _ = utils_params.gen_run_folder(self.FLAGS.suffix, self.FLAGS.model_id)
            run.name = run_paths['path_model_id'].split(os.sep)[-1]

            # set loggers
            utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

            # parse config file and bindings
            gin.parse_config_files_and_bindings([self.FLAGS.config], bindings)
            utils_params.save_config(run_paths['path_gin'], gin.config_str())
            self.FLAGS.append_flags_into_file(run_paths['path_flags'])

            # get data & model
            ds_train, ds_test, ds_val = dataset.load()

            if weakly_supervised:
                generator, discriminator = get_model.get_model(self.FLAGS)
                # train
                trainer = SelfSupervisedTrainer(generator, discriminator, ds_train, ds_val, ds_test, run_paths,
                                                device=self.FLAGS.device, wandb=self.FLAGS.wandb)
                generator, discriminator = trainer.train()
                # eval
                utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
                trainer.eval(reload_best_model=False)
                # vis results
                trainer = SelfSupervisedTrainer(generator, discriminator, ds_train, ds_val, ds_test, run_paths)
                visualizer = Visualizer(trainer, self.FLAGS.dataset)
                visualizer.vis_samples()
                visualizer.vis_complete_images()
            else:
                model = get_model.get_model(self.FLAGS)
                # train
                trainer = SupervisedTrainer(model, ds_train, ds_val, ds_test, run_paths, device=self.FLAGS.device, wandb=True)
                model = trainer.train(weakly_supervised)

                # eval
                utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
                test_score = trainer.eval(reload_best_mode=True)
                d = df([[run_paths['path_model_id'].split(os.sep)[-1], test_score]], columns=['run_id', 'Avg Test MPJPE'])
                self.logs = self.logs.append(d, ignore_index=True)
                self.counter += 1

                # vis results
                # trainer = Trainer(model, ds_train, ds_val, ds_test, run_paths)
                # visualizer = Visualizer(trainer)

                # visualizer.vis_samples()
                # visualizer.vis_complete_images()

            utils_misc.remove_all_handlers()

    def save_scores(self):
        """Save tune MPJPEs to file"""
        sorted_logs = self.logs.sort_values(by=['Avg Test MPJPE'])
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'runs'))
        sorted_logs.to_csv(path + "/" + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + "_" + self.project + ".csv", sep=";")

    def tune(self):
        """Tune Models"""

        if self.FLAGS.sweep_id:
            sweep_id = self.FLAGS.sweep_id
            self.project = "SWEEP_" + self.FLAGS.tune.lower() + "_" + self.FLAGS.model_type
        else:
            if self.FLAGS.tune.lower() == "waymo_3d_lidar_supervised":
                if self.FLAGS.model_type.lower() == "PointNet".lower():
                    sweep_config = waymo_3d_lidar_supervised
                else:
                    logging.error("Please select point network to run this configuration (mode_type)! Exiting now...")
                    sys.exit(0)

            elif self.FLAGS.tune.lower() == "waymo_2d_labels_supervised":
                if self.FLAGS.model_type.lower() == "SimpleLiftingModel".lower():
                    sweep_config = waymo_2d_labels_supervised
                elif self.FLAGS.model_type.lower() == "Lidar2dKeypointFusion".lower():
                    sweep_config = waymo_2d_labels_fusion
                else:
                    logging.error("Please select correct lifting or fusion model properly! Exiting now...")
                    sys.exit(0)

            elif self.FLAGS.tune.lower() == "waymo_3d_2d_projections_supervised":
                if self.FLAGS.model_type.lower() == "Lidar2dKeypointFusion".lower():
                    sweep_config = waymo_3d_projections_fusion
                elif self.FLAGS.model_type.lower() == "SimpleLiftingModel".lower():
                    sweep_config = waymo_3d_2d_projections_supervised
                else:
                    logging.error("Please select correct lifting or fusion model properly! Exiting now...")
                    sys.exit(0)

            elif self.FLAGS.tune.lower() == "waymo_weakly_supervised":
                sweep_config = waymo_weakly_supervised
            elif self.FLAGS.tune.lower() == "waymo_alphapose_weakly_supervised":
                sweep_config = waymo_alphapose_weakly_supervised
            else:
                logging.warning("Could not find correct sweep config. Exiting now!")
                sys.exit(0)

            self.project = "SWEEP_" + self.FLAGS.tune.lower() + "_" + self.FLAGS.model_type
            sweep_id = wandb.sweep(sweep_config, project=self.project)

        logging.info(f'Starting sweep on W&B with configuration {self.FLAGS.tune}...')
        wandb.agent(sweep_id, function=self.train_func, project=self.project,
                    count=self.FLAGS.tune_count)
