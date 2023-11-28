import gin
import logging
import wandb
import os
import torch
import random
import sys
import multiprocessing
from absl import app, flags
from train_supervised import SupervisedTrainer
from train_unsupervised import SelfSupervisedTrainer
from vis import Visualizer
from tune import Tuner
from utils import utils_params, utils_misc, get_model
from input_pipeline import dataset

FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
multiprocessing.Queue(1000)

# import tensorflow as tf 
# tf.config.run_functions_eagerly(True)
# tf.config.experimental_run_functions_eagerly(True)

# Set Seed
random.seed(42)
torch.manual_seed(42)

flags.DEFINE_string("model_id", "", "Name or load a run.")
flags.DEFINE_string('run', 'all', 'Specify whether to train or evaluate a model (all, vis).')
flags.DEFINE_string("dataset", "waymo_3d_2d_projections_supervised", "Dataset to train on. NOTE: Weakly-supervised datasets directly imply the learning paradigm. Was: waymo_alphapose_weakly_supervised")
flags.DEFINE_string('config', '/home/erik/ScePT/ScePT/poses/3D_Pose_Estimation-main/configs/config.gin', 'Config file that should be loaded.')
flags.DEFINE_integer("device", 0, "Cuda device to use (-1 for cpu).")
flags.DEFINE_string("model_type", "Lidar2dKeypointFusion", "Model type to be used for prediction.")
flags.DEFINE_string("suffix", "", "suffix extension for current run.")
flags.DEFINE_string("tune", "", "which tune config to use.")
flags.DEFINE_integer("tune_count", 100, "number of sweeps to do.")
flags.DEFINE_boolean('wandb', False, "Specify if the run should be logged and sent to Weights & Biases.")
flags.DEFINE_string("wandb_project", "ALL", "Project name for Weights & Biases logging.")
flags.DEFINE_string("sweep_id", "", "resume already existing wandb sweep by passing the id.")
flags.DEFINE_list("overwrite_gin_args", "", "Overwrite gin parameters via command line. Form: 'key1,value1, key2,value2'")

# AlphaPose
################
flags.DEFINE_boolean("use_alpha", True, "Use Alphapose to generate 2D keypoints during training.")
flags.DEFINE_string("alpha_cfg", "/home/erik/gitprojects/AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml", "AlphaPose config file.")
flags.DEFINE_string("alpha_checkpoint", "/home/erik/gitprojects/AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth", "AlphaPose checkpoint file.")
flags.DEFINE_list("gpus", ["0"], "List of GPUs to use for AlphaPose.")
flags.DEFINE_string("detector", "yolox-l", "Detector to use for AlphaPose.")
flags.DEFINE_integer("detbatch", 1, "Batch size for AlphaPose detector.")
flags.DEFINE_integer("qsize", 32, "Queue size for AlphaPose detector.")
flags.DEFINE_integer("posebatch", 32, "Batch size for AlphaPose pose estimation.")
flags.DEFINE_boolean("sp", True, "Use single process for Pytorch (AlphaPose).")
flags.DEFINE_boolean("tracking", False, "Use tracking for AlphaPose.")
flags.DEFINE_boolean("save_img", False, "Save images for AlphaPose.")
flags.DEFINE_boolean("vis", False, "Visualize images for AlphaPose.")
flags.DEFINE_boolean("pose_flow", False, "Use pose flow for AlphaPose.")
flags.DEFINE_boolean("pose_track", False, "Use pose track for AlphaPose.")
flags.DEFINE_integer("min_box_area", 0, "Minimum box area for AlphaPose.")
################

flags.register_validator('tune',
                         lambda value: value.lower() in ['', 'waymo_alphapose_weakly_supervised', 'waymo_3d_2d_projections_supervised', 'waymo_3d_lidar_supervised', 'waymo_2d_labels_supervised', 'waymo_weakly_supervised'],
                         message="--tune has to be from ['', 'waymo_alphapose_weakly_supervised', 'waymo_3d_2d_projections_supervised', 'waymo_3d_lidar_supervised', 'waymo_2d_labels_supervised', 'waymo_weakly_supervised']")
flags.register_validator('model_type',
                         lambda value: value.lower() in ['SimpleLiftingModel'.lower(), 'PointNet'.lower(), 'Lidar2dKeypointFusion'.lower()],
                         message="--model has to be from ['SimpleLiftingModel', 'PointNet', 'Lidar2dKeypointFusion']")
flags.register_validator('dataset',
                         lambda value: value.lower() in ['waymo_alphapose_weakly_supervised'.lower(), 'waymo_weakly_supervised'.lower(), 'waymo_3d_2d_projections_supervised'.lower(), 'waymo_2d_labels_supervised'.lower()],
                         message="--dataset has to be from ['waymo_2d_labels_supervised', 'waymo_3d_2d_projections_supervised', 'waymo_weakly_supervised']")


def weakly_supervised(FLAGS, run_paths, load_model):
    """
    Starts self-supervised routine (train, eval, vis or all together)
    Args:
        FLAGS : Flags handed over to the script at starting time
        run_paths (dict): Paths to store output (vis, metrics, files, models, ...)
        load_model (bool): Boolean if previously trained model is to be loaded
    """
    # get data & model
    ds_train, ds_test, ds_val = dataset.load(name=FLAGS.dataset)
    generator, discriminator = get_model.get_model(FLAGS)

    # load pretrained model
    # TODO: Also load discriminator
    if load_model:
        logging.info(f"Loading model weights from {FLAGS.model_id}/ckpts/best_model")
        generator.load_state_dict(torch.load(FLAGS.model_id + "/ckpts/best_model"))

    # TODO: implement model loading, vis, etc. as for supervised method
    if FLAGS.run.lower() == "vis":
        if not load_model:
            logging.error('Can not create vis without pretrained model... Exiting now...')
            sys.exit(0)
        trainer = SelfSupervisedTrainer(generator, discriminator, ds_train, ds_val, ds_test, run_paths)
        visualizer = Visualizer(trainer, FLAGS.dataset)

        visualizer.vis_samples()
        # visualizer.vis_complete_images()
    elif FLAGS.run.lower() == "eval":
        if not load_model:
            logging.error('Can not create vis without pretrained model... Exiting now...')
            sys.exit(0)
        trainer = SelfSupervisedTrainer(generator, discriminator, ds_train, ds_val, ds_test, run_paths, wandb=FLAGS.wandb)
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        trainer.eval()
        # delete datasets not used anymore
        del ds_train
        del ds_val
        ds_test_all = dataset.load(name='all_3D_samples')
        trainer.eval_on_all_3D_poses(ds_test_all)

    elif FLAGS.run.lower() == "all":
        # train
        trainer = SelfSupervisedTrainer(generator, discriminator, ds_train, ds_val, ds_test, run_paths,
                                        device=FLAGS.device, wandb=FLAGS.wandb)
        generator, discriminator = trainer.train()
        # eval
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        # TODO: CHANGE AGAIN reload_best_model=True
        trainer.eval(reload_best_model=True)
        # delete datasets not used anymore
        del ds_train
        del ds_val
        ds_test_all = dataset.load(name='all_3D_samples')
        trainer.eval_on_all_3D_poses(ds_test_all)
        # vis results
        trainer = SelfSupervisedTrainer(generator, discriminator, None, None, ds_test, run_paths)
        visualizer = Visualizer(trainer, FLAGS.dataset)
        visualizer.vis_samples()
        # visualizer.vis_complete_images()


def supervised(FLAGS, run_paths, load_model):
    """
    Starts supervised routine (train, eval, vis or all together)
    Args:
        FLAGS : Flags handed over to the script at starting time
        run_paths (dict): Paths to store output (vis, metrics, files, models, ...)
        load_model (bool): Boolean if previously trained model is to be loaded
    """

    # get data & model
    ds_train, ds_test, ds_val = dataset.load(name=FLAGS.dataset)
    model = get_model.get_model(FLAGS, supervised=True)

    # load pretrained model
    if load_model:
        logging.info(f"Loading model weights from {FLAGS.model_id}/ckpts/best_model")
        model.load_state_dict(torch.load(FLAGS.model_id + "/ckpts/best_model"))

    if FLAGS.run.lower() == "eval":
        if not load_model:
            logging.error('Can not create vis without pretrained model... Exiting now...')
            sys.exit(0)
        trainer = SupervisedTrainer(model, ds_train, ds_val, ds_test, run_paths, wandb=FLAGS.wandb)
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        trainer.eval()

    elif FLAGS.run.lower() == "vis":
        if not load_model:
            logging.error('Can not create vis without pretrained model... Exiting now...')
            sys.exit(0)
        trainer = SupervisedTrainer(model, ds_train, ds_val, ds_test, run_paths)
        visualizer = Visualizer(trainer, FLAGS.dataset)

        visualizer.vis_samples()
        # visualizer.vis_complete_images()

    elif FLAGS.run.lower() == "all":
        # train
        trainer = SupervisedTrainer(model, ds_train, ds_val, ds_test, run_paths,
                                    device=FLAGS.device, wandb=FLAGS.wandb, flags=FLAGS)
        model = trainer.train()
        # eval
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, del_prev_handler=True)
        trainer.eval(reload_best_model=True)
        # vis results
        trainer = SupervisedTrainer(model, ds_train, ds_val, ds_test, run_paths, flags=FLAGS)
        visualizer = Visualizer(trainer, FLAGS.dataset)
        visualizer.vis_samples()
        # visualizer.vis_complete_images()


def main(argv):

    if FLAGS.tune:
        tuner = Tuner(FLAGS)
        tuner.tune()
        tuner.save_scores()
        sys.exit(0)

    # generate folder structures
    run_paths, load_model = utils_params.gen_run_folder(FLAGS.suffix, FLAGS.model_id)

    if load_model:
        logging.info('model_id already exits. Overwriting gin inputs....')
        gin.parse_config_file(run_paths['path_gin'])
        utils_params.update_flags(run_paths['path_flags'], FLAGS)
    else:
        # setup gin-config and save configs
        gin.parse_config_file(FLAGS.config)
        # update command line inputs
        if FLAGS.overwrite_gin_args:
            logging.info('Overwriting gin-config with specified input parameters...')
            for key, value in zip(FLAGS.overwrite_gin_args[0::2], FLAGS.overwrite_gin_args[1::2]):
                var_type = type(gin.query_parameter(key))
                value = var_type(value)
                gin.bind_parameter(key, value)
                print(f"Setting {key} to {value} (type:{type(value)})")
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        FLAGS.append_flags_into_file(run_paths['path_flags'])

    # set logging to training by default
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # init weights & biases
    if FLAGS.wandb and not (FLAGS.run == "visual" or FLAGS.run == "eval"):
        name = run_paths['path_model_id'].split(os.sep)[-1]
        wandb.init(project=FLAGS.wandb_project, name=name,
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # check supervision
    supervision = False if "weakly_supervised" in FLAGS.dataset.lower() else True
    logging.info(f"Model type: {FLAGS.model_type}")
    if supervision:
        supervised(FLAGS, run_paths, load_model)
    else:
        weakly_supervised(FLAGS, run_paths, load_model)


if __name__ == '__main__':
    app.run(main)
