import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='../config/config.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)

parser.add_argument("--num_workers",
                    help="number of processes to spawn for dataset loading",
                    type=int,
                    default=0)

parser.add_argument("--indexing_workers",
                    help="number of processes to spawn for dataset indexing",
                    type=int,
                    default=0)

parser.add_argument("--local_rank",
                    help="local process number for distributed training",
                    type=int,
                    default=0)





# Model Parameters
parser.add_argument("--offline_scene_graph",
                    help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
                    type=str,
                    default='yes')

parser.add_argument("--dynamic_edges",
                    help="whether to use dynamic edges or not, options are 'no' and 'yes'",
                    type=str,
                    default='yes')

parser.add_argument("--edge_state_combine_method",
                    help="the method to use for combining edges of the same type",
                    type=str,
                    default='sum')

parser.add_argument("--edge_influence_combine_method",
                    help="the method to use for combining edge influences",
                    type=str,
                    default='attention')

parser.add_argument('--edge_addition_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're created",
                    type=float,
                    default=[0.25, 0.5, 0.75, 1.0]) # We don't automatically pad left with 0.0, if you want a sharp
                                                    # and short edge addition, then you need to have a 0.0 at the
                                                    # beginning, e.g. [0.0, 1.0].

parser.add_argument('--edge_removal_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're removed",
                    type=float,
                    default=[1.0, 0.0])  # We don't automatically pad right with 0.0, if you want a sharp drop off like
                                         # the default, then you need to have a 0.0 at the end.

parser.add_argument('--override_attention_radius',
                    action='append',
                    help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
                    default=[])

parser.add_argument('--incl_robot_node',
                    help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument('--map_encoding',
                    help="Whether to use map encoding or not",
                    action='store_true')

parser.add_argument('--augment',
                    help="Whether to augment the scene during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--scene_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_viz',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--no_edge_encoding',
                    help="Whether to use neighbors edge encoding",
                    action='store_true')

# Data Parameters
parser.add_argument("--nuscenes_path",
                    help="path to the nuscenes dataset",
                    type=str,
                    default='/home/erik/NAS/publicdatasets/nuscenes')

parser.add_argument("--data_dir",
                    help="what dir to look in for data",
                    type=str,
                    default='/home/erik/ScePT/experiments/processed')

parser.add_argument("--train_data_dict",
                    help="what file to load for training data",
                    type=str,
                    default='train.pkl')

parser.add_argument("--eval_data_dict",
                    help="what file to load for evaluation data",
                    type=str,
                    default='val.pkl')

parser.add_argument("--log_dir",
                    help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str,
                    default='../experiments/logs')
parser.add_argument("--trained_model_dir",
                    help="the direction of a particular trained model",
                    type=str,
                    default=None)

parser.add_argument("--iter_num",
                    help="The iteration number of the trained model",
                    type=int,
                    default=0)

parser.add_argument("--log_tag",
                    help="tag for the log folder",
                    type=str,
                    default='')

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

# Training Parameters
parser.add_argument("--learning_rate",
                    help="initial learning rate, default is whatever the config file has",
                    type=float,
                    default=None)

parser.add_argument("--lr_step",
                    help="number of epochs after which to step down the LR by 0.1, default is no step downs",
                    type=int,
                    default=None)

parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)

parser.add_argument('--eval_batch_size',
                    help='evaluation batch size',
                    type=int,
                    default=256)

parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--vis_every',
                    help='how often to visualize during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--save_every',
                    help='how often to save during training, never if None',
                    type=int,
                    default=1)
parser.add_argument("--video_dir",
                    help="name of the video for sim prediction",
                    type=str,
                    default="../videos")
parser.add_argument("--video_name",
                    help="name of the video for sim prediction",
                    type=str,
                    default=None)
parser.add_argument("--use_processed_data",
                    help="skip the offline processing to save time",
                    action='store_true')
# eval parameters
parser.add_argument("--eval_task",
                    help="what evaluate task to run",
                    default="eval_statistics")
# Pose mode parameters and AlphaPose parameters
parser.add_argument("--mode",
                    type=str,
                    default="base",
                    help="Include pose estimation with GT detections (poses-gt), or with object detector (poses-det)")
parser.add_argument("--alpha_cfg",
                    type=str,
                    default="/home/erik/gitprojects/AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml",
                    help="Path to the AlphaPose config file")
parser.add_argument("--alpha_checkpoint",
                    type=str,
                    default="/home/erik/gitprojects/AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth",
                    help="Path to the AlphaPose checkpoint")
parser.add_argument("--gpus",
                    type=int,
                    nargs='+',
                    default=[0],
                    help="GPUs to use for AlphaPose")
parser.add_argument("--detector",
                    type=str,
                    default="yolox-l",
                    help="Detector to use for AlphaPose")
parser.add_argument("--detbatch",
                    type=int,
                    default=1,
                    help="Batch size for AlphaPose detector")
parser.add_argument("--qsize",
                    type=int,
                    default=1,
                    help="Batch size for AlphaPose pose estimator")
parser.add_argument("--posebatch",
                    type=int,
                    default=1,
                    help="Batch size for AlphaPose pose estimator")
parser.add_argument("--sp",
                    type=bool,
                    default=True,
                    help="Whether to use single-process mode for AlphaPose")
parser.add_argument("--tracking",
                    type=bool,
                    default=False,
                    help="Whether to use tracking for AlphaPose")
parser.add_argument("--save_img",
                    type=bool,
                    default=False,
                    help="Whether to save images for AlphaPose")
parser.add_argument("--vis",
                    type=bool,
                    default=False,
                    help="Whether to visualize images for AlphaPose")
parser.add_argument("--pose_flow",
                    type=bool,
                    default=False,
                    help="Whether to use pose flow for AlphaPose")
parser.add_argument("--pose_track",
                    type=bool,
                    default=False,
                    help="Whether to use pose tracking for AlphaPose")
parser.add_argument("--min_box_area",
                    type=float,
                    default=0,
                    help="Minimum box area for AlphaPose")
parser.add_argument("--model_id",
                    type=str,
                    required=True,
                    help="Model ID for the pose estimator")
parser.add_argument("--sample_to",
                    type=int,
                    default=512,
                    help="Number of points per point cloud")
parser.add_argument("--alpha_path",
                    type=str,
                    default="/home/erik/gitprojects/AlphaPose",
                    help="Path to AlphaPose")
parser.add_argument("--pose_runs",
                    type=str,
                    default="./ScePT/poses/runs",
                    help="Path to pose runs")
parser.add_argument("--pose_path",
                    type=str,
                    default="./ScePT/poses/runs",
                    help="Path to pose runs")
args = parser.parse_args()