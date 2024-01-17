import sys
from poses.Pose_Estimation_main.models.supervised.fusion.lidar_2dkeypoint import Lidar2dKeypointFusionmodel
import torch
import logging

# AlphaPose
#sys.path.append('/home/erik/gitproject/AlphaPose')
from alphapose.utils.config import update_config
from alphapose.models import builder
from trackers.tracker_cfg import cfg as tcfg
from alphapose.utils.detector import DetectionLoader
from detector.apis import get_detector
from alphapose.utils.writer import DataWriter
from poses.Pose_Estimation_main.train import normalize_keypoints2D_batch
from poses.Pose_Estimation_main.configs.constants import *


def get_pose_model():
    return Lidar2dKeypointFusionmodel()


class PoseEstimator:
    def __init__(self, model_id, args):
        self.args = args

        sys.path.append(args.alpha_path)
        # Load Fusion Network
        self.model = Lidar2dKeypointFusionmodel()
        logging.info(f"Loading model weights from {model_id}/ckpts/best_model")
        self.model.load_state_dict(torch.load(args.pose_path + "/" + model_id + "/ckpts/best_model"))
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
    


    def map_keypoints(self, keypoints_coco):
        """
        Method to map keypoints from COCO format to Waymo format
        keypoints_coco: [x1, y1, c1, ..., xk, yk, ck]
        keypoints_waymo: array [B, num_joints, 2]
        """
        keypoints_waymo = torch.zeros((keypoints_coco.shape[0], self.model.num_joints, 2), dtype=torch.float32)
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
    

    def estimate_poses(self, img, pc):
        """
        img: List of images (numpy)
        pc: Numpy array of point clouds as [batch_size, 3, point_num]
        returns
        predictions: Tensor of shape [batch_size, num_joints, 3]
        """
        det_loader = DetectionLoader(img,#[numpy_array.numpy() for numpy_array in img],
                                    get_detector(self.args),
                                    self.alpha_cfg,
                                    self.args,
                                    batchSize=self.args.detbatch,
                                    mode='loaded_image',
                                    queueSize=self.args.qsize)
        det_worker = det_loader.start()
        data_len = det_loader.length
        #im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
        batchSize = self.args.posebatch
        writer = DataWriter(self.alpha_cfg, self.args, save_video=False, queueSize=self.args.qsize).start()
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
                    #print('No Human Detected')
                    # Save indices and add groundtruth keypoints later
                    indices_wo_detection.append(i)
                    continue
                # Pose Estimation
                indices_w_detection.append(i)
                inps = inps.to(self.args.device)
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
        # First filter keypoints to the ones you want then normalize using the re-implemented function
        # Output should be a tensor of shape [batch_size, num_joints, 2]
        # Batch the keypoints to tensor
        # TODO: Recheck if batch_size is correct here
        parsed_keypoints = torch.zeros((self.args.detbatch, self.alpha_cfg.DATA_PRESET.NUM_JOINTS, 2), dtype=torch.float32)
        for i, sample in enumerate(results):
            index = indices_w_detection[i]
            if len(sample['result']) == 0:
                indices_wo_detection.append(index)
            else:
                parsed_keypoints[index] = sample['result'][0]['keypoints']
        # Map Keypoints
        keypoints_2D = self.map_keypoints(parsed_keypoints)
        # Normalize Keypoints
        keypoints_2D = normalize_keypoints2D_batch(keypoints_2D).to(self.args.device)
        #print("Keypoints Normalized")
        # TODO: Decide what to do if there are no detections
        # Right now: fill with zeros and use indices_wo_detection
        for i in indices_wo_detection:
            keypoints_2D[i] = torch.zeros((parsed_keypoints.shape[0], self.model.num_joints, 2), dtype=torch.float32)
            return torch.zeros((1, 13, 3), dtype=torch.float32).to(self.args.device)
        keypoints_2D = keypoints_2D.to(self.args.device)

        # Turn numpy point clouds into tensor
        pc = torch.from_numpy(pc).to(self.args.device)
        predictions, trans_features, loss_contributions = self.model(pc, keypoints_2D)

        return predictions