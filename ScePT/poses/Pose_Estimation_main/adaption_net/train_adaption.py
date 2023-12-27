import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.supervised.lifting_networks.model_utils import Encoder2D, Decoder3D
from evaluation.metrics import Metrics
from input_pipeline.waymo_dataloader import WaymoOpenDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils import data as torch_data
from models.supervised.lifting_networks.model_utils import Encoder2D
from waymo_open_dataset.utils import keypoint_draw
from waymo_open_dataset.camera.ops import py_camera_model_ops
import matplotlib.pyplot as plt
from utils.debug_helper import get_wireframe
import torch.nn as nn
import pandas as pd
import pickle
import torch
import torch.optim as optim
import numpy as np


class AdaptionNet(nn.Module):
    def __init__(self, latent_dim, num_joints=13, activation=nn.ReLU):
        super(AdaptionNet, self).__init__()
        self.type = "adaption"

        self.num_joints = num_joints
        self.encoder_2d = Encoder2D(latent_dim, num_joints, activation)
        self.latent_dim = latent_dim
        # NOTE: +7 since we also use (angular) velocity and cam as additional feature
        self.fc = nn.Linear(self.latent_dim + 7, 2*self.num_joints, dtype=torch.float64)

    def forward(self, poses_2d, cam, velocity, angular_velocity):
        enc_2d = self.encoder_2d(poses_2d)

        cam = cam.unsqueeze(1)
        v_x = velocity[:, 0].unsqueeze(1)
        v_y = velocity[:, 1].unsqueeze(1)
        v_z = velocity[:, 2].unsqueeze(1)

        w_x = angular_velocity[:, 0].unsqueeze(1)
        w_y = angular_velocity[:, 1].unsqueeze(1)
        w_z = angular_velocity[:, 2].unsqueeze(1)

        offset = torch.cat((enc_2d, cam, v_x, v_y, v_z, w_x, w_y, w_z), dim=1)
        offset = self.fc(offset)
        # skip connection to only predict offset
        preds = poses_2d + offset.reshape(poses_2d.shape[0], self.num_joints, 2)

        return preds


class Adaption_Dataset(WaymoOpenDataset):
    def __init__(self, csv="/media/petbau/data/waymo/v0.10/3D_2D/image_segment_relations.csv",
                 labels="/media/petbau/data/waymo/v0.10/3D_2D/labels.pkl") -> None:

        self.min_2D_keypoints = 10

        self.csv = pd.read_csv(csv)
        self.labels = labels
        with open(labels, 'rb') as pickle_file:
            self.labels = pickle.load(pickle_file)
        # self.rm_sparse_labels()
        self.rm_no_hips()

    def rm_no_hips(self):
        del_keys = []
        count = 0
        for key in self.labels.keys():
            if not (8 in self.labels[key]['keypoints_3d'] and 16 in self.labels[key]['keypoints_3d']) or not (8 in self.labels[key]['keypoints_2d'] and 16 in self.labels[key]['keypoints_2d']):
                del_keys.append(key)
                count += 1
        for key in del_keys:
            del self.labels[key]
            self.csv.drop(self.csv.index[self.csv['image_id'] == key], inplace=True)
        print(f'Removing {count} samples from a dataset due to missing 3D hips.')

    # def rm_sparse_labels(self):
    #     del_keys = []
    #     count = 0
    #     for key in self.labels.keys():
    #         if len(self.labels[key]['keypoints_2d'].keys()) <= self.min_2D_keypoints:
    #             del_keys.append(key)
    #             count += 1
    #     for key in del_keys:
    #         del self.labels[key]
    #     self.csv = self.csv[~self.csv['image_id'].isin(del_keys)]
    #     print(f'Removing {count} samples from a dataset due to number of labeled 2D keypoints being smaller than {self.min_2D_keypoints}.')

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> dict:

        # get item from file
        key = self.csv.iloc[idx].image_id
        data = self.labels[key]
        # print(key)
        # get distance for normalization
        inv_extrinsics = np.linalg.inv(data['extrinsic'])
        root_vehicle = np.array([data['bb_3d']['center_x'], data['bb_3d']['center_y'], data['bb_3d']['center_z'], 1])
        # transform to sensor coordinate system
        root = inv_extrinsics @ root_vehicle
        # remove homogenous coordinates
        root = root[:-1] / root[-1]
        depth = root[0]

        keypoints_labeled = data['keypoints_2d_arr'][:, :2]

        occlusions_labeled = data['keypoints_2d_arr'][:, -1]

        mask_2D_labeled = np.ones(keypoints_labeled.shape, dtype=np.bool)
        mask_2D_labeled[data['mask_2d']] = False

                # direct_projection[~mask_G[:, :, 0]] = torch.tensor(50000)
                # x_zero = ((direct_projection[:, 4, 0] + direct_projection[:, 10, 0])/2)
                # x_zero = torch.torch.repeat_interleave(x_zero.unsqueeze(1), direct_projection.shape[1], 1)
                # y_zero = (direct_projection[:, 4, 1] + direct_projection[:, 10, 1])/2
                # y_zero = torch.torch.repeat_interleave(y_zero.unsqueeze(1), direct_projection.shape[1], 1)
                # # min_x = torch.min(direct_projection[:, :, 0], dim=1)[0]
                # # min_y = torch.min(direct_projection[:, :, 1], dim=1)[0]
                # direct_projection[:, :, 0] = direct_projection[:, :, 0] - x_zero
                # direct_projection[:, :, 1] = direct_projection[:, :, 1] - y_zero
                # direct_projection = direct_projection.div(origin_G[:, 0].unsqueeze(1).unsqueeze(1))
                # direct_projection = direct_projection * mask_G.to(torch.uint8)
                # errG_direct_reprojection = Metrics.masked_l1(direct_projection, keypoints_2D_G, mask_G)
                # # DEPRECATED -> REMOVE

        x_zero, y_zero = (keypoints_labeled[4] + keypoints_labeled[10]) / 2

        keypoints_labeled[:, 0] = keypoints_labeled[:, 0] - x_zero
        keypoints_labeled[:, 1] = keypoints_labeled[:, 1] - y_zero

        mask_idx_labeled = np.where(mask_2D_labeled[:, 0] == False)
        keypoints_labeled[mask_idx_labeled] = np.array(np.zeros(2))
        keypoints_labeled = keypoints_labeled/depth
        keypoints_3D = data['keypoints_3d_arr'][:, :3]
        occlusions_3D = data['keypoints_3d_arr'][:, -1]
        extrinsic = data['extrinsic']
        intrinsic = np.squeeze(data['intrinsic'])
        metadata = data['metadata']
        camera_image_metadata = data['camera_image_metadata']
        frame_pose_transform = data['frame_pose_transform']
        keypoints_world = np.einsum('ij,nj->ni', frame_pose_transform[:3, :3], keypoints_3D) + frame_pose_transform[:3, 3]
        cp_keypoints = py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata, camera_image_metadata, keypoints_world)

        # get max and min of x and y for normalization
        valid_cp_mask = np.array(cp_keypoints[:, -1], dtype=bool)

        cp_keypoints_masked = cp_keypoints[valid_cp_mask]
        min_x = np.min(cp_keypoints_masked[:, 0])
        max_x = np.max(cp_keypoints_masked[:, 0])
        min_y = np.min(cp_keypoints_masked[:, 1])
        max_y = np.max(cp_keypoints_masked[:, 1])
        # x_zero, y_zero, _ = (cp_keypoints[4] + cp_keypoints[10]) / 2

        # min_x = max(0, int(data['bb_2d']['center_x'] - data['img_2d_width'] / 2))
        # min_y = max(0, int(data['bb_2d']['center_y'] - data['img_2d_height'] / 2))

        keypoints_projected = np.empty(cp_keypoints[:, :2].shape)
        keypoints_projected[:, 0] = cp_keypoints[:, 0] - min_x
        keypoints_projected[:, 1] = cp_keypoints[:, 1] - min_y
        height = max_y - min_y
        width = max_x - min_x
        # keypoints_2D = self.transform_kp_2D(keypoints_2D, height, width)

        mask_2D_projected = np.expand_dims(valid_cp_mask, 1).repeat(2, 1)
        keypoints_projected[data['mask_3d']] = np.array(np.zeros(2))

        keypoints_labeled = keypoints_labeled[:-2]
        occlusions_labeled = occlusions_labeled[:-2]
        mask_2D_labeled = mask_2D_labeled[:-2]

        keypoints_projected = keypoints_projected[:-2]
        occlusions_projected = occlusions_3D[:-2]
        mask_2D_projected = mask_2D_projected[:-2]
        
        # if np.sum(mask_2D_labeled[:, 0]) > 12:
        # TODO: CHECK THAT ONE: 190_5_ojpChbPNtEiiwJ-F2QI4Hg
        #     # TODO: Implemet way to also remove missing 3D joints otherwise they are maped to zero
        #     from utils.debug_helper import plot_keypoints_2D
        #     plot_keypoints_2D(torch.tensor(keypoints_projected), torch.tensor(mask_2D_projected), name='projected')
        #     plot_keypoints_2D(torch.tensor(keypoints_labeled), torch.tensor(mask_2D_labeled), name='labeled')
        # pose_transform = data['camera_image_metadata'][0]
        # Velocity in m/s.
        v_x = data['camera_image_metadata'][16]
        v_y = data['camera_image_metadata'][17]
        v_z = data['camera_image_metadata'][18]
        # Angular velocity in rad/s.
        w_x = data['camera_image_metadata'][19]
        w_y = data['camera_image_metadata'][20]
        w_z = data['camera_image_metadata'][21]
        # pose_timestamp = data['camera_image_metadata'][7]
        # shutter = data['camera_image_metadata'][8]
        # camera_trigger_time = data['camera_image_metadata'][9]
        # camera_readout_done_time = data['camera_image_metadata'][10]
        cam = int(key.split("_")[1])

        return keypoints_projected, occlusions_projected, mask_2D_projected, keypoints_labeled, occlusions_labeled, mask_2D_labeled, np.arrays((v_x, v_y, v_z)), np.array((w_x, w_y, w_z)), cam, key


def step(model, data, device):
    # sent data to cuda
    data = [d.to(device) if torch.is_tensor(d) else d for d in data]
    keypoints_projected, occlusions_projected, mask_2D_projected, keypoints_labeled, occlusions_labeled, mask_2D_labeled, velocity, angular_velocity, cam, key = data
    preds = model(keypoints_projected, cam=cam, velocity=velocity, angular_velocity=angular_velocity)
    total_mask = mask_2D_projected.logical_and(mask_2D_labeled)

    return Metrics.masked_l1(preds, keypoints_labeled, total_mask), preds


if __name__ == "__main__":
    # load data
    dataset = Adaption_Dataset()
    train_factor = 0.85
    test_factor = 0.15
    train_size = round(train_factor * len(dataset))
    test_size = round(test_factor * len(dataset))
    if len(dataset) != (train_size + test_size):
        train_size += len(dataset) - (train_size + test_size)
    train_data, test_data, = torch_data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    print(f' Train dataset size: {len(train_data)}')
    print(f' Test dataset size: {len(test_data)}')

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # setup network
    model = AdaptionNet(latent_dim=128)
    optimiser = optim.Adam(model.parameters(), lr=0.00005)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)

    epochs = 25
    best_test_loss = np.inf
    for epoch in range(epochs):
        # train
        losses = []
        for i, data in enumerate(train_dataloader):
            # clear gradients
            optimiser.zero_grad()
            loss, _ = step(model, data, device)
            losses.append(loss.item())
            # Backpropagation
            loss.backward()
            optimiser.step()
            if i % 40 == 0:
                print(f"[{epoch}/{epochs}][{i}/{len(train_dataloader)}]: {round((sum(losses)/len(losses)),3)}")
                losses = []
        # test
        model.eval()
        test_losses = []
        for i, data in enumerate(test_dataloader):
            # sent data to cuda
            loss, preds = step(model, data, device)
            test_losses.append(loss.item())
        test_loss = sum(test_losses)/len(test_losses)
        print(f"L1 Error on test set: {round((test_loss),3)}")
        if test_loss < best_test_loss:
            # store model
            best_test_loss = test_loss
            print("New best test loss!")
            print("Saving model to: ./best_adaption_model")
            store_model = f"./adaption_net/best_model_{str(epoch).zfill(5)}"
            torch.save(model.state_dict(), f"./adaption_net/best_model_{str(epoch).zfill(5)}")
            store_images = f"./adaption_net/images_{str(epoch).zfill(5)}/"
            if not os.path.exists(store_images):
                os.mkdir(store_images)
            keypoints_projected, occlusions_projected, mask_2D_projected, keypoints_labeled, occlusions_labeled, mask_2D_labeled, _, _, _, _, _, _, _, key = data
            total_mask = mask_2D_projected.logical_and(mask_2D_labeled)
            keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
            for i in range(15):
                # create subplots
                fig, ax = plt.subplots(nrows=1, ncols=3)

                # label subplot
                wireframe, width, height = get_wireframe(keypoints_labeled[i], mask_2D_labeled[i])
                white_image = np.ones((int(height*1.25), int(width*1.25), 3))
                ax[0].imshow(white_image)
                ax[0].axis('off')
                ax[0].set_autoscale_on(False)
                ax[0].set_title('LABEL')
                keypoint_draw.draw_camera_wireframe(ax[0], wireframe)

                # projection subplot 
                wireframe, _, _ = get_wireframe(keypoints_projected[i], mask_2D_projected[i])
                white_image = np.ones((int(height*1.25), int(width*1.25), 3))
                ax[1].imshow(white_image)
                ax[1].axis('off')
                ax[1].set_autoscale_on(False)
                ax[1].set_title('PROJECTION')
                keypoint_draw.draw_camera_wireframe(ax[1], wireframe)

                # prediction subplot
                wireframe, _, _ = get_wireframe(preds[i], total_mask[i])
                white_image = np.ones((int(height*1.25), int(width*1.25), 3))
                ax[2].imshow(white_image)
                ax[2].axis('off')
                ax[2].set_autoscale_on(False)
                ax[2].set_title('PREDICTION')
                keypoint_draw.draw_camera_wireframe(ax[2], wireframe)
                
                # plt.show()
                plt.savefig(store_images + key[i])
                plt.close()

        model.train()
