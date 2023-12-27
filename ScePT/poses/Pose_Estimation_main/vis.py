from xml.dom import NoModificationAllowedErr
import numpy as np
import logging
import torch
import gin
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
import os
import math
import cv2
import PIL
import io
import matplotlib.patches as patches

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_draw, frame_utils
from waymo_open_dataset.utils.keypoint_draw import Wireframe

from configs.constants import JOINT_KEYS, VIS_COMPLETE_IMG_HTML_STATIC, IMAGES_TO_VIS, JOINT_NAMES, JOINt_COLORS_DICT

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300


class Visualizer():
    """
    Visualize the 3D pose estimation of the trained models.
    """

    logging.info('#'*39)
    logging.info('#' * 7 + "   Starting Visualization   " + '#'*7)
    logging.info('#'*39)
    logging.info('')

    REVERSE_JOINT_KEYS = {value: key for key, value in JOINT_KEYS.items()}

    def __init__(self, trainer, dataset):
        """
        Args:
            trainer (train.Trainer): instance of the Trainer class.
            dataset (string): Name of the used dataset.
        """

        self.trainer = trainer
        if dataset == "waymo_3d_2d_projections_supervised":
            self.display_3D_image_projection = True
        else:
            self.display_3D_image_projection = False

    def update_run_paths(self, training):

        self.run_paths = {}
        self.run_paths["path_animations"] = self.trainer.run_paths["path_animations"] + f"/training_vis_{training}/"
        self.run_paths['path_animations_2d_pngs'] = self.run_paths["path_animations"] + "/2d_pngs"
        self.run_paths['path_animations_rotations'] = self.run_paths["path_animations"] + "/rotations"

        for key, value in self.run_paths.items():
            if not os.path.exists(value):
                os.mkdir(value)

    def vis_predictions_from_training(self, keypoints_2D, predictions, epoch, keys_G, mask_3D, pc, intrinsics, origin, num_samples=5):
        c = 0
        keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
        self.update_run_paths(training=f"wo_gt_{str(epoch)}")
        self.trainer.generator.eval()

        with torch.no_grad():
            for kp in keypoints_2D:
                self.trainer.homogeneous_extension = torch.ones(len(predictions), self.trainer.generator.num_joints, 1)
                store_path = self.run_paths['path_animations_rotations'] + "/" + str(c).zfill(5) + "/"
                store_path_pred = store_path + "pred/"

                if not os.path.exists(store_path):
                    os.mkdir(store_path)
                if not os.path.exists(store_path_pred):
                    os.mkdir(store_path_pred)

                # store normal keypoints
                min_x, min_y = torch.min(kp, dim=0).values
                max_x, max_y = torch.max(kp, dim=0).values

                width = int((max_x - min_x).item()*100)
                height = int((max_y - min_y).item()*100)

                kp[:, 0] = kp[:, 0] - min_x
                kp[:, 1] = kp[:, 1] - min_y

                kp_list = []
                for counter, point in enumerate(kp):
                    if mask_3D[c][:, 0][counter]:
                        cam_keypoint = keypoint_pb2.CameraKeypoint()
                        cam_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                        cam_keypoint.keypoint_2d.location_px.x = point[0].cpu()*100 + int(0.15*width)
                        cam_keypoint.keypoint_2d.location_px.y = point[1].cpu()*100 + int(0.15*height)
                        cam_keypoint.keypoint_2d.visibility.is_occluded = 0
                        kp_list.append(cam_keypoint)
                camera_wireframe = keypoint_draw.build_camera_wireframe(kp_list)

                _, ax = plt.subplots(frameon=False, figsize=(10, 10))
                white_image = np.ones((int(height*1.25), int(width*1.25), 3))
                self._imshow(ax, white_image)
                keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)
                plt.savefig(store_path + "/" + "input" + ".png")
                plt.close()
                with open(f"{store_path}id.txt", "w") as text_file:
                    text_file.write(f"{keys_G[c]}")

                np.savetxt(f"{store_path}preds.txt", predictions[c].cpu().detach().numpy())
                np.savetxt(f"{store_path}keypoints_2D.txt", keypoints_2D[c].cpu().detach().numpy())

                for rot_angle in np.arange(10, 370, 10):
                    # predictions
                    proj_rot_preds, predictions_rot, rot_pc = self.trainer.get_proj_rot_preds(predictions, mask_3D, pc, intrinsics, origin=origin,  deterministic_angle=rot_angle)
                    cropped_camera_keypoints = []
                    # y_shift = abs(proj_rot_preds[sample, :, 1][mask_3D[sample][:, 0]].min())
                    kp_2D = proj_rot_preds[c]
                    min_x, min_y = torch.min(kp_2D, dim=0).values
                    max_x, max_y = torch.max(kp_2D, dim=0).values

                    width = int((max_x - min_x).item())
                    height = int((max_y - min_y).item())
                    if width >= 1000:
                        sf = 0.5
                    elif width >= 100:
                        sf = 1
                    elif width >= 10:
                        sf = 10
                    else:
                        sf = 100
                        width += 1
                        height += 1
                    width = sf * width
                    height = sf * height
                    kp_2D[:, 0] = kp_2D[:, 0] - min_x
                    kp_2D[:, 1] = kp_2D[:, 1] - min_y
                    for counter, keypoint in enumerate(kp_2D):
                        if mask_3D[c][:, 0][counter]:
                            cam_keypoint = keypoint_pb2.CameraKeypoint()
                            cam_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                            cam_keypoint.keypoint_2d.location_px.x = keypoint[0].cpu()*sf + int(0.15*width)
                            cam_keypoint.keypoint_2d.location_px.y = keypoint[1].cpu()*sf + int(0.15*height)
                            cam_keypoint.keypoint_2d.visibility.is_occluded = 0
                            cropped_camera_keypoints.append(cam_keypoint)

                    camera_wireframe = keypoint_draw.build_camera_wireframe(
                        cropped_camera_keypoints)

                    _, ax = plt.subplots(frameon=False, figsize=(10, 10))
                    white_image = np.ones((int(height*1.25), int(width*1.25), 3))
                    self._imshow(ax, white_image)
                    keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)
                    plt.savefig(store_path_pred + "/" + str(rot_angle) + ".png")
                    plt.close()
                c += 1
                if c >= num_samples:
                    return

    def vis_samples(self, num_samples=12, training=False, show_vis=False):
        """Visualize samples from test set

        Args:
            num_samples (int, optional): Number of samples to visualize. Defaults to 12.
            training (bool/string, optional): If visualization is during training, use different store path. Defaults to False.
            show_vis (bool, optional): Only needed for 3D->2D supervised projections. Defaults to False.
        """

        self.trainer.generator.eval()
        with torch.no_grad():
            # get some predictions from the test set
            if training:
                self.update_run_paths(training)
            else:
                self.run_paths = self.trainer.run_paths
                logging.info(f'Saving visualizations to: {self.run_paths["path_animations"]}')

            data = next(iter(self.trainer.test_set))

            for sample in range(0, num_samples):

                complete_sample, image = self.trainer.test_set.dataset.dataset.get_complete_sample(int(data['idx'][sample]))

                # plot 2d keypoints
                keypoints_2D_proto = []
                for keypoint in complete_sample['keypoints_2d']:
                    cam_keypoint = keypoint_pb2.CameraKeypoint()
                    cam_keypoint.type = keypoint
                    cam_keypoint.keypoint_2d.location_px.x = float(complete_sample['keypoints_2d'][keypoint]['x'])
                    cam_keypoint.keypoint_2d.location_px.y = float(complete_sample['keypoints_2d'][keypoint]['y'])
                    cam_keypoint.keypoint_2d.visibility.is_occluded = int(complete_sample['keypoints_2d'][keypoint]['occluded'])
                    keypoints_2D_proto.append(cam_keypoint)

                # remove cam from cp_points and add distance for rgb visualization
                cp_points = complete_sample['cp_points'][:, 1:]
                dist = np.linalg.norm(complete_sample['lidar'], axis=-1, keepdims=True)

                camera_wireframe = keypoint_draw.build_camera_wireframe(keypoints_2D_proto)
                keypoint_draw.OCCLUDED_BORDER_WIDTH = 3

                # set distance values for points that have been selected as 'close to 2D keypoint to 0 -> other color in vis
                if self.trainer.type == 'weakly_supervised':
                    indices_shape = data['closest_cp_idx'][sample].shape
                    indices_3D = data['closest_cp_idx'][sample].type(torch.int32)
                    indices_img = data['closest_cp_idx_before_resampling'][sample]
                    dist[indices_img] = 0
                    cp_points = np.concatenate([cp_points, dist], axis=1)
                    self.plot_points_on_image(cp_points, image, self.rgba)
                    self.color_close_projections_on_image(cp_points, indices_img, cp_points)
                    keypoint_draw.draw_camera_wireframe(plt.gca(), camera_wireframe)
                else:
                    indices_img = []
                    indices_3D = []
                    indices_shape = None
                    fig0, ax = plt.subplots(frameon=False, figsize=(10, 10))
                    self._imshow(ax, image)
                    keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)

                img_store_path = self.run_paths['path_animations_2d_pngs'] + "/" + str(sample).zfill(5) + ".png"
                plt.axis('off')
                plt.savefig(img_store_path)
                plt.grid(b=None)
                plt.close()

                keypoints_2D = data['keypoints_2D']
                occlusions_2D = data['occlusions_2D']
                # mask_2D = data['mask_2D']

                keypoints_3D = data['keypoints_3D']
                occlusions_3D = data['occlusions_3D']
                mask_3D = data['mask_3D']
                origin = data['root']

                pc = data['pc'].transpose(2, 1)
                if self.trainer.type == "weakly_supervised":
                    intrinsics = data['intrinsics']

                # plot 3D camera projection
                if self.display_3D_image_projection:
                    keypoints_3D_projections = []
                    counter = 0
                    mask_3D_projection = data['mask_3D'][sample][:, 0]
                    min_vertical = torch.min(keypoints_2D[sample][:, -1])
                    max_vertical = torch.max(keypoints_2D[sample][:, -1])

                    for keypoint, occlusion in zip(keypoints_2D[sample], occlusions_2D[sample]):
                        if mask_3D_projection[counter]:
                            cam_keypoint = keypoint_pb2.CameraKeypoint()
                            cam_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                            cam_keypoint.keypoint_2d.location_px.x = (keypoint[0] + 1) * 250 + 500  # rescale to fit better to an image [-1,1] [0,200]
                            cam_keypoint.keypoint_2d.location_px.y = (keypoint[1] + abs(min_vertical)) * 250 + 500
                            cam_keypoint.keypoint_2d.visibility.is_occluded = False if int(occlusion) == 1 else True
                            keypoints_3D_projections.append(cam_keypoint)
                        counter += 1

                    camera_wireframe_3d_projections = keypoint_draw.build_camera_wireframe(keypoints_3D_projections*100)
                    fig01, ax_01 = plt.subplots(frameon=False, figsize=(10, 10))
                    white_image = np.ones((1000, 1000, 3))
                    self._imshow(ax_01, white_image)
                    keypoint_draw.draw_camera_wireframe(ax_01, camera_wireframe_3d_projections)
                    img_store_path_3d_projection = self.run_paths['path_animations_3d_projections_pngs'] + "/" + str(sample).zfill(5) + ".png"
                    plt.savefig(img_store_path_3d_projection)
                    plt.close()

                figures = []

                # plot 3d keypoints
                keypoints_3D_gt = []
                counter = 0
                for keypoint in keypoints_3D[sample]:
                    if mask_3D[sample][counter][0]:
                        laser_keypoint = keypoint_pb2.LaserKeypoint()
                        laser_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                        laser_keypoint.keypoint_3d.location_m.x = keypoint[0]  # complete_sample['keypoints_3d'][keypoint]['x'] - complete_sample['bb_3d']['center_x']
                        laser_keypoint.keypoint_3d.location_m.y = keypoint[1]  # complete_sample['keypoints_3d'][keypoint]['y'] - complete_sample['bb_3d']['center_y']
                        laser_keypoint.keypoint_3d.location_m.z = keypoint[2]  # complete_sample['keypoints_3d'][keypoint]['z'] - complete_sample['bb_3d']['center_z']
                        laser_keypoint.keypoint_3d.visibility.is_occluded = False if int(occlusions_3D[sample][counter]) == 1 else True
                        keypoints_3D_gt.append(laser_keypoint)
                    counter += 1
                laser_wireframe_gt = keypoint_draw.build_laser_wireframe(keypoints_3D_gt)

                # fig1 = self._create_plotly_figure(title="3D GT")
                # self.draw_laser_wireframe(fig1, laser_wireframe_gt)
                # figures.append(fig1)

                fig1 = self._create_plotly_figure(title="3D GT with LIDAR")
                self.draw_laser_wireframe(fig1, laser_wireframe_gt)
                self._draw_laser_points(fig1, pc[sample].transpose(0, 1), laser_wireframe_gt.dots, indices_3D=indices_3D,
                                        indices_shape=indices_shape)
                figures.append(fig1)

                fig1_2 = self._create_plotly_figure(title="3D GT")
                self.draw_laser_wireframe(fig1_2, laser_wireframe_gt)
                figures.append(fig1_2)

                # create store path for thesis
                store_path_thesis_pics = self.run_paths['path_animations'] + "/thesis_pics/"
                if not os.path.exists(store_path_thesis_pics):
                    os.mkdir(store_path_thesis_pics)
      
                origin_unit_vect = data['root'][sample] / torch.norm(data['root'][sample])
                camera_vect = - origin_unit_vect * 3


                predictions = self.predictions(pc, keypoints_2D, gt=(keypoints_3D, data['mask_3D']))

                # plot predictions
                # mask = mask_3D[sample][:, 0]  # torch.any(keypoints_3D[sample], -1)
                laser_wireframe_pred = self.build_laser_wireframe(predictions[sample], occlusions_3D[sample])

                fig2 = self._create_plotly_figure(title="Predictions with LIDAR")
                self.draw_laser_wireframe(fig2, laser_wireframe_pred)
                self._draw_laser_points(fig2, pc[sample].transpose(0, 1))
                figures.append(fig2)

                fig2_2 = self._create_plotly_figure(title="Predictions")
                self.draw_laser_wireframe(fig2_2, laser_wireframe_pred)
                figures.append(fig2_2)
                

                fig3 = self._create_plotly_figure(title="Predictions & GT (grey)")
                self.draw_laser_wireframe(fig3, laser_wireframe_pred)
                self.draw_laser_wireframe(fig3, laser_wireframe_gt, grey=True)
                figures.append(fig3)
                
                

                filename = self.run_paths['path_animations'] + "/" + str(sample).zfill(5) + ".html"
                if self.display_3D_image_projection:
                    self.figures_to_html(figures, filename=filename,
                                         img_store_path=img_store_path,
                                         idx=data['idx'][sample],
                                         img_store_path_3d_projection=img_store_path_3d_projection)
                else:
                    self.figures_to_html(figures, filename=filename,
                                         img_store_path=img_store_path,
                                         idx=data['idx'][sample])
                if self.trainer.type == "weakly_supervised":
                    self.vis_rotations(predictions, keypoints_3D, mask_3D, pc, intrinsics, origin, sample)
                    
                    
                # create pics for thesis
                eye = (camera_vect[0].item(), camera_vect[1].item(), camera_vect[2].item())
                self.save_thesis_pic(fig1_2, eye, "_gt_normal", sample, store_path_thesis_pics)
                eye = (0., -3, 0)
                self.save_thesis_pic(fig1_2, eye, "_gt_y", sample, store_path_thesis_pics)
                eye = (-3., 0, 0)
                self.save_thesis_pic(fig1_2, eye, "_gt_x", sample, store_path_thesis_pics)
                fig1_2.layout.title = "3D GT"
                
                eye = (camera_vect[0].item(), camera_vect[1].item(), camera_vect[2].item())
                self.save_thesis_pic(fig3, eye, "_merged_normal", sample, store_path_thesis_pics)
                eye = (0., -3, 0)
                self.save_thesis_pic(fig3, eye, "_merged_y", sample, store_path_thesis_pics)
                eye = (-3., 0, 0)
                self.save_thesis_pic(fig3, eye, "_merged_x", sample, store_path_thesis_pics)
                fig1_2.layout.title = "Predictions & GT (grey)"
                
                eye = (camera_vect[0].item(), camera_vect[1].item(), camera_vect[2].item())
                self.save_thesis_pic(fig2_2, eye, "_pred_normal", sample, store_path_thesis_pics)
                eye = (0., -3, 0)
                self.save_thesis_pic(fig2_2, eye, "_pred_y", sample, store_path_thesis_pics)
                eye = (-3., 0, 0)
                self.save_thesis_pic(fig2_2, eye, "_pred_x", sample, store_path_thesis_pics)
                fig1_2.layout.title = "Predictions"
                    
                

    def figures_to_html(self, figs, filename, img_store_path, idx, img_store_path_3d_projection=None):
        with open(filename, 'w') as dashboard:
            dashboard.write("<html><head></head><body>" + "\n")
            dashboard.write('<center>')
            dashboard.write(f'<b>{self.trainer.test_set.dataset.dataset.get_id(idx)}</b>')
            if self.display_3D_image_projection:
                dashboard.write('<p>&nbsp;</p>')
                dashboard.write('<p>3D projection to camera:</p>')
                dashboard.write('<img src=' + f'".{img_store_path_3d_projection.split("animations")[-1]}"' + ', alt="3D keypoint projection to the image plane"')
            dashboard.write('<p>&nbsp;</p>')
            dashboard.write('<p>2D Joint Labels</p>')
            dashboard.write('<img src=' + f'".{img_store_path.split("animations")[-1]}"' + ', alt="2D keypoints on image"')

            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write('</center>')
            dashboard.write("</body></html>" + "\n")

    def build_laser_wireframe(self, predictions, occlusions_3D):

        keypoints_3D_pred = []
        counter = 0
        for keypoint, occlusion in zip(predictions, occlusions_3D):
            laser_keypoint = keypoint_pb2.LaserKeypoint()
            laser_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
            laser_keypoint.keypoint_3d.location_m.x = keypoint[0]
            laser_keypoint.keypoint_3d.location_m.y = keypoint[1]
            laser_keypoint.keypoint_3d.location_m.z = keypoint[2]
            laser_keypoint.keypoint_3d.visibility.is_occluded = False if int(occlusion) == 1 else True
            keypoints_3D_pred.append(laser_keypoint)
            counter += 1

        return keypoint_draw.build_laser_wireframe(keypoints_3D_pred)

    def predictions(self, pc, keypoints_2D, gt):
        # weakly_supervised
        if self.trainer.type == "weakly_supervised":
            predictions, _ = self.trainer.generator(keypoints_2D, pc)
        # supervised
        else:
            if self.trainer.generator.type == "point_cloud":
                predictions, _ = self.trainer.generator(pc)
            elif self.trainer.generator.type == "keypoints":
                predictions = self.trainer.generator(keypoints_2D)
            elif self.trainer.generator.type == "fusion":
                predictions, _, _ = self.trainer.generator(pc, keypoints_2D, gt=gt)
            else:
                logging.error('Model input not defined properly.')
                sys.exit(1)
        return predictions

    def vis_rotations(self, predictions, keypoints_3D, mask_3D, pc, intrinsics, origin, sample):
        """
        Show projections of rotated keypoints and store them (.png) into an output directory
        """
        # quick fix to change the trainer.homogeneous_extension size from
        # specified batch size to the now used one...
        self.trainer.homogeneous_extension = torch.ones(len(predictions), self.trainer.generator.num_joints, 1)
        store_path = self.run_paths['path_animations_rotations'] + "/" + str(sample).zfill(5) + "/"
        store_path_pred = store_path + "pred/"
        store_path_gt = store_path + "gt/"
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        if not os.path.exists(store_path_gt):
            os.mkdir(store_path_gt)
        if not os.path.exists(store_path_pred):
            os.mkdir(store_path_pred)

        for rot_angle in np.arange(10, 370, 10):
            # predictions
            proj_rot_preds, predictions_rot, rot_pc = self.trainer.get_proj_rot_preds(predictions, mask_3D, pc, intrinsics, origin=origin,  deterministic_angle=rot_angle)
            cropped_camera_keypoints = []
            # y_shift = abs(proj_rot_preds[sample, :, 1][mask_3D[sample][:, 0]].min())
            kp_2D = proj_rot_preds[sample]
            min_x, min_y = torch.min(kp_2D, dim=0).values
            max_x, max_y = torch.max(kp_2D, dim=0).values
            width = int((max_x - min_x).item()*100)
            height = int((max_y - min_y).item()*100)
            kp_2D[:, 0] = kp_2D[:, 0] - min_x
            kp_2D[:, 1] = kp_2D[:, 1] - min_y
            for counter, keypoint in enumerate(kp_2D):
                if mask_3D[sample][:, 0][counter]:
                    cam_keypoint = keypoint_pb2.CameraKeypoint()
                    cam_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                    cam_keypoint.keypoint_2d.location_px.x = keypoint[0].cpu()*100 + int(0.15*width)
                    cam_keypoint.keypoint_2d.location_px.y = keypoint[1].cpu()*100 + int(0.15*height)
                    cam_keypoint.keypoint_2d.visibility.is_occluded = 0
                    cropped_camera_keypoints.append(cam_keypoint)

            camera_wireframe = keypoint_draw.build_camera_wireframe(
                cropped_camera_keypoints)

            keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
            _, ax = plt.subplots(frameon=False, figsize=(10, 10))
            white_image = np.ones((int(height*1.25), int(width*1.25), 3))
            self._imshow(ax, white_image)
            keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)
            plt.savefig(store_path_pred + "/" + str(rot_angle) + ".png")
            plt.close()

            # ground truth
            proj_rot_keypoints_3D, predictions_rot, rot_pc = self.trainer.get_proj_rot_preds(keypoints_3D, mask_3D, pc, intrinsics, origin=origin, deterministic_angle=rot_angle)
            cropped_camera_keypoints = []
            # y_shift = abs(proj_rot_preds[sample, :, 1][mask_3D[sample][:, 0]].min())
            for counter, keypoint in enumerate(proj_rot_keypoints_3D[sample]):
                if mask_3D[sample][:, 0][counter]:
                    cam_keypoint = keypoint_pb2.CameraKeypoint()
                    cam_keypoint.type = self.REVERSE_JOINT_KEYS[counter]
                    cam_keypoint.keypoint_2d.location_px.x = keypoint[0].cpu()*100 + 500
                    cam_keypoint.keypoint_2d.location_px.y = keypoint[1].cpu()*100 + 500
                    cam_keypoint.keypoint_2d.visibility.is_occluded = 0
                    cropped_camera_keypoints.append(cam_keypoint)

            camera_wireframe = keypoint_draw.build_camera_wireframe(
                cropped_camera_keypoints)

            keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
            _, ax = plt.subplots(frameon=False, figsize=(5, 7))
            white_image = np.ones((1000, 1000, 3))
            self._imshow(ax, white_image)
            keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)
            plt.savefig(store_path_gt + "/" + str(rot_angle) + ".png")
            plt.close()

    def vis_complete_images(self):
        """"Vis several walkers in one  (Images are defined in IMAGES_TO_VIS)"""

        self.trainer.generator.eval()
        for (frame_number, cam, tfr_path) in IMAGES_TO_VIS:

            dataloader = self.trainer.test_set.dataset.dataset

            # get obj ids that are labeled in that frame
            df = dataloader.csv
            df = df.loc[(df['segment'] == tfr_path) & (df['frame'] == frame_number) & (df['cam'] == cam)]
            ids = list(df['image_id'])
            tfr_ids = list(df['id'])

            # make predictions with trained model
            figures = []
            for idx, id in enumerate(ids):
                raw_data = dataloader.labels[id]
                keypoints_2D, occlusions_2D, mask_2D, keypoints_3D, occlusions_3D, mask_3D, root = self.add_batch_dim_and_tensor(dataloader.process_kp_data(raw_data, id))
                pc = np.expand_dims(dataloader.get_sampled_pc(raw_data, root.numpy()).astype('float32'), axis=0)
                pc = torch.tensor(pc).transpose(2, 1)

                predictions = self.predictions(pc, keypoints_2D, gt=(keypoints_3D, mask_3D))
                predictions = torch.squeeze(predictions)
                occlusions_3D = torch.squeeze(occlusions_3D)
                laser_wireframe_pred = self.build_laser_wireframe(predictions, occlusions_3D)

                fig = self._create_plotly_figure(title=id, width=500, height=500)
                self.draw_laser_wireframe(fig, laser_wireframe_pred)
                # self._draw_laser_points(, pc.transpose(0, 1))
                figures.append(fig)

            # read the tfr file to fet the complete image
            dataset = tf.data.TFRecordDataset(tfr_path, compression_type='')
            frame_counter = 0
            for data in dataset:
                if frame_counter == int(frame_number):
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    break
                frame_counter += 1

            tfr_labels = keypoint_data.group_object_labels(frame)
            camera_image_by_name = {i.name: i.image for i in frame.images}
            image_np = self._imdecode(camera_image_by_name[cam])

            img_path = self.run_paths['path_animations'] + "/" + "complete_img" + str(frame_number) + "_" + str(cam) + "_" + str(tfr_path.split("/")[-1]) + ".jpg"
            # draw bb for ids
            keypoint_draw.OCCLUDED_BORDER_WIDTH = 7
            _, ax = plt.subplots(frameon=False, figsize=(25, 25))
            self._imshow(ax, image_np)

            # draw kepoints and bb for 3D annotated pedestrains
            for tfr_id in tfr_ids:
                kp = tfr_labels[tfr_id].camera[cam].keypoints.keypoint
                camera_wireframe = keypoint_draw.build_camera_wireframe(kp)
                keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)

                tfr_ids_color_dict = {}
                colors = ['red', 'blue', 'green']
                for tfr_id in tfr_ids:
                    tfr_ids_color_dict[tfr_id] = colors[0]
                    colors.pop(0)

                # draw bb
                for camera_labels in frame.camera_labels:
                    if camera_labels.name != cam:
                        continue
                    for label in camera_labels.labels:
                        # Draw the object bounding box.
                        if label.association.laser_object_id in tfr_ids:
                            ax.add_patch(patches.Rectangle(
                                xy=(label.box.center_x - 0.5 * label.box.length,
                                    label.box.center_y - 0.5 * label.box.width),
                                width=label.box.length,
                                height=label.box.width,
                                linewidth=2.5,
                                edgecolor=tfr_ids_color_dict[label.association.laser_object_id],
                                facecolor='none'))

            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)

            (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose)
            points_all = np.concatenate(points, axis=0)
            fig_pc = self._create_plotly_figure(title="Point_Cloud", width=2500)
            self._draw_laser_points(fig_pc, points_all)

            # create html file
            file_path = self.run_paths['path_animations'] + "/" + str(frame_number) + "_" + str(cam) + "_" + str(tfr_path.split("/")[-1]) + ".html"
            # TODO: => Also store real pdf here

            with open(file_path, 'w') as dashboard:
                dashboard.write(VIS_COMPLETE_IMG_HTML_STATIC)

                plot1 = figures[0].to_html().split('<body>')[1].split('</bo1dy>')[0]
                dashboard.write(f'<div class=box{tfr_ids_color_dict[tfr_ids[0]]}><li>{plot1}</li></div>')
                plot2 = figures[1].to_html().split('<body>')[1].split('</bo1dy>')[0]
                dashboard.write(f'<div class=box{tfr_ids_color_dict[tfr_ids[1]]}><li>{plot2}</li></div>')
                plot3 = figures[2].to_html().split('<body>')[1].split('</bo1dy>')[0]
                dashboard.write(f'<div class=box{tfr_ids_color_dict[tfr_ids[2]]}><li>{plot3}</li></div>')
                dashboard.write('</center>')
                dashboard.write('</ul>')
                dashboard.write('</div>')
                dashboard.write('<div class="col-7">')
                dashboard.write('<center>')
                dashboard.write('<h1>Camera with 2D Keypoints</h1>')
                dashboard.write(f'<img src="{img_path}" alt="Pedestrains with BB and annotated 2D keypoints"></center>')
                dashboard.write("</center>")
                dashboard.write('</div>')
                dashboard.write('</div>')
                dashboard.write('</div>')
                plot_pc = fig_pc.to_html().split('<body>')[1].split('</bo1dy>')[0]
                dashboard.write(plot_pc)
                dashboard.write('</body>')
                dashboard.write('</html>')

    @staticmethod
    def save_thesis_pic(fig, eye, type_str, sample, store_path_thesis_pics):
        fig.layout.scene.xaxis.showaxeslabels = False
        fig.layout.scene.xaxis.showbackground = False
        fig.layout.scene.xaxis.showticklabels = False
        fig.layout.scene.yaxis.showaxeslabels = False
        fig.layout.scene.yaxis.showbackground = False
        fig.layout.scene.yaxis.showticklabels = False
        fig.layout.scene.zaxis.showaxeslabels = False
        fig.layout.scene.zaxis.showbackground = False
        fig.layout.scene.zaxis.showticklabels = False
        fig.layout.title = None

        camera = dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        )
        fig.update_layout(scene_camera=camera, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title=''))

        fig.write_image(store_path_thesis_pics + str(sample).zfill(5) + type_str + ".pdf")

    @ staticmethod
    def add_batch_dim_and_tensor(arrays):
        """Add batch dimension to tensors"""
        l = []
        for arr in arrays:
            l.append(torch.tensor(np.expand_dims(arr, axis=0)))
        return tuple(l)

    @ staticmethod
    def _imshow(ax: plt.Axes, image_np: np.ndarray):
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_autoscale_on(False)

    @ staticmethod
    def _draw_laser_points(fig: go.Figure,
                           points: np.ndarray,
                           laser_wireframe_dots=None,
                           color: str = 'gray',
                           indices_3D=[],
                           indices_shape=None,
                           size: int = 2):
        """Visualizes laser points on a plotly figure."""
        fig.add_trace(
            go.Scatter3d(
                mode='markers',
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                marker=dict(color=color, size=size)))
        if len(indices_3D):
            indices_3D = indices_3D.reshape(indices_shape)
            for idx, points_idxs in enumerate(indices_3D):
                joint_name = JOINT_NAMES[idx]
                # only draw if 2d keypoint is labeled
                if points_idxs[0] != -1:
                    for dot in laser_wireframe_dots:
                        if dot.name == joint_name:
                            points_tmp = points[points_idxs.numpy()]
                            fig.add_trace(
                                go.Scatter3d(
                                    mode='markers',
                                    x=points_tmp[:, 0],
                                    y=points_tmp[:, 1],
                                    z=points_tmp[:, 2],
                                    marker=dict(color=dot.color, size=size)))

    @staticmethod
    def _create_plotly_figure(title="", width=500, height=750) -> go.Figure:
        """Creates a plotly figure for 3D visualization."""
        fig = go.Figure()
        axis_settings = dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showbackground=True,
            showaxeslabels=True,
            showticklabels=True)
        fig.update_layout(
            title_text=f"{title}",
            width=width,
            height=height,
            showlegend=False,
            scene=dict(
                aspectmode='data',  # force xyz has same scale,
                xaxis=axis_settings,
                yaxis=axis_settings,
                zaxis=axis_settings,
            ),
        )
        return fig

    @staticmethod
    def _imdecode(buf: bytes) -> np.ndarray:
        with io.BytesIO(buf) as fd:
            pil = PIL.Image.open(fd)
            return np.array(pil)

    @staticmethod
    def rgba(r):
        """Generates a color based on range.

        Args:
            r: the range value of a given point.
        Returns:
            The color for a given range
        """
        c = plt.get_cmap('jet')((r % 20.0) / 20.0)
        c = list(c)
        c[-1] = 0.5  # alpha
        return c

    @staticmethod
    def plot_image(camera_image):
        """Plot a camera image."""
        plt.figure(figsize=(7, 10))
        plt.imshow(camera_image)
        plt.grid("off")

    @staticmethod
    def plot_points_on_image(projected_points, image, rgba_func,
                             point_size=20.0):
        Visualizer.plot_image(image)
        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append('gray')

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

    @staticmethod
    def color_close_projections_on_image(cp_points, indices, D):
        # TODO: ADJUST SIZE OF DRAWN POINTS
        counter = 0
        for row in indices:
            # negative indices mean point is not labeled...
            if not np.array_equal(row.numpy(), np.array([-1, -1, -1, -1])):
                for idx in row:
                    point = cp_points[idx]
                    color = JOINt_COLORS_DICT[JOINT_NAMES[counter]]
                    plt.scatter(point[0], point[1], c=color)
            counter += 1

    @staticmethod
    def draw_laser_wireframe(fig: go.Figure, wireframe: Wireframe, grey=False) -> None:
        OCCLUDED_BORDER_WIDTH = 3
        """Draws a laser wireframe onto the plotly Figure."""
        for line in wireframe.lines:
            points = np.stack([line.start, line.end], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    mode='lines',
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    line=dict(color='#666b6b' if grey else line.color, width=line.width)))
        dot_coords = np.stack([d.location for d in wireframe.dots], axis=0)
        fig.add_trace(
            go.Scatter3d(
                text=[d.name for d in wireframe.dots],
                mode='markers',
                x=dot_coords[:, 0],
                y=dot_coords[:, 1],
                z=dot_coords[:, 2],
                marker=dict(
                    color=['#666b6b' if grey else d.color for d in wireframe.dots],
                    size=[d.size*3 if d.name=='NOSE' else d.size for d in wireframe.dots],
                    line=dict(
                        width=OCCLUDED_BORDER_WIDTH,
                        color=[d.actual_border_color for d in wireframe.dots]))))
