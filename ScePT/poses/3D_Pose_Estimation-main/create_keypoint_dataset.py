""" This file aims to extract the data from the waymo TFR-files and store it in a PyTorch friendly manner"""

import os
import tensorflow as tf
import glob
import io
import PIL.Image
import cv2
import sys
import numpy as np
import pickle
import csv
import json
import math
import copy


from tqdm import tqdm
from typing import Collection
from typing import Tuple

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.utils import keypoint_draw
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

# Imports for new dataset structure (Waymo V2)
import dask.dataframe as dd
from waymo_open_dataset import v2

from configs.constants import JOINT_KEYS

tf.compat.v1.enable_eager_execution()


tfr_path = sys.argv[1]
base_path = sys.argv[2]


# def crop_camera_keypoints(
#     image: np.ndarray,
#     keypoints: Collection[keypoint_pb2.CameraKeypoint],
#     box: label_pb2.Label.Box,
#     margin: float = 0,
# ):
#     """Crops camera image to the specified bounding box and shifts keypoints.

#     Args:
#       image: input image to crop, an array with shape [height, width, 3].
#       keypoints: a collection of camera keypoints.
#       box: a 2D bounding box to extract from the input image.
#       margin: a ratio of the extra margin to add to the image relative to the
#         input image size.

#     Returns:
#       tuple (new image, shifted keypoints).
#     """
#     # Adapted from:
#     # https://github.com/waymo-research/waymo-open-dataset/blob/520c34a2090460d7498078fa9033e011e28e86b0/waymo_open_dataset/utils/frame_utils.py
#     new_camera_keypoints = []
#     crop_width = (1 + margin) * box.length
#     crop_height = (1 + margin) * box.width
#     min_x = max(0, int(box.center_x - crop_width / 2))
#     min_y = max(0, int(box.center_y - crop_height / 2))
#     for old_kp in keypoints:
#         new_kp = keypoint_pb2.CameraKeypoint()
#         new_kp.CopyFrom(old_kp)
#         new_p = new_kp.keypoint_2d.location_px
#         new_p.x -= min_x
#         new_p.y -= min_y
#         new_camera_keypoints.append(new_kp)
#     max_x = min(image.shape[1] - 1, int(box.center_x + crop_width / 2))
#     max_y = min(image.shape[0] - 1, int(box.center_y + crop_height / 2))
#     new_image = image[min_y:max_y, min_x:max_x, :]
#     return new_image, new_camera_keypoints

# Rewritten function to crop img and camera keypoints to new dataset structure
def crop_camera_keypoints(
    image: np.ndarray,
    keypoints: v2.CameraHumanKeypointsComponent,
    box: v2._column_types.BoxAxisAligned2d,
    margin: float = 0
) -> Tuple[np.ndarray, v2.CameraHumanKeypointsComponent]:
    """Crops camera image to the specified bounding box and shifts keypoints.

    Args:
        image: input image to crop, an array with shape [height, width, 3].
        keypoints: a collection of camera keypoints.
        box: a 2D bounding box to extract from the input image.
        margin: a ratio of the extra margin to add to the image relative to the
        input image size.

    Returns:
        a tuple (new image, shifted keypoints).
    """
    crop_width = (1 + margin) * box.size.x
    crop_height = (1 + margin) * box.size.y
    min_x = max(0, int(box.center.x - crop_width / 2))
    min_y = max(0, int(box.center.y - crop_height / 2))
    new_kps_dict = keypoints.to_flatten_dict()
    new_x = []
    new_y = []
    for old_x, old_y in zip(keypoints.camera_keypoints.keypoint_2d.location_px.x, keypoints.camera_keypoints.keypoint_2d.location_px.y):
        new_x.append(old_x - min_x)
        new_y.append(old_y - min_y)
    new_x = np.array(new_x, dtype=np.float64)
    new_y = np.array(new_y, dtype=np.float64)
    #test = new_kps_dict['[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.x']
    new_kps_dict['[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.x'] = new_x
    new_kps_dict['[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.y'] = new_y
    new_kps = v2.CameraHumanKeypointsComponent.from_dict(new_kps_dict)
    max_x = min(image.shape[1] - 1, int(box.center.x + crop_width / 2))
    max_y = min(image.shape[0] - 1, int(box.center.y + crop_height / 2))
    new_image = image[min_y:max_y, min_x:max_x, :]
    return new_image, new_kps


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
    # Adapted from:
    # https://github.com/waymo-research/waymo-open-dataset/blob/520c34a2090460d7498078fa9033e011e28e86b0/waymo_open_dataset/utils/frame_utils.py
    """Convert range images to point cloud.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points_and_ri = []
    cp_points = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))

        masked_range_image_tensor = tf.gather_nd(range_image_tensor,
                                                 tf.compat.v1.where(range_image_mask))

        points_and_masked_image = tf.concat([points_tensor, tf.cast(masked_range_image_tensor, dtype=tf.float32)], axis=-1)

        points_and_ri.append(points_and_masked_image.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points_and_ri, cp_points


def _imdecode(buf: bytes) -> np.ndarray:
    """Function from waymo open dataset examples """
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_keypoints.ipynb
    with io.BytesIO(buf) as fd:
        pil = PIL.Image.open(fd)
        return np.array(pil)


def load_point_cloud(frame):
    """
    Get point cloud from frame

    Args:
        frame (open dataset frame): Frame of the tfrecord file.

    Returns:
       box points (Nx3 dimensional array).
    """
    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    # points_and_range_image, cp_points = convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)

    return points_all, cp_points_all


def convert_range_image_to_point_cloud_v2(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][ri_index]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points


def load_point_cloud_v2(camera_component: v2.CameraImageComponent,
                        lidar_camera_component: v2.LiDARCameraProjectionComponent,
                        lidar_component: v2.LiDARComponent,
                        lidar_pose_component: v2.LiDARPoseComponent,
                        lidar_calib_component: v2.LiDARCalibrationComponent,
                        laser_names):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for i, laser in enumerate(laser_names):
        # We might run into dimensionality problems here
        range_images[laser] = [tf.reshape(tf.convert_to_tensor(lidar_component.range_image_return1.values[i]), lidar_component.range_image_return1.shape[i])]
        if laser == v2._lidar.LaserName.TOP.value:
            # We might run into dimensionality problems here
            range_image_top_pose = tf.reshape(tf.convert_to_tensor(lidar_pose_component.range_image_return1.values[i]), lidar_pose_component.range_image_return1.shape[i])
        camera_projections[laser] = [tf.reshape(tf.convert_to_tensor(lidar_camera_component.range_image_return1.values[i]), lidar_camera_component.range_image_return1.shape[i])]
        range_images[laser].append(tf.reshape(tf.convert_to_tensor(lidar_component.range_image_return2.values[i]), lidar_component.range_image_return2.shape[i]))
        camera_projections[laser].append(tf.reshape(tf.convert_to_tensor(lidar_camera_component.range_image_return2.values[i]), lidar_camera_component.range_image_return2.shape[i]))
    

    points = []
    cp_points = []
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(camera_component.pose.transform), [4, 4]))

    range_image_top_pose_tensor = range_image_top_pose
    # range_image_top_pose_tensor = tf.reshape(
    #     tf.convert_to_tensor(value=range_image_top_pose.data),
    #     range_image_top_pose.shape.dims)

    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)
    
    for laser, ex_trans, beam_min, beam_max, beam_values in zip(laser_names, lidar_calib_component.extrinsic.transform, lidar_calib_component.beam_inclination.min, lidar_calib_component.beam_inclination.max, lidar_calib_component.beam_inclination.values):
        range_image = range_images[laser][0]
        if beam_values is not None:
            if len(beam_values) == 0:
                beam_inclinations = range_image_utils.compute_inclination(tf.constant([beam_min, beam_max]), height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(beam_values)
            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        else:
            beam_inclinations = range_image_utils.compute_inclination(tf.constant([beam_min, beam_max]), height=range_image.shape.dims[0])
        extrinsic = np.reshape(np.array(ex_trans), [4, 4])

        # range_image_tensor = tf.reshape(
        #     tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_tensor = range_image
        pixel_pose_local = None
        frame_pose_local = None
        if laser == v2._lidar.LaserName.TOP.value:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.constant(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)
        
        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        cartesian_range_images[laser] = range_image_cartesian
    
    for laser in laser_names:
        range_image = range_images[laser][0]
        range_image_tensor = range_image
        # range_image_tensor = tf.reshape(
        #     tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[laser]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                    tf.compat.v1.where(range_image_mask))
        
        cp = camera_projections[laser][0]
        cp_tensor = cp
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)
    
    return points_all, cp_points_all


def get_cropped_cam_data(img, keypoints, box):
    """_summary_

    Args:
        frame (open dataset frame): Frame of the tfrecord file.
        labels (waymo object labels): Labels of the pedestrian.
        cam (int): Camera in which the data can be found.

    Returns:
        tuple: Cropped image and new keypoint coordinates.
    """

    img = _imdecode(img)

    cropped_image, cropped_camera_keypoints = crop_camera_keypoints(
        img,
        keypoints,
        box,
        margin=0.3)
    # store cropped keypoints and image
    return cropped_image, cropped_camera_keypoints


def one_cam_has_keypoints(label):
    if label.camera:
        for elm in label.camera:
            if label.camera[elm].keypoints.keypoint:
                return True
    return False


def create_array(joints, three_dim=False):
    """Create array from labels dict

    Args:
        joints (dict): joints with keys from protos.
        three_dim (bool): true is 3D, false is 2D

    Returns:
        np.array: Array with the data, zeros inserted if joint is not available
    """

    if three_dim:
        j_arr = np.zeros(shape=(15, 4))
        for key in JOINT_KEYS:
            if key in joints['keypoints_3d']:
                j_arr[JOINT_KEYS[key]] = [joints['keypoints_3d'][key]['x'], joints['keypoints_3d'][key]['y'], joints['keypoints_3d'][key]['z'], joints['keypoints_3d'][key]['occluded']]

    else:
        j_arr = np.zeros(shape=(15, 3))
        for key in JOINT_KEYS:
            if key in joints['keypoints_2d']:
                j_arr[JOINT_KEYS[key]] = [joints['keypoints_2d'][key]['x'], joints['keypoints_2d'][key]['y'], joints['keypoints_2d'][key]['occluded']]

    return j_arr.astype('float32')


def store_lidar_and_projections(labels, label, labels_dict, cam):

    num_points_cp = 0

    box = box_utils.box_to_tensor(labels[label].laser.box)[tf.newaxis, :]
    box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
    box_cp_points_all = cp_points[box_utils.is_within_box_3d(points_all, box)[:, 0]]
    labels_dict[id]['lidar'] = box_points.astype('float32')

    # create array of same size and fill it with the lidar projections
    box_cp_points = np.zeros((box_cp_points_all.shape[0], 3))

    if cam in set(box_cp_points_all[:, 0]):
        # get indices that match the camera
        indices = np.where(box_cp_points_all[:, 0] == cam)
        box_cp_points[indices] = box_cp_points_all[:, [0, 1, 2]][indices]
        num_points_cp += len(indices[0])
    if cam in set(box_cp_points_all[:, 3]):
        indices = np.where(box_cp_points_all[:, 3] == cam)
        box_cp_points[indices] = box_cp_points_all[:, [3, 4, 5]][indices]
        num_points_cp += len(indices[0])

    # if num_points_cp != box_points.shape[0]:
    #     print("---")
    #     print("Problem occurred with camera projections, not all lidat points are projected to camera")
    #     print(f"ID:{id}")
    #     print(f"Num cp points: {num_points_cp}")
    #     print(f"Num Lidar points in box: {box_points.shape[0]}")
    #     print(f'Ratio cp to lidar: {num_points_cp/box_points.shape[0]}')
    #     print("---")

    # resize and store camera projections
    min_x = max(0, int(labels_dict[id]['bb_2d']['center_x'] - labels_dict[id]['img_2d_width'] / 2))
    min_y = max(0, int(labels_dict[id]['bb_2d']['center_y'] - labels_dict[id]['img_2d_height'] / 2))
    box_cp_points[:, 1] = box_cp_points[:, 1] - min_x
    box_cp_points[:, 2] = box_cp_points[:, 2] - min_y
    # reset points to zero that or not projected
    box_cp_points[box_cp_points[:, 0] == 0] = 0
    labels_dict[id]['cp_points'] = box_cp_points.astype('float32')
    if box_points.shape[0] > 0:
        labels_dict[id]['lidar_cp_points_ratio'] = num_points_cp/box_points.shape[0]
    else:
        labels_dict[id]['lidar_cp_points_ratio'] = False


def store_lidar_and_projections_v2(lidar_box_component: v2.LiDARBoxComponent,
                                   labels_dict,
                                   cam):
    num_points_cp = 0
    box = tf.constant([
      lidar_box_component.box.center.x, lidar_box_component.box.center.y, lidar_box_component.box.center.z, lidar_box_component.box.size.x, lidar_box_component.box.size.y, lidar_box_component.box.size.z, lidar_box_component.box.heading
  ])[tf.newaxis, :]
    if np.isnan(box.numpy()).any():
        labels_dict[id]['lidar_cp_points_ratio'] = False
        labels_dict[id]['lidar'] = np.nan
        labels_dict[id]['cp_points'] = np.nan
    else:
        box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
        box_cp_points_all = cp_points[box_utils.is_within_box_3d(points_all, box)[:, 0]]
        labels_dict[id]['lidar'] = box_points.astype('float32')

        # create array of same size and fill it with the lidar projections
        box_cp_points = np.zeros((box_cp_points_all.shape[0], 3))

        if cam in set(box_cp_points_all[:, 0]):
            # get indices that match the camera
            indices = np.where(box_cp_points_all[:, 0] == cam)
            box_cp_points[indices] = box_cp_points_all[:, [0, 1, 2]][indices]
            num_points_cp += len(indices[0])
        if cam in set(box_cp_points_all[:, 3]):
            indices = np.where(box_cp_points_all[:, 3] == cam)
            box_cp_points[indices] = box_cp_points_all[:, [3, 4, 5]][indices]
            num_points_cp += len(indices[0])

        # resize and store camera projections
        min_x = max(0, int(labels_dict[id]['bb_2d']['center_x'] - labels_dict[id]['img_2d_width'] / 2))
        min_y = max(0, int(labels_dict[id]['bb_2d']['center_y'] - labels_dict[id]['img_2d_height'] / 2))
        box_cp_points[:, 1] = box_cp_points[:, 1] - min_x
        box_cp_points[:, 2] = box_cp_points[:, 2] - min_y
        # reset points to zero that or not projected
        box_cp_points[box_cp_points[:, 0] == 0] = 0
        labels_dict[id]['cp_points'] = box_cp_points.astype('float32')
        if box_points.shape[0] > 0:
            labels_dict[id]['lidar_cp_points_ratio'] = num_points_cp/box_points.shape[0]
        else:
            labels_dict[id]['lidar_cp_points_ratio'] = False


def store_other_information(frame: v2.CameraImageComponent,
                            lidar_box: v2.LiDARBoxComponent,
                            camera_box: v2.CameraBoxComponent,
                            camera_calib: v2.CameraCalibrationComponent,
                            labels_dict, 
                            id, 
                            cam):

    # store 3D bb info
    labels_dict[id]['bb_3d'] = {
        "center_x": lidar_box.box.center.x,
        "center_y": lidar_box.box.center.y,
        "center_z": lidar_box.box.center.z,
        "width": lidar_box.box.size.x,
        "length": lidar_box.box.size.y,
        "height": lidar_box.box.size.z,
        "heading": lidar_box.box.heading
    }
    if cam != -1:
        labels_dict[id]['bb_2d'] = {
            "center_x": camera_box.box.center.x,
            "center_y": camera_box.box.center.y,
            "width": camera_box.box.size.x,
            "height": camera_box.box.size.y,
        }

    if 'keypoints_2d' in labels_dict[id]:
        # get all indices that are not labeled in the keypoints_2d_arr -> makes it easier to set them to zero after normalization
        labels_dict[id]['mask_2d'] = [JOINT_KEYS[k] for k in JOINT_KEYS.keys() if k not in labels_dict[id]['keypoints_2d'].keys()]

    if 'keypoints_3d' in labels_dict[id]:
        labels_dict[id]['mask_3d'] = [JOINT_KEYS[k] for k in JOINT_KEYS.keys() if k not in labels_dict[id]['keypoints_3d'].keys()]

    # store camera intrinsic, extrinsic and metadata
    if cam != -1:
        image = frame.image
        # cam_calib = next(c for c in frame.context.camera_calibrations if c.name == cam) --> not needed as we 
        labels_dict[id]['extrinsic'] = np.array([camera_calib.extrinsic.transform]).reshape(4, 4)
        labels_dict[id]['intrinsic'] = np.array([camera_calib.intrinsic.f_u, camera_calib.intrinsic.f_v, camera_calib.intrinsic.c_u, camera_calib.intrinsic.c_v, camera_calib.intrinsic.k1, camera_calib.intrinsic.k2, camera_calib.intrinsic.p1, camera_calib.intrinsic.p2, camera_calib.intrinsic.k3], dtype=np.float32)
        labels_dict[id]['metadata'] = np.array([camera_calib.width, camera_calib.height, camera_calib.rolling_shutter_direction], dtype=np.int32)

        camera_image_metadata = list(frame.pose.transform)
        camera_image_metadata.append(frame.velocity.linear_velocity.x)
        camera_image_metadata.append(frame.velocity.linear_velocity.y)
        camera_image_metadata.append(frame.velocity.linear_velocity.z)
        camera_image_metadata.append(frame.velocity.angular_velocity.x)
        camera_image_metadata.append(frame.velocity.angular_velocity.y)
        camera_image_metadata.append(frame.velocity.angular_velocity.z)
        camera_image_metadata.append(frame.pose_timestamp)
        camera_image_metadata.append(frame.rolling_shutter_params.shutter)
        camera_image_metadata.append(frame.rolling_shutter_params.camera_trigger_time)
        camera_image_metadata.append(frame.rolling_shutter_params.camera_readout_done_time)

        labels_dict[id]['camera_image_metadata'] = camera_image_metadata

        labels_dict[id]['frame_pose_transform'] = np.array(frame.pose.transform).reshape(4, 4)


if __name__ == "__main__":

    labels_dict_2d = {}
    labels_dict_3d_2d = {}
    labels_dict_3d = {}
    box_points_list_2d = []
    box_points_list_3d_2d = []
    box_points_list_3d = []
    counter_2d = 0
    counter_3d = 0
    counter_3d_2d = 0
    global_counter = 0
    image_segment_relations_2d = []
    image_segment_relations_3d = []
    image_segment_relations_3d_2d = []

    for folder in ["2D/", "3D/", "3D_2D/"]:
        if not os.path.exists(base_path + folder):
            if folder == "3D/":
                os.makedirs(base_path + folder)
            else:
                os.makedirs(base_path + folder + "images/")
    dir_counter = {"training/": {"2D": 0, '3D': 0, "3D_2D": 0}, "validation/": {"2D": 0, '3D': 0, "3D_2D": 0}, "testing/": {"2D": 0, '3D': 0, "3D_2D": 0}, "domain_adaptation/validation/": {"2D": 0, '3D': 0, "3D_2D": 0},
                   "domain_adaptation/training/": {"2D": 0, '3D': 0, "3D_2D": 0}, "testing_3d_camera_only_detection": {"2D": 0, '3D': 0, "3D_2D": 0}, "domain_adaptation/testing/": {"2D": 0, '3D': 0, "3D_2D": 0}}
    # Original dir_names
    # dir_names = ["training/", "validation/", "testing/", "domain_adaptation/validation/", "domain_adaptation/training/", "testing_3d_camera_only_detection", "domain_adaptation/testing/"]
    dir_names = ["training/", "validation/", "testing/", "testing_interactive/", "training_20s/", "validation_interactive/"]
    # dir_names = ["training/", "validation/"]
    # dir_names = ["validation/"]

    # New dir_names for new Waymo V2
    dir_counter = {"training": {"2D": 0, '3D': 0, "3D_2D": 0}, "validation": {"2D": 0, '3D': 0, "3D_2D": 0}}
    dir_names = ["training", "validation"]

    ##############################################################################################################################
    
    for dir_name in tqdm(dir_names):

        # get all TFRecord files
        print(f"Extracting {dir_name[0:-1]} data...")

        # Get the .parquet files of Waymo V2 for camera images (use this as base for iterating over the entire dataset)
        cam_img_paths = list(glob.glob(f'{tfr_path}' + dir_name + '/camera_image/' + '*.parquet'))
        # cam_box_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/camera_box/' + '*.parquet'))
        # lidar_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/lidar/' + '*.parquet'))
        # lidar_box_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/lidar_box/' + '*.parquet'))
        # association_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/camera_to_lidar_box_association/' + '*.parquet'))

        for frame_path in tqdm(cam_img_paths):

            # Read the .parquet files
            cam_img = dd.read_parquet(frame_path)
            cam_box_path = frame_path.replace('camera_image', 'camera_box')
            cam_box = dd.read_parquet(cam_box_path)
            cam_hkp_path = frame_path.replace('camera_image', 'camera_hkp')
            cam_hkp = dd.read_parquet(cam_hkp_path)
            cam_calib_path = frame_path.replace('camera_image', 'camera_calibration')
            cam_calib = dd.read_parquet(cam_calib_path)
            lidar_path = frame_path.replace('camera_image', 'lidar')
            lidar = dd.read_parquet(lidar_path)
            lidar_box_path = frame_path.replace('camera_image', 'lidar_box')
            lidar_box = dd.read_parquet(lidar_box_path)
            lidar_hkp_path = frame_path.replace('camera_image', 'lidar_hkp')
            lidar_hkp = dd.read_parquet(lidar_hkp_path)
            association_path = frame_path.replace('camera_image', 'camera_to_lidar_box_association')
            association = dd.read_parquet(association_path)
            lidar_cam_proj_path = frame_path.replace('camera_image', 'lidar_camera_projection')
            lidar_cam_proj = dd.read_parquet(lidar_cam_proj_path)
            lidar_pose_path = frame_path.replace('camera_image', 'lidar_pose')
            lidar_pose = dd.read_parquet(lidar_pose_path)
            lidar_calib_path = frame_path.replace('camera_image', 'lidar_calibration')
            lidar_calib = dd.read_parquet(lidar_calib_path)

            # Merge data frames such that we have rows that correspond to objects that either have only 2D keypoints, only 3D keypoints, or both types of keypoints
            cam_hkp = v2.merge(cam_hkp, cam_img)
            cam_hkp = v2.merge(cam_hkp, cam_box)
            cam_hkp = v2.merge(cam_hkp, cam_calib)
            cam_hkp = v2.merge(cam_hkp, association, right_nullable=True)
            lidar_hkp = v2.merge(lidar_hkp, lidar_box, right_nullable=True)
            #lidar_hkp = v2.merge(lidar_hkp, lidar, right_group=True)
            cam_lidar_hkp = v2.merge(cam_hkp, lidar_hkp, left_nullable=True, right_nullable=True)
            lidar = v2.merge(lidar, lidar_cam_proj, right_nullable=True)
            lidar = v2.merge(lidar, lidar_pose, right_nullable=True)
            lidar = v2.merge(lidar, lidar_calib, right_nullable=True)
            cam_lidar_hkp = v2.merge(cam_lidar_hkp, lidar, right_nullable=True, right_group=True)

            # Compute the generate data frame
            computed_lidar_cam_hkp = cam_lidar_hkp.compute()

            # List for storing timestamps of saved point clouds, so a point cloud is not saved multiple times
            pc_timestamps = {}
            missing_timestamps = []

            for index, row in computed_lidar_cam_hkp.iterrows():
                print(row)
                # Create components from row
                camera_hkp_component = v2.CameraHumanKeypointsComponent.from_dict(row)
                camera_box_component = v2.CameraBoxComponent.from_dict(row)
                camera_image_component = v2.CameraImageComponent.from_dict(row)
                camera_calib_component = v2.CameraCalibrationComponent.from_dict(row)
                lidar_box_component = v2.LiDARBoxComponent.from_dict(row)
                lidar_hkp_component = v2.LiDARHumanKeypointsComponent.from_dict(row)
                lidar_component = v2.LiDARComponent.from_dict(row)
                lidar_pose_component = v2.LiDARPoseComponent.from_dict(row)
                lidar_calib_component = v2.LiDARCalibrationComponent.from_dict(row)
                lidar_cam_proj_component = v2.LiDARCameraProjectionComponent.from_dict(row)

                # Load point cloud once for the given timestamp
                # As we do not have camera data for all objects (lidar objects), we don't generate it (we need the frame pose for this function)
                if row['key.frame_timestamp_micros'] not in pc_timestamps.keys():
                    if not math.isnan(row['key.camera_name']):
                        points_all, cp_points = load_point_cloud_v2(camera_image_component, lidar_cam_proj_component, lidar_component, lidar_pose_component, lidar_calib_component, row['key.laser_name'])
                        pc_timestamps[row['key.frame_timestamp_micros']] = [points_all, cp_points]
                        if row['key.frame_timestamp_micros'] in missing_timestamps:
                            missing_timestamps.remove(row['key.frame_timestamp_micros'])
                    else:
                        missing_timestamps.append(row['key.frame_timestamp_micros'])
                        points_all = None
                        cp_points = None
                else:
                    points_all, cp_points = pc_timestamps[row['key.frame_timestamp_micros']]

                # 3D and 2D keypoints are available
                if not np.isnan([camera_hkp_component.camera_keypoints.type]).any() and not np.isnan([lidar_hkp_component.lidar_keypoints.type]).any():
                # if isinstance(camera_hkp_component.camera_keypoints.type, list) and isinstance(lidar_hkp_component.lidar_keypoints.type, list):
                    id = str(row['key.frame_timestamp_micros']) + "_" + str(row['key.camera_name']) + "_" + str(row['key.camera_object_id'])

                    labels_dict_3d_2d[id] = {}
                    counter_3d_2d += 1

                    # Crop camera image and keypoints
                    cropped_image, cropped_camera_keypoints = get_cropped_cam_data(camera_image_component.image, camera_hkp_component, camera_box_component.box)
                    img_height, img_width, _ = cropped_image.shape

                    # save image data to file
                    cv2.imwrite(base_path + "3D_2D/images/" + id + ".jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                    labels_dict_3d_2d[id]['keypoints_2d'] = {}
                    for joint, x, y, vis in zip(cropped_camera_keypoints.camera_keypoints.type, cropped_camera_keypoints.camera_keypoints.keypoint_2d.location_px.x, cropped_camera_keypoints.camera_keypoints.keypoint_2d.location_px.y, cropped_camera_keypoints.camera_keypoints.keypoint_2d.visibility.is_occluded):
                        labels_dict_3d_2d[id]['keypoints_2d'][joint] = {'x': x, 'y': y, 'occluded': vis}
                    
                    labels_dict_3d_2d[id]['img_2d_height'] = img_height
                    labels_dict_3d_2d[id]['img_2d_width'] = img_width

                    # Get full image keypoints as well (I don't know if this is used anywhere)
                    labels_dict_3d_2d[id]['keypoints_2d_image'] = {}
                    for joint, x, y, vis in zip(camera_hkp_component.camera_keypoints.type, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.x, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.y, camera_hkp_component.camera_keypoints.keypoint_2d.visibility.is_occluded):
                        labels_dict_3d_2d[id]['keypoints_2d_image'][joint] = {'x': x, 'y': y, 'occluded': vis}
                    
                    # Store 3D Keypoints
                    labels_dict_3d_2d[id]['keypoints_3d'] = {}
                    for joint, x, y, z, vis in zip(lidar_hkp_component.lidar_keypoints.type, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.x, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.y, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.z, lidar_hkp_component.lidar_keypoints.keypoint_3d.visibility.is_occluded):
                        labels_dict_3d_2d[id]['keypoints_3d'][joint] = {'x': x, 'y': y, 'z': z, 'occluded': vis}
                    
                    labels_dict_3d_2d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d_2d[id], three_dim=True)
                    labels_dict_3d_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_3d_2d[id], three_dim=False)
                    image_segment_relations_3d_2d.append([id, row['key.frame_timestamp_micros'], row['key.camera_object_id'], row['key.camera_name'], frame_path])
                    store_other_information(camera_image_component, lidar_box_component, camera_box_component, camera_calib_component, labels_dict_3d_2d, id, row['key.camera_name'])
                    
                    store_lidar_and_projections_v2(lidar_box_component, labels_dict_3d_2d, row['key.camera_name'])
                    box_points_list_3d_2d.append(labels_dict_3d_2d[id]['lidar'].shape[0])
                    global_counter += 1
                    dir_counter[dir_name]['3D_2D'] += 1
                
                # Only 3D keypoints available
                elif np.isnan(camera_hkp_component.camera_keypoints.type).any() and not np.isnan([lidar_hkp_component.lidar_keypoints.type]).any():
                # elif not isinstance(camera_hkp_component.camera_keypoints.type, list) and isinstance(lidar_hkp_component.lidar_keypoints.type, list):
                    if math.isnan(row['key.camera_name']):
                        cam = -1
                        id = str(row['key.frame_timestamp_micros']) + "_" + str(row['key.laser_name'][0]) + "_" + str(row['key.key.laser_object_id'])
                    else:
                        id = str(row['key.frame_timestamp_micros']) + "_" + str(row['key.camera_name']) + "_" + str(row['key.camera_object_id'])
                        cam = 0
                    labels_dict_3d[id]= {}
                    counter_3d += 1

                    # 3D Keypoints
                    labels_dict_3d[id]['keypoints_3d'] = {}
                    for joint, x, y, z, vis in zip(lidar_hkp_component.lidar_keypoints.type, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.x, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.y, lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.z, lidar_hkp_component.lidar_keypoints.keypoint_3d.visibility.is_occluded):
                        labels_dict_3d[id]['keypoints_3d'][joint] = {'x': x, 'y': y, 'z': z, 'occluded': vis}
                    
                    labels_dict_3d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d[id], three_dim=True)
                    image_segment_relations_3d.append([id, row['key.frame_timestamp_micros'], row['key.laser_object_id'][0], row['key.laser_name'][0], frame_path])
                    store_other_information(camera_image_component, lidar_box_component, camera_box_component, camera_calib_component, labels_dict_3d, id, cam=cam)

                    # Point cloud data is already stored
                    # Check if data is available
                    if points_all is not None:
                        box = box_utils.box_to_tensor(lidar_box_component.box)[tf.newaxis, :]
                        box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
                        labels_dict_3d[id]['lidar'] = box_points.astype('float32')
                        box_points_list_3d.append(box_points.shape[0])
                    else:
                        labels_dict_3d[id]['lidar'] = np.array([])
                        box_points_list_3d.append(0)

                    global_counter += 1
                    dir_counter[dir_name]['3D'] += 1
                
                # Only 2D keypoints available
                elif not np.isnan([camera_hkp_component.camera_keypoints.type]).any() and np.isnan(lidar_hkp_component.lidar_keypoints.type).any():
                # elif isinstance(camera_hkp_component.camera_keypoints.type, list) and not isinstance(lidar_hkp_component.lidar_keypoints.type, list):
                    id = str(row['key.frame_timestamp_micros']) + "_" + str(row['key.camera_name']) + "_" + str(row['key.camera_object_id'])

                    cropped_image, cropped_camera_keypoints = get_cropped_cam_data(camera_image_component.image, camera_hkp_component, camera_box_component.box)
                    img_height, img_width, _ = cropped_image.shape

                    labels_dict_2d[id] = {}
                    counter_2d += 1
                    labels_dict_2d[id]['keypoints_2d'] = {}
                    for joint, x, y, vis in zip(cropped_camera_keypoints.camera_keypoints.type, cropped_camera_keypoints.camera_keypoints.keypoint_2d.location_px.x, cropped_camera_keypoints.camera_keypoints.keypoint_2d.location_px.y, cropped_camera_keypoints.camera_keypoints.keypoint_2d.visibility.is_occluded):
                        labels_dict_2d[id]['keypoints_2d'][joint] = {'x': x, 'y': y, 'occluded': vis}
                    
                    # Get full image keypoints as well, I don't know where this is used
                    labels_dict_2d[id]['keypoints_2d_image'] = {}
                    for joint, x, y, vis in zip(camera_hkp_component.camera_keypoints.type, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.x, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.y, camera_hkp_component.camera_keypoints.keypoint_2d.visibility.is_occluded):
                        labels_dict_2d[id]['keypoints_2d_image'][joint] = {'x': x, 'y': y, 'occluded': vis}
                    labels_dict_2d[id]['img_2d_height'] = img_height
                    labels_dict_2d[id]['img_2d_width'] = img_width

                    labels_dict_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_2d[id], three_dim=False)
                    image_segment_relations_2d.append([id, row['key.frame_timestamp_micros'], row['key.camera_object_id'], row['key.camera_name'], frame_path])
                    store_other_information(camera_image_component, lidar_box_component, camera_box_component, camera_calib_component, labels_dict_2d, id, row['key.camera_name'])

                    # Save image data to file
                    cv2.imwrite(base_path + "2D/images/" + id + ".jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                    store_lidar_and_projections_v2(lidar_box_component, labels_dict_2d, row['key.camera_name'])
                    if np.isnan(labels_dict_2d[id]['lidar']).any():
                        box_points_list_2d.append(0)
                    else:
                        box_points_list_2d.append(labels_dict_2d[id]['lidar'].shape[0])
                    global_counter += 1
                    dir_counter[dir_name]['2D'] += 1
                print("Missing point cloud data for timestamps in context {}: ", frame_path)
                print(missing_timestamps)


    # save everything to file
    print("Directory counts:")
    print(dir_counter)
    info_dict = {
        "global_counter": global_counter,
        "counter_2d": counter_2d,
        "counter_3d": counter_3d,
        "counter_3d_2d": counter_3d_2d,
        "dir_counts": dir_counter
    }

    with open(base_path + 'info.json', 'w') as fp:
        json.dump(info_dict, fp)

    for folder in ["2D/", "3D/", "3D_2D/"]:

        if folder == "2D/":
            data = labels_dict_2d
            box_points_list = np.array(box_points_list_2d)
            image_segment_relations = image_segment_relations_2d
        elif folder == "3D/":
            data = labels_dict_3d
            box_points_list = np.array(box_points_list_3d)
            image_segment_relations = image_segment_relations_3d
        else:
            data = labels_dict_3d_2d
            box_points_list = np.array(box_points_list_3d_2d)
            image_segment_relations = image_segment_relations_3d_2d

        with open(base_path + folder + 'labels.pkl', 'wb')as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(base_path + folder + 'image_segment_relations.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['image_id', 'frame', 'id', 'cam', 'segment'])

        with open(base_path + folder + 'image_segment_relations.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            for elm in image_segment_relations:
                filewriter.writerow(elm)

        np.save(base_path + folder + 'lidar_point_stats.npy', box_points_list)

        # checks
        if len(image_segment_relations) != len(data):
            print("Lengths of image_segment_relations and dict do not match. Please.")
            print(f"image_segment_relations: {len(image_segment_relations)} data points.")
            print(f"dict: {len(data)} data points.")
            print(f"Problem occurred for data in: {dir_name}.")

        if len(box_points_list) != len(data):
            print("Lengths of box_points_list and dict do not match. Please.")
            print(f"box_points_list: {len(box_points_list)} data points.")
            print(f"dict: {len(data)} data points.")
            print(f"Problem occurred for data in: {dir_name}.")


    ####################################################################################################

    # for dir_name in tqdm(dir_names):

    #     # get all TFRecord files
    #     print(f"Extracting {dir_name[0:-1]} data...")

    #     # Get the .parquet files of Waymo V2 for camera images (use this as base for iterating over the entire dataset)
    #     cam_img_paths = list(glob.glob(f'{tfr_path}' + dir_name + '/camera_image/' + '*.parquet'))
    #     # cam_box_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/camera_box/' + '*.parquet'))
    #     # lidar_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/lidar/' + '*.parquet'))
    #     # lidar_box_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/lidar_box/' + '*.parquet'))
    #     # association_paths = list(glob.glob(f'{tfr_path}' + '/' + dir_name + '/camera_to_lidar_box_association/' + '*.parquet'))

    #     for frame_path in tqdm(cam_img_paths):

    #         # Read the .parquet files
    #         cam_img = dd.read_parquet(frame_path)
    #         cam_box_path = frame_path.replace('camera_image', 'camera_box')
    #         cam_box = dd.read_parquet(cam_box_path)
    #         cam_hkp_path = frame_path.replace('camera_image', 'camera_hkp')
    #         cam_hkp = dd.read_parquet(cam_hkp_path)
    #         cam_calib_path = frame_path.replace('camera_image', 'camera_calibration')
    #         cam_calib = dd.read_parquet(cam_calib_path)
    #         lidar_path = frame_path.replace('camera_image', 'lidar')
    #         lidar = dd.read_parquet(lidar_path)
    #         lidar_box_path = frame_path.replace('camera_image', 'lidar_box')
    #         lidar_box = dd.read_parquet(lidar_box_path)
    #         lidar_hkp_path = frame_path.replace('camera_image', 'lidar_hkp')
    #         lidar_hkp = dd.read_parquet(lidar_hkp_path)
    #         association_path = frame_path.replace('camera_image', 'camera_to_lidar_box_association')
    #         association = dd.read_parquet(association_path)
    #         lidar_cam_proj_path = frame_path.replace('camera_image', 'lidar_camera_projection')
    #         lidar_cam_proj = dd.read_parquet(lidar_cam_proj_path)
    #         lidar_pose_path = frame_path.replace('camera_image', 'lidar_pose')
    #         lidar_pose = dd.read_parquet(lidar_pose_path)
    #         lidar_calib_path = frame_path.replace('camera_image', 'lidar_calibration')
    #         lidar_calib = dd.read_parquet(lidar_calib_path)

    #         # Merge camera-lidar projections (range images) and lidar pose
    #         merged_proj_pose = v2.merge(lidar_cam_proj, lidar_pose)
    #         # Add LiDAR data to this data frame as it includes the range images
    #         merged_proj_pose = v2.merge(merged_proj_pose, lidar)
    #         # Add LiDAR calibration data to this data frame
    #         merged_proj_pose = v2.merge(merged_proj_pose, lidar_calib)

    #         # Merge the files
    #         cam_img_w_box = v2.merge(cam_box, cam_img)
    #         cam_img_w_box_w_hkp = v2.merge(cam_img_w_box, cam_hkp, right_nullable=True)
    #         cam_data = v2.merge(cam_img_w_box_w_hkp, association, right_nullable=True)
    #         cam_data = v2.merge(cam_data, cam_calib)
    #         lidar_data = v2.merge(lidar, lidar_box, right_nullable=True)
    #         lidar_w_hkp = v2.merge(lidar_data, lidar_hkp, right_nullable=True, left_group= True, right_group=True)

    #         computed_cam_data = cam_data.compute()

    #         # List for storing timestamps of saved point clouds, so a point cloud is not saved multiple times
    #         pc_timestamps = {}

    #         # Save LiDAR object ids for coresponding context
    #         lidar_object_ids = []
    #         counter = 0
    #         other_counter = 0
    #         for index, row in computed_cam_data.iterrows():
    #             print(row['[CameraBoxComponent].type'])
    #             counter += 1
    #             if row['[CameraBoxComponent].type'] == 2 or row['[CameraBoxComponent].type'] == 4:
    #                 print(row)
    #                 other_counter += 1
    #         print(counter)
    #         print(other_counter)

    #         # Iterate through rows of camera data first (each row corresponds to a single object at a single frame)
    #         for index, row in computed_cam_data.iterrows():
                
    #             print(row)
    #             # Create components from row
    #             camera_hkp_component = v2.CameraHumanKeypointsComponent.from_dict(row)
    #             camera_box_component = v2.CameraBoxComponent.from_dict(row)
    #             camera_image_component = v2.CameraImageComponent.from_dict(row)
    #             camera_calib_component = v2.CameraCalibrationComponent.from_dict(row)
    #             # lidar_box_component = v2.LiDARBoxComponent.from_dict(row)
    #             # lidar_hkp_component = v2.LiDARHumanKeypointsComponent.from_dict(row)
    #             # lidar_component = v2.LiDARComponent.from_dict(row)

    #             # Lidar range image, pose and calibration data
    #             range_image_and_pose_row = merged_proj_pose[merged_proj_pose['key.frame_timestamp_micros'] == row['key.frame_timestamp_micros']].compute()
    #             lidar_cam_proj_component = v2.LiDARCameraProjectionComponent.from_dict(range_image_and_pose_row)
    #             lidar_pose_component = v2.LiDARPoseComponent.from_dict(range_image_and_pose_row)
    #             lidar_range_image_component = v2.LiDARComponent.from_dict(range_image_and_pose_row)
    #             lidar_calib_component = v2.LiDARCalibrationComponent.from_dict(range_image_and_pose_row)
    #             print(range_image_and_pose_row['key.laser_name'])

    #             # Load LiDAR point cloud once for the given context and timestamp
    #             if row['key.frame_timestamp_micros'] not in pc_timestamps.keys():
    #                 points_all, cp_points = load_point_cloud_v2(camera_image_component, lidar_cam_proj_component, lidar_range_image_component, lidar_pose_component, lidar_calib_component, row)
    #                 pc_timestamps[row['key.frame_timestamp_micros']] = (points_all, cp_points)

    #             # Old iteration over frames
    #             # frame = dataset_pb2.Frame()
    #             # pc = False  # prevent from reading same point cloud multiple times
    #             # frame.ParseFromString(bytearray(data.numpy()))
    #             # labels = keypoint_data.group_object_labels(frame)
    #             test = camera_hkp_component.camera_keypoints
    #             #Check if the object is a pedestrian or a cyclist
    #             if (camera_box_component.type == v2._box.BoxType.TYPE_PEDESTRIAN) or (camera_box_component.type == v2._box.BoxType.TYPE_CYCLIST):
    #                 #Check if there are 2D keypoints and 3D keypoints available in the camera_hkp_component
    #                 if not math.isnan(camera_hkp_component.camera_keypoints.keypoint_2d.location_px.x) and not math.isnan(camera_hkp_component.camera_keypoints.keypoint_3d.location_m.x):
    #                     print("Both")
    #                     id = str(row['key.camera_object_id']) + "_" + str(row['key.camera_name']) + "_" + str(camera_box_component.type)
    #                     labels_dict_3d_2d[id] = {}
    #                     counter_3d_2d += 1

    #                     cropped_image, cropped_camera_keypoints = get_croped_cam_data(camera_image_component.image, camera_hkp_component.camera_keypoints, camera_box_component.box)
    #                     img_height, img_width, _ = cropped_image.shape

    #                     # save image data to file
    #                     cv2.imwrite(base_path + "3D_2D/images/" + id + ".jpg",  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    #                     labels_dict_3d_2d[id]['keypoints_2d'] = {}
    #                     # TODO: Continue with writing the keypoints to the labels_dict
    #                     for type, x, y, occlusion in zip(cropped_camera_keypoints.type, cropped_camera_keypoints.keypoint_2d.location_px.x, cropped_camera_keypoints.keypoint_2d.location_px.y, cropped_camera_keypoints.keypoint_2d.visibility.is_occluded):
    #                         labels_dict_3d_2d[id]['keypoints_2d'][type] = {'x': x,
    #                                                                             'y': y,
    #                                                                             'occluded': occlusion,
    #                                                                             }
    #                     labels_dict_3d_2d[id]['img_2d_height'] = img_height
    #                     labels_dict_3d_2d[id]['img_2d_width'] = img_width

    #                     # get full image keypoints as well (later maybe usefull)
    #                     labels_dict_3d_2d[id]['keypoints_2d_image'] = {}
    #                     for type, x, y, occlusion in zip(camera_hkp_component.camera_keypoints.type, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.x, camera_hkp_component.camera_keypoints.keypoint_2d.location_px.y, camera_hkp_component.camera_keypoints.keypoint_2d.visibility.is_occluded):
    #                         labels_dict_3d_2d[id]['keypoints_2d_image'][type] = {'x': x,  # labels[label].camera[cam].box.length/256 + left,
    #                                                                                'y': y,  # * labels[label].camera[cam].box.width/256 + bottom,
    #                                                                                'occluded': occlusion}
                        
    #                     # store 3d keypoints
    #                     labels_dict_3d_2d[id]['keypoints_3d'] = {}
    #                     for type, x, y, z, occlusion in zip(camera_hkp_component.camera_keypoints.type, camera_hkp_component.camera_keypoints.keypoint_3d.location_m.x, camera_hkp_component.camera_keypoints.keypoint_3d.location_m.y, camera_hkp_component.camera_keypoints.keypoint_3d.location_m.z, camera_hkp_component.camera_keypoints.keypoint_3d.visibility.is_occluded):
    #                         labels_dict_3d_2d[id]['keypoints_3d'][type] = {'x': x,
    #                                                                             'y': y,
    #                                                                             'z': z,
    #                                                                             'occluded': occlusion}
                        
    #                     labels_dict_3d_2d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d_2d[id], three_dim=True)
    #                     labels_dict_3d_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_3d_2d[id])
    #                     image_segment_relations_3d_2d.append([id, camera_box_component.key.camera_object_id, camera_box_component.type, camera_box_component.key.camera_name,  frame_path])
    #                     # Get the corresponding LiDAR data
    #                     corresponding_lidar_row = lidar_w_hkp[lidar_w_hkp['key.lidar_object_id'] == row['key.lidar_object_id']].compute()
    #                     lidar_component = v2.LiDARComponent.from_dict(corresponding_lidar_row)
    #                     lidar_box_component = v2.LiDARBoxComponent.from_dict(corresponding_lidar_row)
    #                     lidar_hkp_component = v2.LiDARHumanKeypointsComponent.from_dict(corresponding_lidar_row)
    #                     store_other_information(camera_image_component, lidar_box_component, camera_box_component, camera_calib_component, labels_dict_3d_2d, id, row['key.camera_name'])

    #                     # Loading point clouds is done at the beginning of the loop through the camera objects within the context
    #                     store_lidar_and_projections_v2(lidar_box_component, labels_dict_3d_2d, row['key.camera_name'])
    #                     box_points_list_3d_2d.append(labels_dict_3d_2d[id])
    #                     global_counter += 1
    #                     dir_counter[dir_name]["3D_2D"] += 1
    #                 elif not math.isnan(camera_hkp_component.camera_keypoints.keypoint_2d.location_px.x):
    #                     # Check if LiDAR data has keypoints
    #                     if not math.isnan(row['key.lidar_object_id']):
    #                         # Build LiDAR components
    #                         corresponding_lidar_row = lidar_w_hkp[lidar_w_hkp['key.lidar_object_id'] == row['key.lidar_object_id']].compute()
    #                         lidar_component = v2.LiDARComponent.from_dict(corresponding_lidar_row)
    #                         lidar_box_component = v2.LiDARBoxComponent.from_dict(corresponding_lidar_row)
    #                         lidar_hkp_component = v2.LiDARHumanKeypointsComponent.from_dict(corresponding_lidar_row)
    #                         if not math.isnan(lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.x):
    #                             print("2D Camera and 3D LiDAR keypoints")
    #                     else:
    #                         print("Only 2D Keypoints")
    #                 elif not math.isnan(camera_hkp_component.camera_keypoints.keypoint_3d.location_m.x):
    #                     print("Only 3D Camera Keypoints")
    #                 # Finally check only LiDAR keypoints
    #                 elif not math.isnan(row['key.lidar_object_id']):
    #                     # Build LiDAR components
    #                     corresponding_lidar_row = lidar_w_hkp[lidar_w_hkp['key.lidar_object_id'] == row['key.lidar_object_id']].compute()
    #                     lidar_component = v2.LiDARComponent.from_dict(corresponding_lidar_row)
    #                     lidar_box_component = v2.LiDARBoxComponent.from_dict(corresponding_lidar_row)
    #                     lidar_hkp_component = v2.LiDARHumanKeypointsComponent.from_dict(corresponding_lidar_row)
    #                     if not math.isnan(lidar_hkp_component.lidar_keypoints.keypoint_3d.location_m.x):
    #                             print("Only LiDAR keypoints!")

    #         print("Done")
                
    #         for index, row in computed_cam_data.iterrows():
    #             # extract pedestrians only | load all information into one file per frame
    #             for label in labels:

    #                 if (labels[label].object_type == label_pb2.Label.TYPE_PEDESTRIAN) or (labels[label].object_type == label_pb2.Label.TYPE_CYCLIST):
    #                     # 3d and 2d label available
    #                     if labels[label].laser.keypoints.keypoint and one_cam_has_keypoints(labels[label]):

    #                         # iterate over cameras --> Note that this produces same 3D points, but different 2D annotations
    #                         for cam in labels[label].camera:
    #                             # prevent from storing cam data that does not have keypoints
    #                             if labels[label].camera[cam].keypoints.keypoint:

    #                                 id = str(frame_counter) + "_" + str(cam) + "_" + label

    #                                 labels_dict_3d_2d[id] = {}
    #                                 counter_3d_2d += 1

    #                                 cropped_image, cropped_camera_keypoints = get_croped_cam_data(frame, cam, labels[label])
    #                                 img_height, img_width, _ = cropped_image.shape

    #                                 # save image data to file
    #                                 cv2.imwrite(base_path + "3D_2D/images/" + id + ".jpg",  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    #                                 labels_dict_3d_2d[id]['keypoints_2d'] = {}
    #                                 for joint in cropped_camera_keypoints:
    #                                     labels_dict_3d_2d[id]['keypoints_2d'][joint.type] = {'x': joint.keypoint_2d.location_px.x,
    #                                                                                          'y': joint.keypoint_2d.location_px.y,
    #                                                                                          'occluded': joint.keypoint_2d.visibility.is_occluded,
    #                                                                                          }
    #                                 labels_dict_3d_2d[id]['img_2d_height'] = img_height
    #                                 labels_dict_3d_2d[id]['img_2d_width'] = img_width

    #                                 # get full image keypoints as well (later maybe usefull)
    #                                 labels_dict_3d_2d[id]['keypoints_2d_image'] = {}
    #                                 for joint in labels[label].camera[cam].keypoints.keypoint:
    #                                     labels_dict_3d_2d[id]['keypoints_2d_image'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
    #                                                                                                'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
    #                                                                                                'occluded': joint.keypoint_2d.visibility.is_occluded}

    #                                 # store 3d keypoints
    #                                 labels_dict_3d_2d[id]['keypoints_3d'] = {}
    #                                 for joint in labels[label].laser.keypoints.keypoint:
    #                                     labels_dict_3d_2d[id]['keypoints_3d'][joint.type] = {'x': joint.keypoint_3d.location_m.x,
    #                                                                                          'y': joint.keypoint_3d.location_m.y,
    #                                                                                          'z': joint.keypoint_3d.location_m.z,
    #                                                                                          'occluded': joint.keypoint_3d.visibility.is_occluded}

    #                                 labels_dict_3d_2d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d_2d[id], three_dim=True)
    #                                 labels_dict_3d_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_3d_2d[id])
    #                                 image_segment_relations_3d_2d.append([id, frame_counter, label, cam,  frame_path])
    #                                 store_other_information(frame, labels_dict_3d_2d, id, labels, cam)
    #                                 # load pc and store data
    #                                 if not pc:
    #                                     points_all, cp_points = load_point_cloud(frame)
    #                                     pc = True
    #                                 store_lidar_and_projections(labels, label, labels_dict_3d_2d, cam)
    #                                 box_points_list_3d_2d.append(labels_dict_3d_2d[id]['lidar'].shape[0])
    #                                 global_counter += 1
    #                                 dir_counter[dir_name]['3D_2D'] += 1

    #                     # only 3d label available
    #                     elif labels[label].laser.keypoints.keypoint and (not one_cam_has_keypoints(labels[label])):

    #                         if not labels[label].camera:
    #                             cam = -1
    #                         else:
    #                             # just take the first one as referece
    #                             cam = list(labels[label].camera.keys())[0]

    #                         id = str(frame_counter) + "_" + str(cam) + "_" + label

    #                         labels_dict_3d[id] = {}
    #                         counter_3d += 1

    #                         # store 2d keypoints
    #                         labels_dict_3d[id]['keypoints_3d'] = {}
    #                         for joint in labels[label].laser.keypoints.keypoint:
    #                             labels_dict_3d[id]['keypoints_3d'][joint.type] = {'x': joint.keypoint_3d.location_m.x,
    #                                                                               'y': joint.keypoint_3d.location_m.y,
    #                                                                               'z': joint.keypoint_3d.location_m.z,
    #                                                                               'occluded': joint.keypoint_3d.visibility.is_occluded}

    #                         labels_dict_3d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d[id], three_dim=True)
    #                         image_segment_relations_3d.append([id, frame_counter, label, cam,  frame_path])
    #                         store_other_information(frame, labels_dict_3d, id, labels, cam)

    #                         # load pc and store data
    #                         if not pc:
    #                             points_all, cp_points = load_point_cloud(frame)
    #                             pc = True
    #                         box = box_utils.box_to_tensor(labels[label].laser.box)[tf.newaxis, :]
    #                         box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
    #                         labels_dict_3d[id]['lidar'] = box_points.astype('float32')
    #                         box_points_list_3d.append(box_points.shape[0])
    #                         global_counter += 1
    #                         dir_counter[dir_name]['3D'] += 1

    #                     # check 2d keypoints in cameras
    #                     elif labels[label].camera:
    #                         for cam in labels[label].camera:
    #                             if labels[label].camera[cam].keypoints.keypoint:

    #                                 id = str(frame_counter) + "_" + str(cam) + "_" + label

    #                                 # Dimensions of the box. length: dim x. width: dim y. height: dim z.
    #                                 # length = labels[label].camera[cam].box.length
    #                                 # width = labels[label].camera[cam].box.width

    #                                 # if width < length:
    #                                 #     labels[label].camera[cam].box.width = length
    #                                 # else:
    #                                 #     labels[label].camera[cam].box.length = width

    #                                 cropped_image, cropped_camera_keypoints = get_croped_cam_data(frame, cam, labels[label])
    #                                 img_height, img_width, _ = cropped_image.shape

    #                                 # cropped_image, cropped_camera_keypoints = crop_camera_keypoints(
    #                                 #     img,
    #                                 #     labels[label].camera[cam].keypoints.keypoint,
    #                                 #     labels[label].camera[cam].box,
    #                                 #     margin=0)

    #                                 # check if croping worked properly, otherwise add padding to it
    #                                 # ratio = cropped_image.shape[0]/cropped_image.shape[1]
    #                                 # bottom = 0
    #                                 # left = 0
    #                                 # if not (0.97 < ratio < 1.03):
    #                                 #     if ratio > 1:
    #                                 #         left = cropped_image.shape[0] - cropped_image.shape[1]
    #                                 #     else:
    #                                 #         bottom = cropped_image.shape[1] - cropped_image.shape[0]
    #                                 #     cropped_image = cv2.copyMakeBorder(cropped_image, 0, bottom, left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    #                                 # # resize image to output format
    #                                 # res_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    #                                 labels_dict_2d[id] = {}
    #                                 counter_2d += 1
    #                                 labels_dict_2d[id]['keypoints_2d'] = {}
    #                                 for joint in cropped_camera_keypoints:
    #                                     labels_dict_2d[id]['keypoints_2d'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
    #                                                                                       'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
    #                                                                                       'occluded': joint.keypoint_2d.visibility.is_occluded}
    #                                 # get full image keypoints as well (later maybe usefull)
    #                                 labels_dict_2d[id]['keypoints_2d_image'] = {}
    #                                 for joint in labels[label].camera[cam].keypoints.keypoint:
    #                                     labels_dict_2d[id]['keypoints_2d_image'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
    #                                                                                             'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
    #                                                                                             'occluded': joint.keypoint_2d.visibility.is_occluded}
    #                                 labels_dict_2d[id]['img_2d_height'] = img_height
    #                                 labels_dict_2d[id]['img_2d_width'] = img_width

    #                                 labels_dict_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_2d[id])
    #                                 image_segment_relations_2d.append([id, frame_counter, label, cam, frame_path])
    #                                 store_other_information(frame, labels_dict_2d, id, labels, cam)

    #                                 # save image data to file
    #                                 cv2.imwrite(base_path + "2D/images/" + id + ".jpg",  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    #                                 # load point cloud data
    #                                 if not pc:
    #                                     points_all, cp_points = load_point_cloud(frame)
    #                                     pc = True
    #                                 store_lidar_and_projections(labels, label, labels_dict_2d, cam)
    #                                 box_points_list_2d.append(labels_dict_2d[id]['lidar'].shape[0])
    #                                 global_counter += 1
    #                                 dir_counter[dir_name]['2D'] += 1
    #             frame_counter += 1

    # # save everything to file
    # print("Directory counts:")
    # print(dir_counter)
    # info_dict = {
    #     "global_counter": global_counter,
    #     "counter_2d": counter_2d,
    #     "counter_3d": counter_3d,
    #     "counter_3d_2d": counter_3d_2d,
    #     "dir_counts": dir_counter
    # }

    # with open(base_path + 'info.json', 'w') as fp:
    #     json.dump(info_dict, fp)

    # for folder in ["2D/", "3D/", "3D_2D/"]:

    #     if folder == "2D/":
    #         data = labels_dict_2d
    #         box_points_list = np.array(box_points_list_2d)
    #         image_segment_relations = image_segment_relations_2d
    #     elif folder == "3D/":
    #         data = labels_dict_3d
    #         box_points_list = np.array(box_points_list_3d)
    #         image_segment_relations = image_segment_relations_3d
    #     else:
    #         data = labels_dict_3d_2d
    #         box_points_list = np.array(box_points_list_3d_2d)
    #         image_segment_relations = image_segment_relations_3d_2d

    #     with open(base_path + folder + '/labels.pkl', 'wb')as f:
    #         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     with open(base_path + folder + '/image_segment_relations.csv', 'w') as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',')
    #         filewriter.writerow(['image_id', 'frame', 'id', 'cam', 'segment'])

    #     with open(base_path + folder + '/image_segment_relations.csv', 'a') as csvfile:
    #         filewriter = csv.writer(csvfile, delimiter=',')
    #         for elm in image_segment_relations:
    #             filewriter.writerow(elm)

    #     np.save(base_path + folder + '/lidar_point_stats.npy', box_points_list)

    #     # checks
    #     if len(image_segment_relations) != len(data):
    #         print("Lengths of image_segment_relations and dict do not match. Please.")
    #         print(f"image_segment_relations: {len(image_segment_relations)} data points.")
    #         print(f"dict: {len(data)} data points.")
    #         print(f"Problem occurred for data in: {dir_name}.")

    #     if len(box_points_list) != len(data):
    #         print("Lengths of box_points_list and dict do not match. Please.")
    #         print(f"box_points_list: {len(box_points_list)} data points.")
    #         print(f"dict: {len(data)} data points.")
    #         print(f"Problem occurred for data in: {dir_name}.")
