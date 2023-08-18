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


from tqdm import tqdm
from typing import Collection

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.utils import keypoint_draw

from configs.constants import JOINT_KEYS

tf.compat.v1.enable_eager_execution()


tfr_path = sys.argv[1]
base_path = sys.argv[2]


def crop_camera_keypoints(
    image: np.ndarray,
    keypoints: Collection[keypoint_pb2.CameraKeypoint],
    box: label_pb2.Label.Box,
    margin: float = 0,
):
    """Crops camera image to the specified bounding box and shifts keypoints.

    Args:
      image: input image to crop, an array with shape [height, width, 3].
      keypoints: a collection of camera keypoints.
      box: a 2D bounding box to extract from the input image.
      margin: a ratio of the extra margin to add to the image relative to the
        input image size.

    Returns:
      tuple (new image, shifted keypoints).
    """
    # Adapted from:
    # https://github.com/waymo-research/waymo-open-dataset/blob/520c34a2090460d7498078fa9033e011e28e86b0/waymo_open_dataset/utils/frame_utils.py
    new_camera_keypoints = []
    crop_width = (1 + margin) * box.length
    crop_height = (1 + margin) * box.width
    min_x = max(0, int(box.center_x - crop_width / 2))
    min_y = max(0, int(box.center_y - crop_height / 2))
    for old_kp in keypoints:
        new_kp = keypoint_pb2.CameraKeypoint()
        new_kp.CopyFrom(old_kp)
        new_p = new_kp.keypoint_2d.location_px
        new_p.x -= min_x
        new_p.y -= min_y
        new_camera_keypoints.append(new_kp)
    max_x = min(image.shape[1] - 1, int(box.center_x + crop_width / 2))
    max_y = min(image.shape[0] - 1, int(box.center_y + crop_height / 2))
    new_image = image[min_y:max_y, min_x:max_x, :]
    return new_image, new_camera_keypoints


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


def get_croped_cam_data(frame, cam, label):
    """_summary_

    Args:
        frame (open dataset frame): Frame of the tfrecord file.
        labels (waymo object labels): Labels of the pedestrian.
        cam (int): Camera in which the data can be found.

    Returns:
        tuple: Cropped image and new keypoint coordinates.
    """

    # get camera images by name
    camera_image_by_name = {i.name: i.image for i in frame.images}

    img = _imdecode(camera_image_by_name[cam])

    cropped_image, cropped_camera_keypoints = keypoint_draw.crop_camera_keypoints(
        img,
        label.camera[cam].keypoints.keypoint,
        label.camera[cam].box,
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


def store_other_information(frame, labels_dict, id, labels, cam):

    # store 3D bb info
    labels_dict[id]['bb_3d'] = {
        "center_x": labels[label].laser.box.center_x,
        "center_y": labels[label].laser.box.center_y,
        "center_z": labels[label].laser.box.center_z,
        "width": labels[label].laser.box.width,
        "length": labels[label].laser.box.length,
        "height": labels[label].laser.box.height,
        "heading": labels[label].laser.box.heading
    }
    if cam != -1:
        labels_dict[id]['bb_2d'] = {
            "center_x": labels[label].camera[cam].box.center_x,
            "center_y": labels[label].camera[cam].box.center_y,
            "width": labels[label].camera[cam].box.length,
            "height": labels[label].camera[cam].box.width,
        }

    if 'keypoints_2d' in labels_dict[id]:
        # get all indices that are not labeled in the keypoints_2d_arr -> makes it easier to set them to zero after normalization
        labels_dict[id]['mask_2d'] = [JOINT_KEYS[k] for k in JOINT_KEYS.keys() if k not in labels_dict[id]['keypoints_2d'].keys()]

    if 'keypoints_3d' in labels_dict[id]:
        labels_dict[id]['mask_3d'] = [JOINT_KEYS[k] for k in JOINT_KEYS.keys() if k not in labels_dict[id]['keypoints_3d'].keys()]

    # store camera intrinsic, extrinsic and metadata
    if cam != -1:
        image = next(im for im in frame.images if im.name == cam)
        cam_calib = next(c for c in frame.context.camera_calibrations if c.name == cam)
        labels_dict[id]['extrinsic'] = np.array([cam_calib.extrinsic.transform]).reshape(4, 4)
        labels_dict[id]['intrinsic'] = np.array([cam_calib.intrinsic])
        labels_dict[id]['metadata'] = np.array([cam_calib.width, cam_calib.height, cam_calib.rolling_shutter_direction], dtype=np.int32)

        camera_image_metadata = list(image.pose.transform)
        camera_image_metadata.append(image.velocity.v_x)
        camera_image_metadata.append(image.velocity.v_y)
        camera_image_metadata.append(image.velocity.v_z)
        camera_image_metadata.append(image.velocity.w_x)
        camera_image_metadata.append(image.velocity.w_y)
        camera_image_metadata.append(image.velocity.w_z)
        camera_image_metadata.append(image.pose_timestamp)
        camera_image_metadata.append(image.shutter)
        camera_image_metadata.append(image.camera_trigger_time)
        camera_image_metadata.append(image.camera_readout_done_time)

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
                os.makedirs(base_path + folder + "/images/")
    dir_counter = {"training/": {"2D": 0, '3D': 0, "3D_2D": 0}, "validation/": {"2D": 0, '3D': 0, "3D_2D": 0}, "testing/": {"2D": 0, '3D': 0, "3D_2D": 0}, "domain_adaptation/validation/": {"2D": 0, '3D': 0, "3D_2D": 0},
                   "domain_adaptation/training/": {"2D": 0, '3D': 0, "3D_2D": 0}, "testing_3d_camera_only_detection": {"2D": 0, '3D': 0, "3D_2D": 0}, "domain_adaptation/testing/": {"2D": 0, '3D': 0, "3D_2D": 0}}
    # Original dir_names
    # dir_names = ["training/", "validation/", "testing/", "domain_adaptation/validation/", "domain_adaptation/training/", "testing_3d_camera_only_detection", "domain_adaptation/testing/"]
    dir_names = ["training/", "validation/", "testing/", "testing_interactive/", "training_20s/", "validation_interactive/"]
    # dir_names = ["training/", "validation/"]
    # dir_names = ["validation/"]
    

    for dir_name in tqdm(dir_names):

        # get all TFRecord files
        print(f"Extracting {dir_name[0:-1]} data...")
        files = list(glob.glob(f'{tfr_path}' + dir_name + '*.tfrecord'))

        for frame_path in tqdm(files):

            frame_counter = 0
            tfr_data = tf.data.TFRecordDataset(frame_path, compression_type='')

            for data in tfr_data:
                frame = dataset_pb2.Frame()
                pc = False  # prevent from reading same point cloud multiple times
                frame.ParseFromString(bytearray(data.numpy()))
                labels = keypoint_data.group_object_labels(frame)

                # extract pedestrians only | load all information into one file per frame
                for label in labels:

                    if (labels[label].object_type == label_pb2.Label.TYPE_PEDESTRIAN) or (labels[label].object_type == label_pb2.Label.TYPE_CYCLIST):
                        # 3d and 2d label available
                        if labels[label].laser.keypoints.keypoint and one_cam_has_keypoints(labels[label]):

                            # iterate over cameras --> Note that this produces same 3D points, but different 2D annotations
                            for cam in labels[label].camera:
                                # prevent from storing cam data that does not have keypoints
                                if labels[label].camera[cam].keypoints.keypoint:

                                    id = str(frame_counter) + "_" + str(cam) + "_" + label

                                    labels_dict_3d_2d[id] = {}
                                    counter_3d_2d += 1

                                    cropped_image, cropped_camera_keypoints = get_croped_cam_data(frame, cam, labels[label])
                                    img_height, img_width, _ = cropped_image.shape

                                    # save image data to file
                                    cv2.imwrite(base_path + "3D_2D/images/" + id + ".jpg",  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                                    labels_dict_3d_2d[id]['keypoints_2d'] = {}
                                    for joint in cropped_camera_keypoints:
                                        labels_dict_3d_2d[id]['keypoints_2d'][joint.type] = {'x': joint.keypoint_2d.location_px.x,
                                                                                             'y': joint.keypoint_2d.location_px.y,
                                                                                             'occluded': joint.keypoint_2d.visibility.is_occluded,
                                                                                             }
                                    labels_dict_3d_2d[id]['img_2d_height'] = img_height
                                    labels_dict_3d_2d[id]['img_2d_width'] = img_width

                                    # get full image keypoints as well (later maybe usefull)
                                    labels_dict_3d_2d[id]['keypoints_2d_image'] = {}
                                    for joint in labels[label].camera[cam].keypoints.keypoint:
                                        labels_dict_3d_2d[id]['keypoints_2d_image'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
                                                                                                   'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
                                                                                                   'occluded': joint.keypoint_2d.visibility.is_occluded}

                                    # store 3d keypoints
                                    labels_dict_3d_2d[id]['keypoints_3d'] = {}
                                    for joint in labels[label].laser.keypoints.keypoint:
                                        labels_dict_3d_2d[id]['keypoints_3d'][joint.type] = {'x': joint.keypoint_3d.location_m.x,
                                                                                             'y': joint.keypoint_3d.location_m.y,
                                                                                             'z': joint.keypoint_3d.location_m.z,
                                                                                             'occluded': joint.keypoint_3d.visibility.is_occluded}

                                    labels_dict_3d_2d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d_2d[id], three_dim=True)
                                    labels_dict_3d_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_3d_2d[id])
                                    image_segment_relations_3d_2d.append([id, frame_counter, label, cam,  frame_path])
                                    store_other_information(frame, labels_dict_3d_2d, id, labels, cam)
                                    # load pc and store data
                                    if not pc:
                                        points_all, cp_points = load_point_cloud(frame)
                                        pc = True
                                    store_lidar_and_projections(labels, label, labels_dict_3d_2d, cam)
                                    box_points_list_3d_2d.append(labels_dict_3d_2d[id]['lidar'].shape[0])
                                    global_counter += 1
                                    dir_counter[dir_name]['3D_2D'] += 1

                        # only 3d label available
                        elif labels[label].laser.keypoints.keypoint and (not one_cam_has_keypoints(labels[label])):

                            if not labels[label].camera:
                                cam = -1
                            else:
                                # just take the first one as referece
                                cam = list(labels[label].camera.keys())[0]

                            id = str(frame_counter) + "_" + str(cam) + "_" + label

                            labels_dict_3d[id] = {}
                            counter_3d += 1

                            # store 2d keypoints
                            labels_dict_3d[id]['keypoints_3d'] = {}
                            for joint in labels[label].laser.keypoints.keypoint:
                                labels_dict_3d[id]['keypoints_3d'][joint.type] = {'x': joint.keypoint_3d.location_m.x,
                                                                                  'y': joint.keypoint_3d.location_m.y,
                                                                                  'z': joint.keypoint_3d.location_m.z,
                                                                                  'occluded': joint.keypoint_3d.visibility.is_occluded}

                            labels_dict_3d[id]['keypoints_3d_arr'] = create_array(labels_dict_3d[id], three_dim=True)
                            image_segment_relations_3d.append([id, frame_counter, label, cam,  frame_path])
                            store_other_information(frame, labels_dict_3d, id, labels, cam)

                            # load pc and store data
                            if not pc:
                                points_all, cp_points = load_point_cloud(frame)
                                pc = True
                            box = box_utils.box_to_tensor(labels[label].laser.box)[tf.newaxis, :]
                            box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
                            labels_dict_3d[id]['lidar'] = box_points.astype('float32')
                            box_points_list_3d.append(box_points.shape[0])
                            global_counter += 1
                            dir_counter[dir_name]['3D'] += 1

                        # check 2d keypoints in cameras
                        elif labels[label].camera:
                            for cam in labels[label].camera:
                                if labels[label].camera[cam].keypoints.keypoint:

                                    id = str(frame_counter) + "_" + str(cam) + "_" + label

                                    # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                                    # length = labels[label].camera[cam].box.length
                                    # width = labels[label].camera[cam].box.width

                                    # if width < length:
                                    #     labels[label].camera[cam].box.width = length
                                    # else:
                                    #     labels[label].camera[cam].box.length = width

                                    cropped_image, cropped_camera_keypoints = get_croped_cam_data(frame, cam, labels[label])
                                    img_height, img_width, _ = cropped_image.shape

                                    # cropped_image, cropped_camera_keypoints = crop_camera_keypoints(
                                    #     img,
                                    #     labels[label].camera[cam].keypoints.keypoint,
                                    #     labels[label].camera[cam].box,
                                    #     margin=0)

                                    # check if croping worked properly, otherwise add padding to it
                                    # ratio = cropped_image.shape[0]/cropped_image.shape[1]
                                    # bottom = 0
                                    # left = 0
                                    # if not (0.97 < ratio < 1.03):
                                    #     if ratio > 1:
                                    #         left = cropped_image.shape[0] - cropped_image.shape[1]
                                    #     else:
                                    #         bottom = cropped_image.shape[1] - cropped_image.shape[0]
                                    #     cropped_image = cv2.copyMakeBorder(cropped_image, 0, bottom, left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                                    # # resize image to output format
                                    # res_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

                                    labels_dict_2d[id] = {}
                                    counter_2d += 1
                                    labels_dict_2d[id]['keypoints_2d'] = {}
                                    for joint in cropped_camera_keypoints:
                                        labels_dict_2d[id]['keypoints_2d'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
                                                                                          'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
                                                                                          'occluded': joint.keypoint_2d.visibility.is_occluded}
                                    # get full image keypoints as well (later maybe usefull)
                                    labels_dict_2d[id]['keypoints_2d_image'] = {}
                                    for joint in labels[label].camera[cam].keypoints.keypoint:
                                        labels_dict_2d[id]['keypoints_2d_image'][joint.type] = {'x': joint.keypoint_2d.location_px.x,  # labels[label].camera[cam].box.length/256 + left,
                                                                                                'y': joint.keypoint_2d.location_px.y,  # * labels[label].camera[cam].box.width/256 + bottom,
                                                                                                'occluded': joint.keypoint_2d.visibility.is_occluded}
                                    labels_dict_2d[id]['img_2d_height'] = img_height
                                    labels_dict_2d[id]['img_2d_width'] = img_width

                                    labels_dict_2d[id]['keypoints_2d_arr'] = create_array(labels_dict_2d[id])
                                    image_segment_relations_2d.append([id, frame_counter, label, cam, frame_path])
                                    store_other_information(frame, labels_dict_2d, id, labels, cam)

                                    # save image data to file
                                    cv2.imwrite(base_path + "2D/images/" + id + ".jpg",  cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                                    # load point cloud data
                                    if not pc:
                                        points_all, cp_points = load_point_cloud(frame)
                                        pc = True
                                    store_lidar_and_projections(labels, label, labels_dict_2d, cam)
                                    box_points_list_2d.append(labels_dict_2d[id]['lidar'].shape[0])
                                    global_counter += 1
                                    dir_counter[dir_name]['2D'] += 1
                frame_counter += 1

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

        with open(base_path + folder + '/labels.pkl', 'wb')as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(base_path + folder + '/image_segment_relations.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['image_id', 'frame', 'id', 'cam', 'segment'])

        with open(base_path + folder + '/image_segment_relations.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            for elm in image_segment_relations:
                filewriter.writerow(elm)

        np.save(base_path + folder + '/lidar_point_stats.npy', box_points_list)

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
