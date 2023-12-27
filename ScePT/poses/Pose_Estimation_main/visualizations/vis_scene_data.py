# from asyncio.format_helpers import extract_stack
import os
import tensorflow as tf
# import glob
import io
import PIL.Image
import logging
import sys
import numpy as np
import pandas as pd
import pickle
import collections

import numpy as np
import open3d as o3d

# from tqdm import tqdm
from pyntcloud import PyntCloud
from google.protobuf.json_format import MessageToDict

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
# from waymo_open_dataset.metrics.python import keypoint_metrics
# from waymo_open_dataset.protos import keypoint_pb2
# from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import keypoint_data
# from waymo_open_dataset.utils import keypoint_draw
# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils

tf.compat.v1.enable_eager_execution()


def nested_dict():
    return collections.defaultdict(nested_dict)


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


def get_string_camera_name(name):
    """
     Args:
        name (int): Integer name of the camera in the waymo setup

    Returns:
        str: String name of the camera in the waymo setup
    """

    if name == dataset_pb2.CameraName.Name.FRONT:
        return "FRONT"
    elif name == dataset_pb2.CameraName.Name.FRONT_RIGHT:
        return "FRONT_RIGHT"
    elif name == dataset_pb2.CameraName.Name.FRONT_LEFT:
        return "FRONT_LEFT"
    elif name == dataset_pb2.CameraName.Name.SIDE_LEFT:
        return "SIDE_LEFT"
    elif name == dataset_pb2.CameraName.Name.SIDE_RIGHT:
        return "SIDE_RIGHT"
    else:
        logging.error('No matching Camera found. Problem camera has index name: {name}\nExiting now...')
        sys.exit(1)


def store_lidar_data(frame, tmp_path, counter):
    """
    Store lidar data from current frame to disk (TODO format).

        Args:
            frame (waymo_open_dataset.dataset_pb2.Frame): Frame that stores data
            tmp_path (str): Base path for folder creation
            counter (int): count of the current frame

    """

    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points_and_range_image, cp_points = convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
    point_cloud = np.concatenate(points_and_range_image, axis=0)

    dataframe = pd.DataFrame(data=point_cloud,
                             columns=["x", "y", "z",                                    # nlz = "no label zone"
                                      "range", "intensity", "elongation", "is_in_nlz"])  # (1 = in, -1 = not in)
    cloud = PyntCloud(dataframe)
    cloud.to_file(tmp_path + "/lidar/" + f"/{str(counter).zfill(5)}" ".ply")


def store_labels(labels, tmp_path, counter):
    """
    Store labels from current frame to disk.

    Args:
        labels (labels): Dictionary of labeld objects in the scene
        tmp_path (str): Base path for folder creation
        counter (int): count of the current frame

     Returns:
            int, int: Counters for 2D/3D keypoints observed in that frame
    """
    labels_dict = nested_dict()
    keypoint_2d_counter = 0
    keypoint_3d_counter = 0
    keypoint_3d_only = 0

    # extract pedestrains only | load all information into one file per frame
    for label in labels:
        cam_keypoints_available = False
        if labels[label].object_type == label_pb2.Label.TYPE_PEDESTRIAN:
            # check 2d keypoints in cameras
            if labels[label].camera:
                for cam in labels[label].camera:
                    if labels[label].camera[cam].keypoints.keypoint:
                        cam_keypoints_available = True
                        # store keypoints and bb
                        for joint in labels[label].camera[cam].keypoints.keypoint:
                            joint = MessageToDict(joint)
                            key = joint.pop('type')
                            labels_dict[label]['camera'][cam]['keypoints'][key] = joint
                        labels_dict[label]['camera'][cam]['bb'] = MessageToDict(labels[label].camera[cam].box)
                        keypoint_2d_counter += 1

            # save 3d keypoint labels
            if labels[label].laser.keypoints.keypoint:
                # check if 3d keypoints are availabale although 2d are not.
                # If this is frequently the case a redesign is needed...
                if not cam_keypoints_available:
                    keypoint_3d_only += 1
                    logging.error("3D keypoints are available without 2d image reference")
                    logging.error(f"tmp_path: {tmp_path}\nframe: {counter}\n ID: {label}")

                for joint in labels[label].laser.keypoints.keypoint:
                    joint = MessageToDict(joint)
                    key = joint.pop('type')
                    labels_dict[label]['lidar']['keypoints'][key] = joint
                labels_dict[label]['lidar']['bb'] = MessageToDict(labels[label].laser.box)
                keypoint_3d_counter += 1

    # dumb labels to file
    if len(labels_dict) > 0:
        with open(tmp_path + "/labels/" + f"/{str(counter).zfill(5)}.pkl", 'wb')as f:
            pickle.dump(labels_dict, f)

    return keypoint_2d_counter, keypoint_3d_counter, keypoint_3d_only


def store_camera_data(frame, tmp_path, counter):
    """
    Store camera data from current frame to disk (jpg format).

    Args:
        frame (waymo_open_dataset.dataset_pb2.Frame): Frame that stores data
        tmp_path (str): Base path for folder creation
        counter (int): count of the current frame
    """

    camera_image_by_name = {i.name: i.image for i in frame.images}

    # create data directorys
    if counter == 0:
        for camera_name in camera_image_by_name:
            if not os.path.exists(tmp_path + "/images/" + get_string_camera_name(camera_name)):
                os.makedirs(tmp_path + "/images/" + get_string_camera_name(camera_name))

    # store image data to disk
    for camera_name in camera_image_by_name:
        # decode image
        with io.BytesIO(camera_image_by_name[camera_name]) as fd:
            pil = PIL.Image.open(fd)
            pil.save(tmp_path + "/images/" + get_string_camera_name(camera_name) + f"/{str(counter).zfill(5)}" ".jpg")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Please specify segment to extract data from .tfrecord file (first argument) and location to store data to (second argument)')
        sys.exit(0)
    else:
        TFR_PATH = sys.argv[1]
        STORE_BASE_PATH = sys.argv[2]

    keypoint_2d_counter = 0
    keypoint_3d_counter = 0
    keypoint_3d_2d_counter = 0
    keypoint_3d_only = 0

    if not os.path.exists(STORE_BASE_PATH):
        os.makedirs(STORE_BASE_PATH)

    frame_counter = 0
    tfr_data = tf.data.TFRecordDataset(TFR_PATH, compression_type='')

    if not os.path.exists(STORE_BASE_PATH + "/images"):
        os.makedirs(STORE_BASE_PATH + "/images")

    if not os.path.exists(STORE_BASE_PATH + "/lidar"):
        os.makedirs(STORE_BASE_PATH + "/lidar")

    if not os.path.exists(STORE_BASE_PATH + "/labels"):
        os.makedirs(STORE_BASE_PATH + "/labels")

    for data in tfr_data:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        labels = keypoint_data.group_object_labels(frame)

        store_camera_data(frame, STORE_BASE_PATH, frame_counter)

        # store lidar data from frame
        store_lidar_data(frame, STORE_BASE_PATH, frame_counter)

        # store labels
        keypoint_2d_counter, keypoint_3d_counter, keypoint_3d_only = store_labels(labels, STORE_BASE_PATH, frame_counter)

        # store other data
        # TODO Maybe put a complete sequence into 1 dict and safe it as a large json, pckl or csv

        frame_counter += 1
        print(f"Processing frame {frame_counter}...")

    print('Creating Lidar views...')

    # create folders
    if not os.path.exists(STORE_BASE_PATH + "/lidar/BIRDS_EYE/"):
        os.makedirs(STORE_BASE_PATH + "/lidar/BIRDS_EYE/")
        
    if not os.path.exists(STORE_BASE_PATH + "/lidar/FOLLOWER/"):
        os.makedirs(STORE_BASE_PATH + "/lidar/FOLLOWER/")

    files = [os.path.join(STORE_BASE_PATH + "/lidar", f) for f in os.listdir(STORE_BASE_PATH + "/lidar") if (os.path.isfile(os.path.join(STORE_BASE_PATH + "/lidar", f)) and ".ply" in f)]

    # Initialise the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()

    once = True

    files_iter = enumerate(iter(sorted(files)))

    view = vis.get_view_control()

    for frame_counter, frame in files_iter:

        # get point cloud

        pcd.points = o3d.io.read_point_cloud(frame).points

        vis.add_geometry(pcd)

        # set camera position -> parameters can be obtained
        # by executing read_pointcloud.py and getting viewpoint by pressing STR+c
        # FOLLOWER
        view.set_front(np.array([-1, 0, 0.5]))
        view.set_up(np.array([0, 0, 1]))
        view.set_zoom(0.025)
        view.set_lookat(np.array([1, 0, 2]))

        vis.capture_screen_image(f"{STORE_BASE_PATH}/lidar/FOLLOWER/{str(frame_counter).zfill(5)}.png", do_render=True)

        # BIRDS EYE
        view.set_front(np.array([0, 0 , 1]))
        view.set_up(np.array([0, 1, 0]))
        view.set_zoom(0.120)
        view.set_lookat(np.array([0., -1, 1]))

        vis.capture_screen_image(f"{STORE_BASE_PATH}/lidar/BIRDS_EYE/{str(frame_counter).zfill(5)}.png", do_render=True)
    vis.destroy_window()

    print(
        f"Finished! In total stored:\n2D-Keypoints: {keypoint_2d_counter}\n3D-Keypoints: {keypoint_3d_counter} \n3D-Keypoints whithout 2D on camera: {keypoint_3d_only}\nPointclouds: {frame_counter}\nImages: {frame_counter*5}")
