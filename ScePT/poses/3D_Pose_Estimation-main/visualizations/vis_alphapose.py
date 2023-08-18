import logging
import pickle
import torch
import pandas as pd
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.protos import keypoint_pb2

from waymo_open_dataset.utils import keypoint_draw
JOINT_KEYS = {1: 0, 5: 1,
              6: 2, 7: 3, 8: 4,
              9: 5, 10: 6, 13: 7,
              14: 8, 15: 9, 16: 10,
              17: 11, 18: 12, 20: 13,
              }
REVERSE_JOINT_KEYS = {value: key for key, value in JOINT_KEYS.items()}


if __name__ == "__main__":
    labels = "/media/petbau/data/waymo/v0.10/alpha_pose/labels.pkl"
    ids = ["80_2_l1jmpmqm9n1c71YymDLv1Q", "131_1_KiLv-hRicxRgRM6NGLL8rw", "158_1_kO9Df1WTlSDyiH_PAibaGA", "159_1__77TG6zk8fi0Io4JRWXH8w"]
    image_segment_relations = pd.read_csv("/media/petbau/data/waymo/v0.10/alpha_pose/image_segment_relations.csv")

    logging.info(f'Loading labels from {labels}...')
    with open(labels, 'rb') as pickle_file:
        labels = pickle.load(pickle_file)
    for id in ids:
        margin = 0.3
        keypoints = labels[id]['keypoints_2d_arr_un'][:, :2].copy()
        occlusions_scores = labels[id]['keypoints_2d_arr_un'][:, -1]
        crop_width = (1 + margin) * labels[id]['bb_2d']['width']
        crop_height = (1 + margin) * labels[id]['bb_2d']['height']
        min_x = max(0, labels[id]['bb_2d']['center_x'] - crop_width/2)
        min_y = max(0, labels[id]['bb_2d']['center_y'] - crop_height/2)

        keypoints[:, 0] = keypoints[:, 0] - min_x
        keypoints[:, 1] = keypoints[:, 1] - min_y

        ap_cropped_keypoints = []
        counter = 0
        for keypoint, occlusion in zip(keypoints, occlusions_scores):
            cam_keypoint = keypoint_pb2.CameraKeypoint()
            cam_keypoint.type = REVERSE_JOINT_KEYS[counter]
            cam_keypoint.keypoint_2d.location_px.x = keypoint[0]
            cam_keypoint.keypoint_2d.location_px.y = keypoint[1]
            cam_keypoint.keypoint_2d.visibility.is_occluded = int(occlusion)
            ap_cropped_keypoints.append(cam_keypoint)
            counter += 1

        camera_wireframe_cropped = keypoint_draw.build_camera_wireframe(ap_cropped_keypoints)

        ap_keypoints = []
        counter = 0
        keypoints_wo_crop = labels[id]['keypoints_2d_arr_un'][:, :2].copy()
        for keypoint, occlusion in zip(keypoints_wo_crop, occlusions_scores):
            cam_keypoint = keypoint_pb2.CameraKeypoint()
            cam_keypoint.type = REVERSE_JOINT_KEYS[counter]
            cam_keypoint.keypoint_2d.location_px.x = keypoint[0]
            cam_keypoint.keypoint_2d.location_px.y = keypoint[1]
            cam_keypoint.keypoint_2d.visibility.is_occluded = int(occlusion)
            ap_keypoints.append(cam_keypoint)
            counter += 1

        camera_wireframe_complete = keypoint_draw.build_camera_wireframe(ap_keypoints)

        tfr_path = image_segment_relations.loc[image_segment_relations['image_id'] == id]['segment'].item()
        cam = int(image_segment_relations.loc[image_segment_relations['image_id'] == id]['cam'].item())
        frame_number, _,  obj_id = id.split("_", 2)
        print(f"Path: {tfr_path}")
        print(f"Cam: {cam}")
        print(f'Frame number: {frame_number}')
        print(f'Obj-ID:{obj_id}')

        dataset = tf.data.TFRecordDataset(tfr_path, compression_type='')
        frame_counter = 0
        for data in dataset:
            if frame_counter == int(frame_number):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                break
            frame_counter += 1

        tfr_labels = keypoint_data.group_object_labels(frame)
        obj_labels = tfr_labels[obj_id]

        # get image
        import io
        import PIL
        import numpy as np
        import matplotlib.pyplot as plt

        def _imdecode(buf: bytes):
            with io.BytesIO(buf) as fd:
                pil = PIL.Image.open(fd)
                return np.array(pil)

        def _imshow(ax: plt.Axes, image_np: np.ndarray):
            # image_np = cv2.copyMakeBorder(image_np, top=250, bottom=250, left=250, right=250, borderType=cv2.BORDER_CONSTANT)
            ax.imshow(image_np)
            ax.axis('off')
            ax.set_autoscale_on(False)

        camera_image_by_name = {i.name: i.image for i in frame.images}
        image_np = _imdecode(camera_image_by_name[cam])
        keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
        _, ax = plt.subplots(frameon=False, figsize=(25, 25))
        _imshow(ax, image_np)
        keypoint_draw.draw_camera_wireframe(ax, camera_wireframe_complete)
        plt.show()

        import cv2

        # load cropped image from other source for now
        cropped_img_path = "/media/petbau/data/waymo/v0.10/3D_2D/images/" + id
        img = cv2.imread(cropped_img_path + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, ax = plt.subplots(frameon=False, figsize=(25, 25))
        _imshow(ax, img)
        keypoint_draw.draw_camera_wireframe(ax, camera_wireframe_cropped)
        plt.show()
