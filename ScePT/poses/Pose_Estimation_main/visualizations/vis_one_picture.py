# path to data
import matplotlib.pyplot as plt
import PIL
import numpy as np
import io
import pickle
import pandas as pd
path = "/media/petbau/data/waymo/v0.7/3D_2D/"


image_segment_relations = pd.read_csv(path + "image_segment_relations.csv")


# load the labels file
with open(path + "labels.pkl", 'rb') as pickle_file:
    labels = pickle.load(pickle_file)

cams = image_segment_relations['cam']
frames = image_segment_relations['frame']
ids = image_segment_relations['id']
segments = image_segment_relations['segment']
unique_segments = set(segments)

l = []
num_labled_peds_in_cam = 3


for segment in unique_segments:
    seg_frames = image_segment_relations.loc[image_segment_relations['segment'] == segment]

    # get list of frames in the segment that are labeled with at least one pedestrain
    frames = sorted(seg_frames['frame'])
    frame_counts = {item: frames.count(item) for item in frames}
    # remove all appearances that only occur once
    multiple_counts_frame = [k for k, v in frame_counts.items() if v >= num_labled_peds_in_cam]

    for frame_count in multiple_counts_frame:
        fame_cams = seg_frames.loc[seg_frames['frame'] == frame_count]
        cams = sorted(fame_cams['cam'])
        cam_counts = {item: cams.count(item) for item in cams}
        multiple_counts_cam = [k for k, v in cam_counts.items() if v >= num_labled_peds_in_cam]
        if len(multiple_counts_cam) > 0:
            for cam_count in multiple_counts_cam:
                out = seg_frames.loc[(seg_frames['frame'] == frame_count) & (seg_frames['cam'] == cam_count)]
                l.append(out)


print(f"Found {len(l)} frames with more than {num_labled_peds_in_cam} people annotated in it...")


def _imdecode(buf: bytes) -> np.ndarray:
    with io.BytesIO(buf) as fd:
        pil = PIL.Image.open(fd)
        return np.array(pil)


def _imshow(ax: plt.Axes, image_np: np.ndarray):
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_autoscale_on(False)


for idx, df in enumerate(l):

    frame_number = df['frame'].iloc[0]
    cam = df['cam'].iloc[0]
    tfr_path = df['segment'].iloc[0]

    import tensorflow as tf
    import cv2
    from waymo_open_dataset.utils import keypoint_draw
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import keypoint_data

    objs_id_list = list(df['id'])

    dataset = tf.data.TFRecordDataset(tfr_path, compression_type='')
    frame_counter = 0
    for data in dataset:
        if frame_counter == int(frame_number):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
        frame_counter += 1

    tfr_labels = keypoint_data.group_object_labels(frame)
    obj_labels_list = []

    for obj_id in objs_id_list:
        obj_labels = tfr_labels[obj_id]
        obj_labels_list.append(obj_labels)

    camera_image_by_name = {i.name: i.image for i in frame.images}

    # display the images and save them to disk
    image_np = _imdecode(camera_image_by_name[cam])
    im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    if idx == 70:
        print('thats my frame')
    cv2.imwrite(f"/lhome/petbau/master_thesis/images_from_visualization/blank_{str(idx).zfill(4)}vis_one_picture.jpg", im_rgb)

    # draw 2d keypoints on the image and save them to disk

    keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
    _, ax = plt.subplots(frameon=False, figsize=(25, 25))
    _imshow(ax, image_np)

    for obj_labels in obj_labels_list:
        camera_wireframe = keypoint_draw.build_camera_wireframe(
            obj_labels.camera[cam].keypoints.keypoint)
        keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)
    
    plt.title(f'frame_number: {frame_number}; cam: {cam};\n tfr_path: {tfr_path}')

    plt.savefig(f"/lhome/petbau/master_thesis/images_from_visualization/annotated_{str(idx).zfill(4)}vis_one_picture.jpg")
    plt.close()
# create 3d GT plots | 3D PRED plots | 3D PRED & GT plots
