import matplotlib.pyplot as plt
import torch
import numpy as np
import plotly.graph_objects as go 
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import keypoint_draw
from configs.constants import JOINT_KEYS


REVERSE_JOINT_KEYS = {value: key for key, value in JOINT_KEYS.items()}


def _imshow(ax: plt.Axes, image_np: np.ndarray):
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_autoscale_on(False)


def plot_keypoints_2D(keypoints_2D, mask_2D, name="test"):

    min_x, min_y = torch.min(keypoints_2D, dim=0).values
    max_x, max_y = torch.max(keypoints_2D, dim=0).values

    width = int((max_x - min_x).item()*10)
    height = int((max_y - min_y).item()*10)

    keypoints_2D[:, 0] = keypoints_2D[:, 0] - min_x
    keypoints_2D[:, 1] = keypoints_2D[:, 1] - min_y

    cropped_camera_keypoints = []
    for counter, keypoint in enumerate(keypoints_2D):
        if mask_2D[:, 0][counter]:
            cam_keypoint = keypoint_pb2.CameraKeypoint()
            cam_keypoint.type = REVERSE_JOINT_KEYS[counter]
            cam_keypoint.keypoint_2d.location_px.x = keypoint[0].cpu() * 10 + int(0.15*width)
            cam_keypoint.keypoint_2d.location_px.y = keypoint[1].cpu() * 10 + int(0.15*height)
            cam_keypoint.keypoint_2d.visibility.is_occluded = 0
            cropped_camera_keypoints.append(cam_keypoint)

    camera_wireframe = keypoint_draw.build_camera_wireframe(
        cropped_camera_keypoints)
    # print(cropped_camera_keypoints)

    keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
    _, ax = plt.subplots(frameon=False, figsize=(10, 10))
    white_image = np.ones((int(height*1.25), int(width*1.25), 3))
    _imshow(ax, white_image)
    keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)

    plt.savefig(name)
    # plt.show()


def get_wireframe(keypoints_2D, mask_2D):
    min_x, min_y = torch.min(keypoints_2D, dim=0).values
    max_x, max_y = torch.max(keypoints_2D, dim=0).values

    width = int((max_x - min_x).item()*100)
    height = int((max_y - min_y).item()*100)

    keypoints_2D[:, 0] = keypoints_2D[:, 0] - min_x
    keypoints_2D[:, 1] = keypoints_2D[:, 1] - min_y

    cropped_camera_keypoints = []
    for counter, keypoint in enumerate(keypoints_2D):
        if mask_2D[:, 0][counter]:
            cam_keypoint = keypoint_pb2.CameraKeypoint()
            cam_keypoint.type = REVERSE_JOINT_KEYS[counter]
            cam_keypoint.keypoint_2d.location_px.x = keypoint[0].cpu() * 100 + int(0.15*width)
            cam_keypoint.keypoint_2d.location_px.y = keypoint[1].cpu() * 100 + int(0.15*height)
            cam_keypoint.keypoint_2d.visibility.is_occluded = 0
            cropped_camera_keypoints.append(cam_keypoint)

    camera_wireframe = keypoint_draw.build_camera_wireframe(
        cropped_camera_keypoints)
    return camera_wireframe, width, height


def _create_plotly_figure() -> go.Figure:
    """Creates a plotly figure for 3D visualization."""
    fig = go.Figure()
    axis_settings = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        showaxeslabels=False,
        showticklabels=False)
    fig.update_layout(
        width=600,
        height=600,
        showlegend=False,
        scene=dict(
            aspectmode='data',  # force xyz has same scale,
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
        ),
    )
    return fig


def get_3D_plotly_plot(keypoints):
    """Show plotly plot of 3D keypoints

    Args:
        keypoints (torch.tensor): Torch tensor of shape [13x3]
    """

    keypoints_3D = []
    c = 0
    for keypoint in keypoints:
        laser_keypoint = keypoint_pb2.LaserKeypoint()
        laser_keypoint.type = REVERSE_JOINT_KEYS[c]
        laser_keypoint.keypoint_3d.location_m .x = keypoint[0]
        laser_keypoint.keypoint_3d.location_m .y = keypoint[1]
        laser_keypoint.keypoint_3d.location_m .z = keypoint[2]
        laser_keypoint.keypoint_3d.visibility.is_occluded = False
        keypoints_3D.append(laser_keypoint)
        c += 1

    laser_wireframe = keypoint_draw.build_laser_wireframe(keypoints_3D)
    fig = _create_plotly_figure()
    keypoint_draw.draw_laser_wireframe(fig, laser_wireframe)
    
    fig.show()
