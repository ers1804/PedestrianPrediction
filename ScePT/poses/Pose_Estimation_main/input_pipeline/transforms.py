import logging
import sys
import torch


# # def normalize_keypoints2D_batch_by_depth(keypoint_batch, mask, depth):
# #     # keypoint_batch[~mask] = torch.tensor(float('-inf'))
# #     # max_values = torch.max(keypoint_batch, dim=1).values
# #     keypoint_batch[~mask] = torch.tensor(float('inf'))
# #     min_values = torch.min(keypoint_batch, dim=1).values
# #     keypoint_batch[~mask] = torch.tensor(0, dtype=torch.float32)
# #     keypoint_batch_zz = keypoint_batch.transpose(0, 1) - min_values
# #     keypoint_batch_trans = keypoint_batch_zz.transpose(1, 2)
# #     norm_kp = keypoint_batch_trans/depth.to(torch.float32)
# #     return norm_kp.permute(2, 0, 1)


def normalize_keypoints2D_batch(keypoint_batch, mask, epsilon=1e-6):
    keypoint_batch[~mask] = torch.tensor(float('-inf'))
    max_values = torch.max(keypoint_batch, dim=1).values
    keypoint_batch[~mask] = torch.tensor(float('inf'))
    min_values = torch.min(keypoint_batch, dim=1).values
    keypoint_batch[~mask] = torch.tensor(0, dtype=torch.float32)

    height = max_values[:, 1] - min_values[:, 1]
    width = max_values[:, 0] - min_values[:, 0]

    keypoint_batch_zz = keypoint_batch.transpose(0, 1) - min_values

    keypoint_batch_trans = keypoint_batch_zz.transpose(1, 2)

    # norm height [-1,1]
    norm_kp = (keypoint_batch_trans/(height+epsilon))*2
    norm_kp[:, 0, :] = norm_kp[:, 0, :] - width/(height+epsilon)
    norm_kp[:, 1, :] = norm_kp[:, 1, :] - 1

    return norm_kp.permute(2, 0, 1)

    # norm width [-1,1]
    # norm_kp = (keypoint_batch_trans/(width+epsilon)*2
    # norm_kp[:, 0, :] = norm_kp[:, 0, :] - 1
    # norm_kp[:, 1, :] = norm_kp[:, 1, :] - height/(width+epsilon)

    # return norm_kp.permute(2, 0, 1)


class NormalizeKeypoints2D():
    """
    Normalize 3D keypoints such that width is between [-1,1] and aspect ratio stays the same
    """

    def __init__(self):
        pass

    def __call__(self, sample, height, width):

        # drop_rows = np.any(sample, axis=-1, )
        # keep_rows = np.invert(drop_rows)

        # kp_2d_norm = sample/width*2 - [1, height/width]

        # # mask unlabeled joints -> set them to zero
        # kp_2d_norm[keep_rows] = np.array([0, 0])

        if width == 0 or height == 0:
            logging.error('Division by zero in normalization detected. Exiting now')
            sys.exit(1)

        return sample/height*2 - [width/height, 1]
        # return sample/width*2 - [1, height/width]


class NormalizeKeypoints3D():
    """
    Normalize 3D keypoints such that width is between [-1,1] and aspect ratio stays the same
    """

    def __init__(self):
        pass

    def __call__(self, sample, length, width, height):

        return sample/length


class NormalizePointCloud():
    """
    Normalize point cloud input
    """

    def __init__(self):
        # TODO:
        pass
