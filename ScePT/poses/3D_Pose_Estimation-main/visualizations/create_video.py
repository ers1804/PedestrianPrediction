import glob
import cv2
import sys
import os


def create_video(image_folder, folder_names, video_out, type="jpg"):

    for i in folder_names:
        tmp_video_name = str(i) + ".mp4"

        img_array = []
        images = sorted(glob.glob(image_folder + f"/{'_'.join(str(i).split('_')[1:])}/*.{type}"))
        for filename in images:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        store_path = '/'.join(image_folder.split("/")[0:-2]) + "/videos/"

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        out = cv2.VideoWriter(
            store_path + tmp_video_name, cv2.VideoWriter_fourcc(*"mp4v"), 10, size
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def main(image_folder, video_out=None):
    """Create video out of extracted waymo jpg files for all cameras

    Args:
        image_folder (str): Path to the root image folder (not cam folder!)
        video_out (_type_, optional): Path where video should be stored . Default is same folder as the images.
    """
    if video_out is None:
        video_out = image_folder

    # create camera videos
    folder_names = ["cam_FRONT", "cam_FRONT_LEFT", "cam_FRONT_RIGHT", "cam_SIDE_LEFT", "cam_SIDE_RIGHT"]
    cam_image_folder = image_folder + "images/"
    create_video(cam_image_folder, folder_names, video_out)

    # create lidar videos
    folder_names = ["lidar_BIRDS_EYE", "lidar_FOLLOWER"]
    lidar_image_folder = image_folder + "lidar/"
    create_video(lidar_image_folder, folder_names, video_out, type='png')


if __name__ == '__main__':

    if len(sys.argv) == 2:
        img_folder = sys.argv[1]
    else:
        print('Please specify path to camera root (not image folder like "FRONT" but rather its parent).')
    main(image_folder=img_folder)
