import numpy as np
import glob

file_pattern = "/home/erik/ssd2/datasets/waymo_pose/2D/*_lidar_point_stats.npy"
file_list = glob.glob(file_pattern)

# List to hold individual arrays from each .npy file
arrays = []

# Loop through each file, load its content (as a NumPy array), and append to the list
for file in file_list:
    arr = np.load(file)
    arrays.append(arr)

# Concatenate all arrays
combined_array = np.concatenate(arrays)

# Save the combined array to a new .npy file
np.save("/home/erik/ssd2/datasets/waymo_pose/2D/lidar_point_stats.npy", combined_array)
