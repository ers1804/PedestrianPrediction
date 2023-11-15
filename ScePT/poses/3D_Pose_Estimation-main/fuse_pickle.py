import pickle
import glob

file_pattern = "/home/erik/ssd2/datasets/waymo_pose/2D/*_labels.pkl"
file_list = glob.glob(file_pattern)

# Initialize an empty dictionary to store the merged results
merged_dict = {}

# Loop through each file, load its contents (as a dictionary), and merge it into the merged_dict
for file in file_list:
    with open(file, 'rb') as f:
        current_dict = pickle.load(f)
        merged_dict.update(current_dict)  # Merge the current dictionary into merged_dict

# Save the merged dictionary to a new .pkl file
with open("/home/erik/ssd2/datasets/waymo_pose/2D/labels.pkl", 'wb') as f:
    pickle.dump(merged_dict, f)
