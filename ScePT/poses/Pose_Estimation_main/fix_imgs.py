import os
import glob

# Specify your folder path here
folder_path = "/home/erik/ssd2/datasets/waymo_pose/2D/images"

# List all jpg files in the given folder
image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# Iterate through each file
for image_file in image_files:
    # Extract the filename without the path
    filename = os.path.basename(image_file)
    
    # Split the filename parts
    parts = filename.split('_')
    
    # Check if the middle part (ID) has a decimal and rename the file if it does
    if '.' in parts[1]:
        new_id = str(int(float(parts[1])))  # Convert x.0 to x
        new_filename = f"{parts[0]}_{new_id}_{parts[2]}"
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(image_file, new_file_path)
        print(f"Renamed {filename} to {new_filename}")

print("Done!")
