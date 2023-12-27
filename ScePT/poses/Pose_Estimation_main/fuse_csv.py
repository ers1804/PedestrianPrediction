import pandas as pd
import glob

# Use glob to get a list of all .csv files matching the naming pattern
file_pattern = "/home/erik/ssd2/datasets/waymo_pose/2D/*_image_segment_relations.csv"
file_list = glob.glob(file_pattern)

# Create an empty list to store individual dataframes
dfs = []

# Loop through each file, read its contents, and append to the dfs list
for file in file_list:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes in the dfs list into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("/home/erik/ssd2/datasets/waymo_pose/2D/image_segment_relations.csv", index=False)