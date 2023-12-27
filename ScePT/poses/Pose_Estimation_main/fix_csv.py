import pandas as pd

# Load the .csv file into a pandas DataFrame
df = pd.read_csv("/home/erik/ssd2/datasets/waymo_pose/2D/image_segment_relations.csv")

# Define a function to correct the image_id format
def correct_image_id(image_id):
    parts = image_id.split('_')
    if '.' in parts[1]:
        parts[1] = str(int(float(parts[1])))  # Convert x.0 to x
    return '_'.join(parts)

# Apply the correction function to the 'image_id' column
df['image_id'] = df['image_id'].apply(correct_image_id)

# Correct the 'cam' column values
df['cam'] = df['cam'].apply(lambda x: str(int(float(x))) if isinstance(x, str) and '.' in x else x)

# Save the modified DataFrame back to the .csv file
df.to_csv("/home/erik/ssd2/datasets/waymo_pose/2D/image_segment_relations.csv", index=False)
