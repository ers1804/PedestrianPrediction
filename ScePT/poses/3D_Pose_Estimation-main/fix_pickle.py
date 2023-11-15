import pickle

# Load the dictionary from the .pkl file
with open("/home/erik/ssd2/datasets/waymo_pose/2D/labels.pkl", 'rb') as f:
    data = pickle.load(f)

# Create a new dictionary to store modified data
new_data = {}

for key, value in data.items():
    # Split the key based on underscore
    parts = key.split('_')
    
    # Check if the middle part (ID) has a decimal
    if '.' in parts[1]:
        # Extract the integer part and reconstruct the key
        new_key = f"{parts[0]}_{int(float(parts[1]))}_{parts[2]}"
        new_data[new_key] = value
    else:
        # If no modification needed, keep the original key
        new_data[key] = value

# Save the updated dictionary back to the .pkl file
with open("/home/erik/ssd2/datasets/waymo_pose/2D/labels.pkl", 'wb') as f:
    pickle.dump(new_data, f)
