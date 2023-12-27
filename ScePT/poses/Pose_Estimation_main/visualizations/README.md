# Visualization scripts for the Waymo Open Dataset

This folder contains some scripts to visualize the data directly from the provided TFRecord files.

## Lidar and Camera output 
Create lidar point cloud (`.ply`) and camera images (`.jpg`) for each frame in the file. Note that the Lidar data is merged at each frame and the complete sensor output is stored together in one file.

To start the extration process run:

```python
# make sure the folder in the <store_path> exists before executing the command
python vis_scene_data.py <tfr_path> <store_path>
```

## Crate video from sensor data
Once the camera and lidar data is extracted, the videos for the 20s segemnt cna be created.

```python
# make sure vis_scene_data.py was extracted beforehand
python create_video.py <store_path>
```

## Folder structure
```
.
├── images
│   ├── FRONT
│   ├── FRONT_LEFT
│   ├── FRONT_RIGHT
│   ├── SIDE_LEFT
│   └── SIDE_RIGHT
├── labels
├── lidar
│   ├── BIRDS_EYE
│   └── FOLLOWER
└── videos
```
