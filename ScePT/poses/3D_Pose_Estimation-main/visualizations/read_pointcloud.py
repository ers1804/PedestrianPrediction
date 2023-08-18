import sys
import os
import open3d as o3d


import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")

if len(sys.argv) == 2:
    path = sys.argv[1]
else:
    # /media/petbau/My Book1/waymo/v0.0/validation/validation_0000/segment-10203656353524179475_7625_000_7645_000_with_camera_labels/lidar  ")
    print("No folder with '.ply' files provided as argument. Try to read from default folder at: /lhome/petbau/test/vis/lidar")
    path = "/lhome/petbau/test/vis/lidar"


files = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and ".ply" in f)]

print("There are %d frames in this file." % len(files))

# Initialise the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window()
pcd = o3d.geometry.PointCloud()

once = True

files_iter = enumerate(iter(sorted(files)))


def display_next_frame(event=None):
    global once
    try:
        frame_counter, frame = next(files_iter)
    except StopIteration:
        print('End of sequence. Exiting now!')
        sys.exit(0)

    # load point cloud data
    pcd.points = o3d.io.read_point_cloud(frame).points

    if once:
        vis.add_geometry(pcd)
        once = False
    else:
        vis.update_geometry(pcd)

    print(f'Currently at frame: {frame_counter}')


vis.register_key_callback(262, display_next_frame)  # Right arrow key

display_next_frame()

while True:
    vis.poll_events()
    vis.update_renderer()
    vis.run()
