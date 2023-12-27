import os
import glob
import tensorflow as tf

path = os.path.expanduser("/media/petbau/My Book1/waymo/v1.3/")
files = list(glob.glob(f'{path}*/*/*.tfrecord'))

global_frames = 0
num_files = 0

print('This is the output of count_frames.py...')
for file in files:
    tfr_data = tf.data.TFRecordDataset(file, compression_type='')
    frame = 0
    for data in tfr_data:
        frame = frame + 1
    print(f"{file} contains {frame} frames...")
    global_frames += frame
    num_files += 1
print(f"In total counted {num_files} files")
print(f"In total counted {global_frames} frames.")