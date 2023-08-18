JOINT_KEYS = {1: 0, 5: 1,
              6: 2, 7: 3, 8: 4,
              9: 5, 10: 6, 13: 7,
              14: 8, 15: 9, 16: 10,
              17: 11, 18: 12, 20: 13,
              }


JOINT_NAMES = ["NOSE", "LEFT_SHOULDER", "LEFT_ELBOW",
               "LEFT_WRIST", "LEFT_HIP", "LEFT_KNEE",
               "LEFT_ANKLE", "RIGHT_SHOULDER", "RIGHT_ELBOW",
               "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
JOINt_COLORS_DICT = {'NOSE': '#00FF00', 'LEFT_SHOULDER': '#FFA6FE', 'LEFT_ELBOW': '#FFE502', 'LEFT_WRIST': '#006401', 'LEFT_HIP': '#010067', 'LEFT_KNEE': '#95003A',
                     'LEFT_ANKLE': '#007DB5', 'RIGHT_SHOULDER': '#774D00', 'RIGHT_ELBOW': '#90FB92', 'RIGHT_WRIST': '#0076FF', 'RIGHT_HIP': '#D5FF00',
                     'RIGHT_KNEE': '#A75740', 'RIGHT_ANKLE': '#6A826C'}
# "FOREHEAD/HEAD_CENTER"]

# tuples of (frame_number, cam, tfr_path)
IMAGES_TO_VIS = [(66, 3, "/media/petbau/data/waymo/v1.3.2/individual_files/training/segment-9529958888589376527_640_000_660_000_with_camera_labels.tfrecord")]


# html templates
VIS_COMPLETE_IMG_HTML_STATIC = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {
  box-sizing: border-box;
}

img {
    max-width: 100%;
    max-height: 100%;
}

.row::after {
  content: "";
  clear: both;
  display: table;
}

[class*="col-"] {
  float: left;
  padding: 15px;
}

.col-1 {width: 8.33%;}
.col-2 {width: 16.66%;}
.col-3 {width: 25%;}
.col-4 {width: 33.33%;}
.col-5 {width: 41.66%;}
.col-6 {width: 50%;}
.col-7 {width: 58.33%;}
.col-8 {width: 66.66%;}
.col-9 {width: 75%;}
.col-10 {width: 83.33%;}
.col-11 {width: 91.66%;}
.col-12 {width: 100%;}

html {
  font-family: "Lucida Sans", sans-serif;
}

.header {
  background-color: #9933cc;
  color: #ffffff;
  padding: 15px;
}

.boxblue li {
  padding: 8px;
  margin-bottom: 7px;
  background-color: #0002EF;
  color: #ffffff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}

.boxred li {
  padding: 8px;
  margin-bottom: 7px;
  background-color: #EE0911;
  color: #ffffff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}

.boxgreen li {
  padding: 8px;
  margin-bottom: 7px;
  background-color: #008000;
  color: #ffffff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}
</style>
</head>
<body>


<div class="row">
  <div class="col-5 menu">
    <ul>
    <center>
    <h1>3D Predictions</h1>
"""
