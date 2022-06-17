import json
import sys
from os.path import join

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from detectors import RCNNDetector
from utils import add_box, parse_arguments

# load configuration file
with open(sys.argv[1], "r") as f:
    config = json.load(f)

# parse arguments & assign default values
config = parse_arguments(config)

# create model
detector = RCNNDetector(model_path="data/faster_rcnn_inception_v2_coco.pb",
                        confidence_threshold=config["thresholds"]["confidence"])

# load image
frame = cv2.imread(config["capture_path"])

# resize image
if len(config["resolution"]) > 0:
    frame = cv2.resize(frame, config["resolution"])

# using a greyscale picture for faster detection
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
frame_rotated = np.rot90(frame_gray)  # rotate image by 90 degrees

# detect humans in standard frame
results = detector.detect(frame)
boxes = [box for (box, _) in results]

# group boxes that overlap
boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
boxes = non_max_suppression(boxes, overlapThresh=config["thresholds"]["overlap"])
boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]

# add boxes to frame
for box in boxes:
    frame = add_box(frame=frame, box=box, color=config["box_color"], thickness=2)

# display image
if config["logging"]["display"]:
    cv2.imshow('image', frame)
    cv2.waitKey(0)

# export image
cv2.imwrite(filename=join(config["output_folder"], "output.jpg"), img=frame)

# close model (only if RCNN detector is used)
if isinstance(detector, RCNNDetector):
    detector.close()

# close window
cv2.destroyAllWindows()
