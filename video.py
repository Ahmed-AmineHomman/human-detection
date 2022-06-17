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
if len(config["resolution"]) == 0:
    raise Exception(f"you must provide a resolution for video detection")

# define model
detector = RCNNDetector(model_path="data/faster_rcnn_inception_v2_coco.pb",
                        confidence_threshold=config["thresholds"]["confidence"])

cv2.startWindowThread()

# open video stream
recorder = cv2.VideoCapture(config["capture_path"])

# initialize output object
writer = cv2.VideoWriter(join(config["output_folder"], "output.mp4"), 0x7634706d, 15., config["resolution"])

frame_count = 0
not_last_frame = True
while recorder.isOpened():
    # get next frame
    not_last_frame, frame = recorder.read()

    if not not_last_frame or ((config["frames"]["max"] > 0) and (frame_count > config["frames"]["max"])):
        break

    if frame_count >= config["frames"]["min"]:
        # resize frame to output resolution
        frame = cv2.resize(frame, config["resolution"])

        # detect humans in standard frame
        results = detector.detect(frame)
        boxes = [box for (box, _) in results]

        # group boxes that overlap
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        boxes = non_max_suppression(boxes, overlapThresh=config["thresholds"]["overlap"])
        boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]

        # display the detected boxes in the colour picture
        for box in boxes:
            frame = add_box(frame=frame, box=box, color=config["box_color"], thickness=2)

        writer.write(frame.astype('uint8'))  # insert resulting frame to output
        if config["logging"]["display"]:
            cv2.imshow('video', frame)

    # update frame count & eventually print it
    frame_count += 1
    if config["logging"]["console"] & (frame_count % 10 == 0):
        print(f"   . frame {frame_count}")

    # break loop if user presses key 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
recorder.release()

# and release the output
writer.release()

# close model (only if RCNN detector is used)
if isinstance(detector, RCNNDetector):
    detector.close()

# finally, close the window
key = cv2.waitKey(1)
cv2.destroyAllWindows()
