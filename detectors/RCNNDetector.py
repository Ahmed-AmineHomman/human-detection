# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
# Code also adapted from:
# https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
import numpy as np
import tensorflow as tf

from .AbstractHumanDetector import AbstractHumanDetector


class RCNNDetector(AbstractHumanDetector):
    """
    Class using an RCNN for detecting people in an image.

    This class uses models from the TensorFlow v1 Detection Model Zoo available at:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
    The model graph (".pb" file) produced by TensorFlow should be specified at creation.

    The confidence level can also be specified at creation. Outputted boxes will only be those whose score is greater
    than the value of the specified confidence_level.

    This class has been tested with the "faster_rcnn_inception_v2_coo" model.
    This class might work with any other model of the above zoo but it has not been tested with those.
    """

    def __init__(self, model_path, confidence_threshold):
        self.model_path = model_path
        self.threshold = confidence_threshold

        self.detection_graph = tf.compat.v1.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.session = tf.compat.v1.Session(graph=self.detection_graph)

        # definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # each score represent how level of confidence for each of the objects.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # classes represents the class of the detected object (human correspond to class 1)
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def detect(self, frame):
        # expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        expanded_frame = np.expand_dims(frame, axis=0)

        # Actual detection.
        (all_boxes, all_scores, all_classes) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: expanded_frame})
        all_scores = all_scores[0].tolist()
        all_classes = all_classes[0].tolist()

        # get boxes representing humans
        height, width, _ = frame.shape
        result = []
        for i in range(all_boxes.shape[1]):
            if (all_classes[i] == 1) & (all_scores[i] > self.threshold):
                y1, x1, y2, x2 = (int(all_boxes[0, i, 0] * height),
                                  int(all_boxes[0, i, 1] * width),
                                  int(all_boxes[0, i, 2] * height),
                                  int(all_boxes[0, i, 3] * width))
                box = (min(x1, x2), min(y1, y2), max(x1, x2) - min(x1, x2), max(y1, y2) - min(y1, y2))
                result.append((box, all_scores[i]))
        return result

    def close(self):
        self.session.close()
