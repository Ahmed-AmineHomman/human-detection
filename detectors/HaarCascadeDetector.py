import cv2

from .AbstractHumanDetector import AbstractHumanDetector


class HaarCascadeDetector(AbstractHumanDetector):
    def __init__(self, detector, confidence_threshold=0, **parameters):
        self.detector = cv2.CascadeClassifier(detector)
        self.threshold = confidence_threshold
        self.parameters = parameters
        self.boxes = []
        self.weights = []

    def detect(self, frame):
        # using a greyscale picture for faster detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        boxes, _, weights = self.detector.detectMultiScale3(image=gray_frame,
                                                            outputRejectLevels=True,
                                                            **self.parameters)
        boxes = [box for box in boxes]
        weights = [weight[0] for weight in weights]

        # drop all boxes not confident enough
        results = []
        for box, weight in list(zip(boxes, weights)):
            if weight > self.threshold:
                results.append((box, weight))

        return results
