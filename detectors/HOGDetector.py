import cv2

from .AbstractHumanDetector import AbstractHumanDetector


class HOGDetector(AbstractHumanDetector):
    def __init__(self, detector, confidence_threshold, **parameters):
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(detector)
        self.threshold = confidence_threshold
        self.parameters = parameters
        self.boxes = []
        self.weights = []

    def detect(self, frame):
        # using a greyscale picture for faster detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        boxes, weights = self.detector.detectMultiScale(img=gray_frame,
                                                        finalThreshold=self.threshold,
                                                        **self.parameters)
        boxes = [box for box in boxes]
        weights = [weight[0] for weight in weights]

        # drop all boxes not confident enough
        results = []
        for box, weight in list(zip(boxes, weights)):
            if weight > self.threshold:
                results.append((box, weight))

        return results
