from abc import abstractmethod, ABC


class AbstractHumanDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass
