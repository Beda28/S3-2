import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self):
        self.model = cv2.createBackgroundSubtractorMOG2(
            history       = 500,
            varThreshold  = 16,
            detectShadows = True
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return self.model.apply(frame, learningRate=0.01)