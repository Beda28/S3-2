import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self):
        self.model = cv2.createBackgroundSubtractorMOG2(
            history       = 500,
            varThreshold  = 16,
            detectShadows = True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.apply(frame, learningRate=0.01)

        _, clean = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)

        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN , self.kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        return raw, clean