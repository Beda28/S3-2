import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self):
        self.model = cv2.createBackgroundSubtractorMOG2(
            history       = 500,
            varThreshold  = 36,
            detectShadows = True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.ref    = None
        self.count  = 0

    def adjust(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (90, 51), interpolation=cv2.INTER_AREA)
        small = small.astype(np.int16)

        if self.ref is None: self.ref = small.copy()

        diff  = small - self.ref
        shift = np.median(diff.reshape(-1, 3), axis=0)
        frame = frame.astype(np.int16) - shift

        return np.clip(frame, 0, 255).astype(np.uint8)
        

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.count += 1
        rate  = 0.01 if self.count <= 300 else 0.0001
        frame = self.adjust(frame)
        raw   = self.model.apply(frame, learningRate=rate)

        _, clean = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN , self.kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        return raw, clean