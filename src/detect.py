import cv2
import numpy as np

def find(mask: np.ndarray, min: float = 150):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes       = []

    for contour in contours:
        if cv2.contourArea(contour) < min: continue

        boxes.append(cv2.boundingRect(contour))

    return boxes