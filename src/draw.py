import cv2
import numpy as np

def draw(frame, tracks):
    view  = frame.copy()
    color = (0, 255, 0)

    for item in tracks:
        x, y, w, h = item.box

        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            view, f"ID {item.id}", (x, max(20, y - 8)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        points = np.array(item.path, dtype=np.int32).reshape(-1, 1, 2)

        if len(points) > 1:
            cv2.polylines(view, [points], False, color, 2)
        if len(points) > 0:
            point = tuple(map(int, points[-1, 0]))
            cv2.circle(view, point, 4, (0, 0, 255), -1)

    cv2.putText(
        view, f"Regions: {len(tracks)}", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return view

def info(view, rate):
    cv2.putText(
        view, f"Processing FPS: {rate:.1f}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)