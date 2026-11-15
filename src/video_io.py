from pathlib import Path

import cv2

def open_video(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)

    if not path.is_file(): raise FileNotFoundError(f"영상 파일이 없습니다: {path}")
    
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    return cap