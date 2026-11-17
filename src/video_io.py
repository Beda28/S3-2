from pathlib  import Path
from datetime import datetime

import cv2

def open_video(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)

    if not path.is_file(): raise FileNotFoundError(f"영상 파일이 없습니다: {path}")
    
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    return cap

def load(path):
    path  = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
    
    image = cv2.imread(str(path))
    if image is None: raise RuntimeError(f"이미지를 열 수 없습니다: {path}")
    
    return image

def save(frame):
    folder = Path(__file__).resolve().parent.parent / "results" / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)

    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f.png")
    path = folder / name

    if not cv2.imwrite(str(path), frame): raise RuntimeError(f"이미지 저장에 실패했습니다: {path}")
    return path