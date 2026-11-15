from pathlib import Path

import cv2

from src.video_io import open_video

def main():
    video_path = Path(__file__).resolve().parent / "data" / "original.mp4"
    cap = open_video(video_path)

    try:
        success, frame = cap.read()

        if not success or frame is None: raise RuntimeError("첫 프레임을 읽지 못했습니다.")

        print(f"프레임 크기: {frame.shape}")
        
        cv2.imshow("First frame", frame)
        cv2.waitKey(0)
   
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()