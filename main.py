from pathlib import Path

import cv2

from src.video_io   import open_video
from src.background import BackgroundSubtractor
from src.detect     import find
from src.track      import Tracker
from src.draw       import draw

def main():
    video = Path(__file__).resolve().parent / "data" / "original.mp4"
    cap   = open_video(video)
    back  = BackgroundSubtractor()
    track = Tracker()

    try:
        fps   = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, round(1000 / fps)) if fps > 0 else 33

        while True:
            success, frame = cap.read()

            if not success or frame is None: 
                print("영상 읽기가 끝났습니다.")
                break 

            raw, clean = back.apply(frame)
            boxes  = find(clean, min=150)
            tracks = track.update(boxes)
            view   = draw(frame, tracks)

            cv2.imshow("Video"      , view)
            cv2.imshow("Raw mask"   , raw)
            cv2.imshow("Clean mask" , clean)

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27): break
   
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()