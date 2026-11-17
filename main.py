from pathlib import Path
from time    import perf_counter

import cv2

from src.video_io   import open_video, load
from src.background import BackgroundSubtractor
from src.detect     import find
from src.track      import Tracker
from src.draw       import draw, info, mark, windows
from src.control    import Control
from src.match      import Matcher

def main():
    root    = Path(__file__).resolve().parent
    image   = load(root / "data" / "target.png")
    matcher = Matcher(image)

    video   = root / "data" / "original.mp4"
    cap     = open_video(video)
    back    = BackgroundSubtractor()
    track   = Tracker()
    control = Control()

    try:
        fps   = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, round(1000 / fps)) if fps > 0 else 33

        windows()
        while control.running:
            if not control.paused:
                start = perf_counter()
                success, frame = cap.read()

                if not success or frame is None: break 

                raw, clean = back.apply(frame)
                boxes  = find(clean, min=1200)
                tracks = track.update(boxes)
                result = matcher.match(frame)
                view   = draw(frame, tracks)

                mark(view, result)

                elap = perf_counter() - start
                rate = 1 / elap if elap > 0 else 0
                info(view, rate)

            cv2.imshow("Video"      , view)
            cv2.imshow("Raw mask"   , raw)
            cv2.imshow("Clean mask" , clean)

            wait = 30 if control.paused else delay
            key  = cv2.waitKey(wait) & 0xFF
            control.handle(key, view)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()