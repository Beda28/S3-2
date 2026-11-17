import cv2

from src.video_io import save

class Control:
    def __init__(self):
        self.paused  = False
        self.running = True

    def handle(self, key, frame):
        if   key in (ord("q"), 27): self.running = False
        elif key == ord("p"):       self.paused = not self.paused
        elif key == ord("s"): 
            try: save(frame)
            except (OSError, RuntimeError, cv2.error) as e: print(f"저장 실패: {e}")