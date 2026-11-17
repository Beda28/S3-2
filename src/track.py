from collections import deque
from dataclasses import dataclass, field
from math        import hypot

def center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2

@dataclass
class Track:
    id  : int
    box : tuple
    miss: int = 0
    path: deque = field(default_factory=lambda: deque(maxlen=40))

class Tracker:
    def __init__(self, distance = 60, keep = 10):
        self.distance = distance
        self.keep     = keep
        self.tracks   = {}
        self.next_id  = 1

    def update(self, boxes):
        pairs = []

        for uid, track in self.tracks.items():
            tx, ty = center(track.box)

            for i, box in enumerate(boxes):
                cx, cy = center(box)
                dist   = hypot(tx - cx, ty - cy)

                if dist <= self.distance: pairs.append((dist, uid, i))

        used_ids = set()
        used_box = set()

        for _, uid, i in sorted(pairs):
            if uid in used_ids or i in used_box: continue

            track = self.tracks[uid]
            track.box  = boxes[i]
            track.miss = 0
            track.path.append(center(track.box))

            used_ids.add(uid)
            used_box.add(i)

        for uid in list(self.tracks):
            if uid in used_ids: continue

            track = self.tracks[uid]
            track.miss += 1

            if track.miss > self.keep: del self.tracks[uid]

        for i, box in enumerate(boxes):
            if i in used_box: continue

            uid   = self.next_id
            track = Track(uid, box)
            track.path.append(center(box))
            
            self.tracks[uid] = track
            self.next_id    += 1

        return [t for t in self.tracks.values() if t.miss == 0]