from dataclasses import dataclass

import cv2

@dataclass
class Result:
    target   : int
    features : int
    good     : int
    rate     : int
    found    : bool

class Matcher:
    def __init__(self, image, ratio=0.75, limit=12):
        self.ratio = ratio
        self.limit = limit
        self.orb   = cv2.ORB_create(nfeatures=1000)
        self.bf    = cv2.BFMatcher(cv2.NORM_HAMMING)

        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        points, self.desc = self.orb.detectAndCompute(gray, None)
        self.count = len(points)

    def match(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, desc = self.orb.detectAndCompute(gray, None)
        good = []

        if self.desc is not None and desc is not None and len(desc) >= 2:
            pairs = self.bf.knnMatch(self.desc, desc, k=2)

            for pair in pairs:
                if len(pair) < 2: continue
                first, second = pair
                
                if first.distance < self.ratio * second.distance:
                    good.append(first)

        count = len(good)
        rate  = 100 * count / self.count if self.count else 0

        return Result(
            self.count, len(points),
            count, rate, count >= self.limit
        )