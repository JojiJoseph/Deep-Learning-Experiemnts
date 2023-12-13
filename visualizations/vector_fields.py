import cv2
import numpy as np
from collections import deque

cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.count = 0
        self.path = [[x, y]]
        self.color = np.random.randint(0, 255, (3)).tolist()
    def update(self, x, y):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x = x
        self.y = y
        self.path.append((self.x, self.y))
        self.count += 1

points = deque()

def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append(Point(x, y))

cv2.setMouseCallback("Output", mouse_callback)


def swirl_force(x, y):
    x -= 256
    y -= 256
    return (y-x), (-x-y)

while True:
    img = np.zeros((512, 512, 3), np.uint8)
    for _ in range(len(points)):
        point = points.popleft() if len(points) > 0 else None
        
        if point:
            # cv2.circle(img, (int(point.x), int(point.y)), 5, point.color, -1)
            cv2.polylines(img, np.int32([point.path]), False, point.color, 2)
            # cv2.line(img, (int(point.pre), int(point.y)), (int(point.x), int(point.y)), point.color, 2)
            # cv2.putText(img, str(point.count), (point.x, point.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(img, str(point.path), (point.x, point.y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            dt = 0.01
            fx, fy = swirl_force(point.x, point.y)
            point.update(point.x + dt * fx, point.y + dt * fy)

            if point.count < 1000:
                points.append(point)
    cv2.imshow("Output", img)
    if cv2.waitKey(10) == ord('q'):
        break