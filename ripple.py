from typing import Deque
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

ret, prev = cap.read()
# prev_frame = None
h = 0
h_max = 640
q = Deque(maxlen=48)
while True:
    # h += 10
    # h %= 480
    # if h == 0:
    #     break
    # ret, frame = cap.read()
    # # if h >
    # temp = frame[:h,:,:].copy()
    # frame[:h,:,:] = prev[:h,:,:]
    # cv2.imshow("img", frame)
    # prev = frame
    ret, frame = cap.read()
    q.append(frame)
    frame = frame.copy()
    if len(q) == q.maxlen:
        for i, h in enumerate(range(0,480,10)):
            frame[max(0, h-30):h,:,:] = 0.1*frame[max(0, h-30):h,:,:] + 0.9*q[i][max(0,h-30):h,:,:]# + 0.2*frame[max(0, h-30):h,:,:]
        cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.waitKey()