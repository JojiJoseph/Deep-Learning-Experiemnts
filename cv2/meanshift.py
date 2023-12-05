import numpy as np
import cv2

points = np.random.normal((200,200), 50, (1000,2))
img = np.zeros((400,400,3), np.uint8)
for p in points:
    cv2.circle(img, np.int0(p), 1, (255,255,255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)

# Meanshift roi
track_window = (50,0,100,100)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )
prev_window = None
for i in range(100):
    img_out = img.copy()
    x,y,w,h = track_window
    prev_window = track_window
    cv2.rectangle(img_out, (x,y), (x+w,y+h), (0,0,255), 2)
    ret, track_window = cv2.meanShift(img[...,0], track_window, term_crit)
    if prev_window == track_window:
        break
    cv2.imshow('img', img_out)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break