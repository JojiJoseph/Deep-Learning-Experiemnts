import cv2
from face_detection import RetinaFace
import numpy as np
import time

detector = RetinaFace()
camera = cv2.VideoCapture(0)
while True:
    _, img = camera.read()
    t1 = time.time()
    faces = detector(img)
    print(time.time()-t1)
    if faces:
        box, landmarks, score = faces[0]
        cv2.rectangle(img, np.int0((box[0],box[1])), np.int0((box[2],box[3])),(0,0,255))
    cv2.imshow(" ", cv2.flip(img,1))
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
