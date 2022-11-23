
import math
import random
import numpy as np
import cv2

links = [2,4,2]
thetas = [0,0,0]
goal_x = 4
goal_y = 2
def forward(links, thetas):
    theta = 0
    x,y = 0,0
    for l, t in zip(links, thetas):
        theta += t
        x += l*math.cos(theta)
        y += l*math.sin(theta)
    return x, y

d = 1000
for iter in range(1000):
    thetas_new = [t+random.random()/2-0.25 for t in thetas]
    pre_x, pre_y = forward(links, thetas)
    x, y = forward(links, thetas_new)
    if (goal_x-x)**2 + (goal_y-y)**2 < d:
        d = (goal_x-x)**2 + (goal_y-y)**2
        # print(d)
        thetas = thetas_new
        # print(x,y)

    img = np.zeros((800,800,3)).astype(np.uint8)

    theta = 0
    pre_x, pre_y = 200, 200
    x, y = 0, 0
    for l, t in zip(links, thetas):
        theta += t
        x += l*math.cos(theta)
        y += l*math.sin(theta)
        cv2.circle(img, (200+int(x*100), int(200+y*100)),4,(0,0,255),-1)

        cv2.line(img, (pre_x, pre_y), (200+int(x*100), 200+int(y*100)),(255,0,0))
        pre_x = 200+int(x*100)
        pre_y = 200+int(y*100)
        # print(pre_x, pre_y)
    cv2.circle(img, (200+int(goal_x*100), 200+int(goal_y*100)),4,(0,255,255),-1)

    cv2.imshow("Img", img)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    print(d)
    if d < 0.02:
        print(iter)
        break
print("End of Simulation")
import time
time.sleep(2)
cv2.waitKey()