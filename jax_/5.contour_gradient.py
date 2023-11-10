from math import ceil, floor
from turtle import right
import numpy as np
import cv2

img = np.zeros((512, 512, 3)).astype(np.uint8)

clicked = False
finished = False
contour = []
def mouse_callback(event, x, y, flags, param):
    global clicked, finished, contour   
    if event == cv2.EVENT_LBUTTONDOWN and not finished:
        clicked = True
    if event == cv2.EVENT_MOUSEMOVE and clicked and not finished:
        print(x, y)
        contour.append([x, y])
    if event == cv2.EVENT_LBUTTONUP:
        if clicked:
            finished = True
        clicked = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
while True:
    img = np.zeros((512, 512, 3)).astype(np.uint8)
    if contour:
        print(np.array(contour).shape)
        # contour = [[0,0],[100,10], [240, 230]]

        contour_ = np.array(contour)
        x_c, y_c = contour_.mean(axis=0)
        # cv2.circle(img, (int(x_c), int(y_c)), 5, (255, 255, 0), -1)
        cv2.drawContours(img, [np.array(contour).reshape((-1,1,2))], -1, (255, 255, 255), 2)
    cv2.imshow('image', img)
    if finished:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Get centroid of contour
# indices of contour drawn on the image, not recorded
indices = np.where(img[:,:,0] == 255)
img[indices] = [0, 255, 0]
y_c = np.mean(indices[0])
x_c = np.mean(indices[1])
cv2.circle(img, (int(x_c), int(y_c)), 5, (255, 255, 0), -1)
cv2.imshow("image", img)
cv2.waitKey(0)
print(indices)

# Get color gradient
color1 = np.array([255, 255, 0])
color2 = np.array([0, 0, 255])

img = np.zeros((512, 524, 3)).astype(np.uint8)

YY = np.array(indices[0])
XX = np.array(indices[1])
XX, YY = np.meshgrid(XX, YY, indexing='xy')
cv2.circle(img, (int(x_c), int(y_c)), 5, (255, 255, 0), -1)
img[indices] = [0, 255, 0]
def get_color(x, y):
    vec = np.array([x, y]) - np.array([x_c, y_c])
    dist = 0
    alpha  = 0
    # dist = (XX - x_c)*vec[0] + (YY - y_c)*vec[1] / np.linalg.norm(vec)
    for y2, x2 in zip(*indices):
        vec2 = np.array([x2, y2]) - np.array([x_c, y_c])
        d = np.dot(vec, vec2) / np.linalg.norm(vec2) / np.linalg.norm(vec)
        if d > dist:
            dist = abs(d)
            alpha = np.dot(vec, vec2) / np.linalg.norm(vec2) / np.linalg.norm(vec2)
            # print(alpha)
    alpha = np.clip(alpha, 0, 1)
    return color1 * alpha + color2 * (1-alpha)

import bisect
search_space = []
search_space_angles = []

for x, y in zip(indices[1], indices[0]):
    angle = np.arctan2(y-y_c, x-x_c)
    search_space.append([angle, x, y])
    search_space.sort(key=lambda x: x[0])
    # search_space_angles.append(angle)

search_space_angles = [x[0] for x in search_space]

RAIN_BOW_COLORS = [ 
    [255, 0, 0],
    [255, 127, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 0, 255],
    [75, 0, 130],
    [148, 0, 211]
]

# Interpolate between colors to create 255 shades
# RAIN_BOW_COLORS_255 = np.inter(np.arange(0, 255), np.arange(0, 255, 255/len(RAIN_BOW_COLORS)), RAIN_BOW_COLORS).astype(np.uint8)

def get_color_bsearch(x, y):
    angle = np.arctan2(y-y_c, x-x_c)
    idx = bisect.bisect_left(search_space_angles, angle)
    candidates = [search_space[i][1:] for i in range(idx-1, idx+1)]
    vec = np.array([x, y]) - np.array([x_c, y_c])
    dist = 0
    alpha  = 0
    # dist = (XX - x_c)*vec[0] + (YY - y_c)*vec[1] / np.linalg.norm(vec)
    for x2, y2 in candidates:
        vec2 = np.array([x2, y2]) - np.array([x_c, y_c])
        d = np.dot(vec, vec2) / np.linalg.norm(vec2) / np.linalg.norm(vec)
        if d > dist:
            dist = d
            alpha = np.dot(vec, vec2) / np.linalg.norm(vec2) / np.linalg.norm(vec2)
            # print(alpha)
            best_x2 = x2
            best_y2 = y2
    # cv2.line(img, (int(x_c), int(y_c)), (int(best_x2), int(best_y2)), (0, 0, 255))#, 2)
    # cv2.line(img, (int(x_c), int(y_c)), (int(x), int(y)), (0, 255, 0))#, 2)
    alpha = np.clip(alpha, 0, 1)
    color_idx = int(alpha * 7)
    if alpha < 1:
        color1 = np.array(RAIN_BOW_COLORS[ceil(alpha*6)])
        color2 = np.array(RAIN_BOW_COLORS[floor(alpha*6)])
        alpha = alpha * 6 - floor(alpha*6)
        # print(np.int32(color2 * alpha + color1 * (1-alpha)))
        return RAIN_BOW_COLORS[color_idx][::-1]
        return np.int32(color1 * alpha + color2 * (1-alpha))[::-1]
    return [0, 0, 0]
    # return color1 * alpha + color2 * (1-alpha)

XY_stack = np.vstack((indices[1], indices[0])).T

def get_color_vec(x, y):
    vec = np.array([x, y]) - np.array([x_c, y_c])
    dist = 0
    alpha  = 0
    # dist = (XX - x_c)*vec[0] + (YY - y_c)*vec[1] / np.linalg.norm(vec)
    # D = (XX * vec[0] + YY * vec[1]) / np.linalg.norm(vec) / ((XX-x_c)**2 + (YY-y_c)**2)
    # print(XY_stack.shape, vec.shape)
    D = XY_stack @ vec
    # print(D.shape)
    D = D / np.linalg.norm(vec) / ((XY_stack[:,0]-x_c)**2 + (XY_stack[:,1]-y_c)**2)**0.5
    # D[D>]
    # print(XX.shape)
    D[D>1] = -1
    # print("D_shape", D.shape)
    idx = np.unravel_index(np.argmax(D), D.shape)
    # print(np.max(D))
    x2, y2 = XY_stack[idx]
    # print("x2 y2", x2, y2)
    # print("x y", x, y)
    # print("x_c y_c", x_c, y_c)
    # cv2.line(img, (int(x_c), int(y_c)), (int(x), int(y)), (0, 255, 0))#, 2)
    # cv2.line(img, (int(x_c), int(y_c)), (int(x2), int(y2)), (0, 0, 255))#, 2)
    vec2 = np.array([x2, y2]) - np.array([x_c, y_c])
    alpha = np.dot(vec, vec2) / np.linalg.norm(vec2) / np.linalg.norm(vec2)
    alpha = np.clip(alpha, 0, 1)
    return color1 * alpha + color2 * (1-alpha)

left = min(indices[1])
right = max(indices[1])
top = min(indices[0])
bottom = max(indices[0])
left, right, top, bottom = map(int, [left, right, top, bottom])


import bisect   

print(left, right, top, bottom)

cv2.destroyAllWindows()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
import tqdm
for i in tqdm.tqdm(range(top,bottom,1)):
    for j in range(left,right,1):
        # img[i, j] = np.clip(np.int32(get_color(j, i)), 0, 255)
        try:
            # img[i, j] = np.clip(np.int32(get_color(j, i)), 0, 255)
            # img[i, j] = 
            # img = np.zeros((512, 524, 3)).astype(np.uint8)
            img[i,j] = np.clip(np.int32(get_color_bsearch(j, i)), 0, 255)
            # cv2.circle(img, (j, i), 4, np.clip(np.int32(get_color(j, i), 0, 255)), -1)
        except:
            pass
        if i % 10 == 0 and j % 100 == 0:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()