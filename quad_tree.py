import numpy as np
import cv2

class QuadTree:
    def __init__(self, mtx, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.mtx = mtx
        self.children = []
    def construct(self):
        # print(self.xmin, self.xmax)
        if self.xmin == self.xmax and self.ymin == self.ymax:
            if self.mtx[self.ymin, self.xmin] > 0:
                return self
            else:
                return None
        self.children = []
        dx = (self.xmax+1-self.xmin)//2
        dy = (self.ymax+1-self.ymin)//2
        for i in range(2):
            # print("#", xstart, dx)
            for j in range(2):
                xstart = self.xmin + i * dx
                ystart = self.ymin + j * dy
                next_node = QuadTree(self.mtx, xstart, ystart, xstart+dx-1, ystart+dy-1)
                next_node = next_node.construct()
                if next_node:
                    self.children.append(next_node)
        if self.children:
            return self
        return None
    def visualize(self, img):
        if self.children:
            dx = (self.xmax+1-self.xmin)//2
            dy = (self.ymax+1-self.ymin)//2
            for i in range(2):
                for j in range(2):
                    cv2.rectangle(img, (self.xmin+ i * dx, self.ymin+ j * dy), (self.xmax+i * dx +dx, self.ymax+j+dy+j), (0,0,255), 1)
        for child in self.children:
            child.visualize(img)

cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
img = cv2.imread("./opencv-logo.png", 0)
ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
img = cv2.resize(img, (400,400))
print(img.shape)
tree = QuadTree(img, 0, 0, 399, 399)
tree = tree.construct()
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
tree.visualize(img)
print(tree)
cv2.imshow("Output", img)
cv2.waitKey()