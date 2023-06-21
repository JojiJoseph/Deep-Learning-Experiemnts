# https://github.com/secnot/rectpack
import cv2
import numpy as np

from rectpack import newPacker

rectangles = [(100, 30), (40, 60), (30, 30),(70, 70), (100, 50), (30, 30)]
bins = [(200, 200), (80, 40), (100, 150)]

packer = newPacker()

# Add the rectangles to packing queue
for r in rectangles:
	packer.add_rect(*r)

# Add the bins where the rectangles will be placed
for b in bins:
	packer.add_bin(*b)

# Start packing
packer.pack()

nbins = len(packer)
for i in range(nbins):
	rects = packer[i]
	img = np.zeros((packer[i].height, packer[i].width, 3))
	for rect in rects:
		cv2.rectangle(img, (rect.x, rect.y,  rect.width, rect.height), (255, 0, 0))
	cv2.imshow("img", img)
	cv2.waitKey()
	print(packer[i].height)