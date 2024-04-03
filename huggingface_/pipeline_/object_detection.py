import requests
from transformers import pipeline
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import cv2

object_detection = pipeline('object-detection')

img = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/d/d0/Cycliste_%C3%A0_place_d%27Italie-Paris_crop.jpg", stream=True).raw)

result = object_detection(img)
img_numpy = np.asarray(img).copy()
for item in result:
    print(item)
    cv2.rectangle(img_numpy, (item['box']['xmin'], item['box']['ymin']), (item['box']['xmax'], item['box']['ymax']), (0, 255, 0), 2)
    cv2.putText(img_numpy, item['label'], (item['box']['xmin'], item['box']['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
plt.figure()
plt.imshow(img_numpy)
plt.show()