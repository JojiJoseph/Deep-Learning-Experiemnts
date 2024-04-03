import requests
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_segmentation = pipeline('image-segmentation', model="facebook/maskformer-swin-small-coco", device=0)

img = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg", stream=True).raw)

result = image_segmentation(img)
img_numpy = np.asarray(img)
for item in result:
    print(item)
    plt.figure()
    img_out = 0.5 * img_numpy + 0.5 * np.asarray(item['mask'])[...,None]
    plt.title(item['label'])
    plt.imshow(img_out.astype(np.uint8))
    plt.show()