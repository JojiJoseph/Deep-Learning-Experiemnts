import requests
from transformers import pipeline
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import cv2

vqa = pipeline('vqa')

img = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/d/d0/Cycliste_%C3%A0_place_d%27Italie-Paris_crop.jpg", stream=True).raw)

result = vqa(img,"What is the vehicle shown in the image?")
print(result)

# zero shot vqa - not working
vqa = pipeline('vqa', model='Salesforce/blip2-opt-2.7b', device=0)
result = vqa(img,"What is the color of cycle?")
print(result)
