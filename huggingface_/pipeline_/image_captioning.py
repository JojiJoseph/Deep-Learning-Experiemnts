import requests
from transformers import pipeline
from PIL import Image

image_captioning = pipeline('image-to-text')

img = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/2/22/Interspecies_Friendship.jpg", stream=True).raw)

result = image_captioning(img)
print(result)