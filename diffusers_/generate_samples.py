import matplotlib.pyplot as plt
import numpy as np

from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
print(generator)
images = generator(prompt="a photo of an spiderman riding a bike on saturn").images
print(len(images))
l = len(images)
for i in range(l):
    plt.figure()
    plt.imshow(images[i])
    plt.show()
# plt.imshow(images[0])
# plt.show()
