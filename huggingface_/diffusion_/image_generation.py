from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker = None).to("cuda")
res = pipe(prompt="A cute little cat.")
plt.imshow(res.images[0])
plt.show()

