from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker = None).to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

res = pipe(prompt="A cute little cat.")
res = pipe(prompt="A cute little cat.")
plt.imshow(res.images[0])
plt.show()

