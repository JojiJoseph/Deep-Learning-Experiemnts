from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import matplotlib.pyplot as plt
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True, safety_checker = None, torch_type=torch.float16).to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# print(pipe.unet)
# pipe = torch.compile(pipe)#.cuda()
generator = torch.Generator("cuda")
# res = pipe(prompt="A cute little cat.", generator=generator)
res = pipe(prompt="A cute little panda.", generator=generator)
plt.imshow(res.images[0])
plt.show()

