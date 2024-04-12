from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
generator = torch.Generator("cuda").manual_seed(0)
image_without_lora = pipeline('A drawing of a woman', generator=generator).images[0]
generator = torch.Generator("cuda").manual_seed(0)
pipeline.load_lora_weights('artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl', weight_name='DoodleRedmond-Doodle-DoodleRedm.safetensors')
image_with_lora = pipeline('A drawing of a woman', generator=generator).images[0]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_without_lora)
axs[0].axis('off')
axs[0].set_title('Without LoRA')
axs[1].imshow(image_with_lora)
axs[1].axis('off')
axs[1].set_title('With LoRA')
plt.show()
print(pipeline.get_list_adapters())

