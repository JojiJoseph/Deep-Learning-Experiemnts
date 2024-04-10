import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", #torch_dtype=torch.float16
).to("cuda")
# pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# base and mask images
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(0)

prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
t1 = time.time()
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
t2 = time.time()
print("Time taken for inpaninting: ", t2-t1, " seconds")

out = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
plt.imshow(out)
plt.axis("off")
plt.show()


# To check what will happen if there is no prompt
generator = torch.Generator("cuda").manual_seed(0)
prompt = ""
negative_prompt = ""
t1 = time.time()
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
t2 = time.time()
print("Time taken for inpaninting: ", t2-t1, " seconds")

out = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
plt.imshow(out)
plt.axis("off")
plt.show()