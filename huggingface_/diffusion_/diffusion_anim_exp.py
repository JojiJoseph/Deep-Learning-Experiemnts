from diffusers import DDPMScheduler, UNet2DModel
import requests
import torch
import cv2
from PIL import Image

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
import numpy as np
sample_size = model.config.sample_size #* 2
# sample_size = 320

arr = []
input = torch.randn((1, 3, sample_size, sample_size), device="cuda")

condition_image = Image.open(requests.get("https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/e073c8191b03635372961f219ef5ca3ad7d60b65eb2a71a7dbd1a3a28f86b4fe?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27sdxl-text2img.png%3B+filename%3D%22sdxl-text2img.png%22%3B&response-content-type=image%2Fpng&Expires=1712418270&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMjQxODI3MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9lMDczYzgxOTFiMDM2MzUzNzI5NjFmMjE5ZWY1Y2EzYWQ3ZDYwYjY1ZWIyYTcxYTdkYmQxYTNhMjhmODZiNGZlP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=B4btG2%7EtVf8%7EEbX%7EkAa2JSvcSv2K76-q0eqZc570RE2bRI26KmTWHfQbHelmNAMCNBcWNUZvKoQyV5O5UpYuwYddaurNyp5YizVJ5MJJUsmpkrZIS7HgKzQSDWTum%7ETV%7E7fRS0S9r5ouKobGCv7-T7l4BhMKl%7Ekcc%7EX0PJyhlxDHAGWnbjJl1lOgwYgZm%7EcvgzR0PGde7eGJHR4an1UQ9kDZq81ZcLNYj5Y9yEM6mSBNsvD%7EO98ideQiSqV9dz2wG2mDeLLc44TrXioJ%7Ev6IV3UFs4qjkkZsk8WZ4k0IZ%7E4qmebKFzAtoMT1mrTUwEKv7b0t23M8YOe0Mry6MFbJMw__&Key-Pair-Id=KVTP0A1DKRTAX", stream=True).raw)
condition_image = condition_image.resize((sample_size, sample_size))
condition_image = condition_image.convert("RGB")
condition_image = np.asarray(condition_image).copy()
condition_image = torch.tensor(condition_image).permute(2, 0, 1).float().cuda() / 255.0 - 0.5
condition_image = condition_image / 100.0

scheduler.set_timesteps(100)
print(scheduler.timesteps)
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample
    if t > 950:
        input += condition_image
        condition_image = condition_image * 0.99
    arr.append(input[0].cpu().numpy().transpose(1, 2, 0))
    img = arr[-1]
    img = (img - img.min()) / (img.max() - img.min())
    arr[-1] = (img * 255).astype("uint8")
    cv2.imshow("current", arr[-1])
    # print(arr[-1].max(), arr[-1].min())
    print(t)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord("q")]:
        break

cv2.imshow("final", arr[-1])
cv2.waitKey(0)