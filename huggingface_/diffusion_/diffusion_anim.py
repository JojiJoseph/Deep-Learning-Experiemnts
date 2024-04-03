from diffusers import DDPMScheduler, UNet2DModel
import torch
import cv2

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")

sample_size = model.config.sample_size #* 2
# sample_size = 320

arr = []
input = torch.randn((1, 3, sample_size, sample_size), device="cuda")

# scheduler.set_timesteps(100)
print(scheduler.timesteps)
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample
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