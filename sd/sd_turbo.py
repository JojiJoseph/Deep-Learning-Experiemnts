# from diffusers import AutoPipelineForText2Image
# import torch
# import matplotlib.pyplot as plt

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipeline_text2image = pipeline_text2image.to("cuda")

# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
# image
# # print(image.shape, image.min(), image.max())
# plt.imshow(image)
# plt.show()

from diffusers import AutoPipelineForText2Image
import torch
import gradio as gr

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

def generate_image(prompt, negative_prompt):
    image = pipeline_text2image(prompt=prompt,negative_prompt=negative_prompt, guidance_scale=1.0, num_inference_steps=2).images[0]
    return image

iface = gr.Interface(fn=generate_image, inputs=["text", "text"], outputs="image")
iface.launch()