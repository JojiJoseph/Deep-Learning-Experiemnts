import torch
from diffusers import ShapEPipeline
from diffusers.utils.export_utils import export_to_obj, export_to_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["paper boat", "A birthday cupcake"]

res = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
    output_type="pil" # pil, np, latent, mesh  
)
print(res['images'][0])
# export_to_obj(res['images'][0], "output.obj")
export_to_gif(res['images'][0], "output.gif")
