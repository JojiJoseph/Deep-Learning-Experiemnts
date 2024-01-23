import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

def generate():
  model = GenerativeModel("gemini-pro")
  responses = model.generate_content(
    """What is your knowledge cut off?""",
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1
    },
  stream=True,
  )
  
  for response in responses:
      print(response.text, end="")


generate()
