import time
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig, pipeline
from PIL import Image
checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )

# processor = AutoProcessor.from_pretrained(checkpoint)

# model = IdeficsForVisionText2Text.from_pretrained(
#     checkpoint,
#     quantization_config=quantization_config,
#     # torch_dtype=torch.float32,
#     device_map="auto"
# )

prompt = [
    { "role": "user", "content": "Which is the largest democracy in the world?"}
    # "Question: What is 1+1? Answer:"
]
pipe = pipeline("conversational", model=checkpoint, device=0)#, tokenizer=checkpoint)
# inputs = processor(prompt, return_tensors="pt").to("cuda")
# bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

# generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
# print(generated_text[0])
t1 = time.time()
res = pipe(prompt)
t2 = time.time()
print("time taken", t2-t1)
print(res)