import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig
from PIL import Image
checkpoint = "HuggingFaceM4/idefics-9b"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(checkpoint)

model = IdeficsForVisionText2Text.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    # torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(torch.cuda.memory_allocated()//1024//1024, "MB")
from pathlib import Path

prompt = [
    # "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
    Image.open("/mnt/hdd1/experiments/huggingface_/idefics_/idefics-im-captioning.jpg"),
    " What is the breed of this dog? \n Answer: "
]

inputs = processor(prompt, return_tensors="pt").to("cuda")
print(processor.tokenizer.decode(inputs["input_ids"][0]))
print(inputs["input_ids"][0])
for key in inputs:
    print(key, inputs[key].shape)
print(inputs["pixel_values"].min(), inputs["pixel_values"].max())
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
print(generated_ids[0])
print(processor.tokenizer.decode(generated_ids[0]))
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])