checkpoint = "HuggingFaceM4/idefics-9b"
import torch

from transformers import IdeficsForVisionText2Text, AutoProcessor

processor = AutoProcessor.from_pretrained(checkpoint)
import gc

for dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto")
    # Get total gpu memory allocated
    print("dtype", dtype, torch.cuda.memory_allocated()//1024//1024, "MB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Get total gpu memory allocated
    # print(torch.cuda.memory_allocated()//1024//1024, "MB")
    print("=="*20)

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(checkpoint)

model = IdeficsForVisionText2Text.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto"
)

print("dtype", model.dtype, torch.cuda.memory_allocated()//1024//1024, "MB")