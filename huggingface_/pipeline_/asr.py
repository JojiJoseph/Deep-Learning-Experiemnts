from transformers import pipeline
import time
import numpy as np
import torch

# # from datasets import load_dataset
# # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# # speaker_embeddings = embeddings_dataset[7306]["xvector"]
# speaker_embeddings = embeddings_dataset[5665]["xvector"]
# for i, item in enumerate(embeddings_dataset):
#     if "ksp" in item['filename']:
#         print(i, item['filename'])
# speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)


pipe = pipeline(task='automatic-speech-recognition')

# pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts", device="cuda")#, model_kwargs={"speaker_embeddings":speaker_embeddings})
# text = "Hello World! This is a test.          "
# forward_params = {"speaker_embeddings": speaker_embeddings}

import soundfile as sf
wave, rate = sf.read("output.wav")
t1 = time.time()
output = pipe(wave)#,forward_params=forward_params)
t2 = time.time()
print(f"Time taken: {t2-t1}")
print(output)


# Save the audio file
# sf.write("output.wav", np.ravel(output["audio"]), output["sampling_rate"])
