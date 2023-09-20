import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in TTS().list_models():
    print(model_name)

model_name = "tts_models/en/ek1/tacotron2"
# Init TTS
tts = TTS(model_name, gpu=True)
print(tts.speakers)
print(tts.languages)
# exit()

# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
# wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file
# tts.tts_to_file(text="Hello! How are you?", speaker=tts.speakers[-1], language=tts.languages[0], file_path="output2.wav")
tts.tts_to_file(text="Coqui-AI TTS", speaker=None, language=None, file_path="output.wav")
