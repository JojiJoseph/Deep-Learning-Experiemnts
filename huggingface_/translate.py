from transformers import pipeline

text = "Hugging Face is a community-based open-source platform for machine learning."
translator = pipeline(task="translation_en_to_ml", model="Helsinki-NLP/opus-mt-en-ml")
out = translator(text)
print(out)