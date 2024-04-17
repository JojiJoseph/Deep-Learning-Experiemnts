from transformers import pipeline
pipe = pipeline('translation', model='facebook/nllb-200-distilled-600M')
res = pipe("Hello, I am a student", src_lang="eng_Latn", tgt_lang="mal_Mlym")
print(res)

res = pipe("ഹലോ, ഞാൻ ഒരു വിദ്യാർത്ഥിയാണ്.", src_lang="mal_Mlym", tgt_lang="eng_Latn")
print(res)

# Check available languages
print(sorted(pipe.tokenizer.additional_special_tokens))