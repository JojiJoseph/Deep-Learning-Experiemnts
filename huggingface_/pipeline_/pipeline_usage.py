from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = ["I love the new Transformers movie!", "I hate you!", "100+100=201"]
result = classifier(sentences)
print(result)

generator = pipeline('text-generation')
text = generator(["In this tutorial, we will learn", "Hey, there..."], max_length=50, do_sample=True)
print(text)