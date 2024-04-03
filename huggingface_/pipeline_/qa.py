from transformers import pipeline

qa_pipe = pipeline('question-answering')
context = "The Indian batsman Sachin Tendulkar scored most international runs in cricket."
question = "Which cricketer scored the heighest number of runs in international cricket?"
result = qa_pipe(question=question, context=context)
print(result)