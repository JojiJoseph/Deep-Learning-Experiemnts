# https://www.gutenberg.org/cache/epub/1513/pg1513.txt

from collections import deque, Counter, defaultdict
import numpy as np

text = ""
with open("pg1513.txt", "r") as f:
    text = f.read()

text = text.lower()

q = deque(maxlen=5)

freq = defaultdict(Counter)
q.extend(" " * 5)

for ch in text:
    #print("".join(list(q)))
    freq["".join(list(q))][ch] += 1
    q.append(ch)

q = deque("julie", maxlen=5)
text = ["julie"]
import random
for i in range(100):
    freq_count = [freq["".join(list(q))][ch] for ch in "abcdefghijklmnopqrstuvwxyz "]
    #print(freq_count)
    #freq_count = (1.2**np.array(freq_count)/np.sum(1.2**np.array(freq_count))).astype(int)
    if sum(freq_count) == 0:
        freq_count = [1] * len(freq_count)
    #print(freq_count)
    idx = random.sample(range(27),1, counts = freq_count)
    #idx = [np.argmax(freq_count)]
    #print(idx)
    #print(idx, freq_count, freq_count[idx[0]])
    q.append(("abcdefghijklmnopqrstuvwxyz ")[idx[0]])
    text.append(("abcdefghijklmnopqrstuvwxyz ")[idx[0]])
    #print(list(q))

print("".join(text))