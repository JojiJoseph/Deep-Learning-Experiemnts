from wordcloud import WordCloud
import matplotlib.pyplot as plt
with open("1661-0.txt","r") as f:
    words = f.read()
# print(words)
cloud = WordCloud().generate(words)

plt.imshow(cloud)
plt.show()