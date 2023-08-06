import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

data=open('C:/Users/yit_z/Downloads/part-r-00000.txt', 'r')
print(data.read())

# create wordcloud
data = open("C:/Users/yit_z/Downloads/part-r-00000.txt").read()

wordcloud = WordCloud(font_path='C:\\Windows\\Fonts\\Verdana.ttf', 
                      stopwords=STOPWORDS, 
                      background_color='white', 
                      width=1200, 
                      height=1000).generate(data)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()
