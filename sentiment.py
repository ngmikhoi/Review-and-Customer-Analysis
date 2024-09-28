import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("BA_clean_reviews.csv")
sia = SentimentIntensityAnalyzer()

positive = 0
neutral = 0
negative = 0
sentences = df['reviews']
for sentence in sentences:
    if(sia.polarity_scores(sentence)['compound'] > 0):
        positive += 1
    elif(sia.polarity_scores(sentence)['compound'] < 0):
        negative += 1
    else:
        neutral += 1

print(positive, neutral, negative)

sentiment_scores = df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['val'] = sentiment_scores.apply(lambda x: 'neutral' if x == 0 else('positive' if x > 0 else 'negative'))

print(df.loc[:, ['val', 'reviews']])

plt.pie([positive, neutral, negative], labels=['positive', 'neutral', 'negative'], startangle=90, colors=['green', 'grey', 'red'])
plt.legend(title='Reviews')
plt.show()

percentage = np.array([positive, neutral, negative])
percentage = percentage/sum(percentage) * 100
print(percentage)
