import pandas as pd

df = pd.read_csv("BA_reviews.csv")

verified = df[df['reviews'].str.contains('Trip Verified', regex = False)]
print(verified.shape[0])

not_verified = df[df['reviews'].str.contains('Not Verified', regex = False)]
print(not_verified.shape[0])

df['reviews'] = df['reviews'].str.replace('Trip Verified', '')
df['reviews'] = df['reviews'].str.replace('Not Verified', '')
df['reviews'] = df['reviews'].str.replace('✅', '')
df['reviews'] = df['reviews'].str.replace('| ', '')
df.to_csv("BA_clean_reviews.csv")
