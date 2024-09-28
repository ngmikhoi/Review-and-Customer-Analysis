import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("BA_clean_reviews.csv")
print(df.shape[0])
service = df[df['reviews'].str.contains('service', regex = True)][['reviews']].rename(columns={'reviews': 'service'})
print(service.shape[0])

seat = df[df['reviews'].str.contains('seat', regex = True)][['reviews']].rename(columns={'reviews': 'seat'})
print(seat.shape[0])

food = df[df['reviews'].str.contains('food|cuisine|beverage|drink', regex = True)][['reviews']].rename(columns={'reviews': 'food'})
print(food.shape[0])

staff = df[df['reviews'].str.contains('crew|staff', regex = True)][['reviews']].rename(columns={'reviews': 'staff'})
print(staff.shape[0])

classes = df[df['reviews'].str.contains('class|classes|Economy|Premium Economy| Business|First Class', regex = True)][['reviews']].rename(columns={'reviews': 'class'})
print(classes.shape[0])


ans = pd.DataFrame(data = {'classification': ['service','staff', 'seat', 'food', 'class'],
                           'frequency': [service.shape[0], staff.shape[0], seat.shape[0], food.shape[0], classes.shape[0]]})
print(ans)

plt.bar(ans['classification'], ans['frequency'])

plt.ylabel('Frequency')
plt.show()








