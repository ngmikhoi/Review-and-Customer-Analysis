import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()
df.info()
df["flight_day"].unique()
mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)
df["flight_day"].unique()
df.describe()


x = df.iloc[:, 0:13]
x.drop(columns=['route'], inplace=True)
y = df.iloc[:, 13]


from sklearn.ensemble import RandomForestClassifier

x_numerical = x.select_dtypes(exclude=['object']).values
# x_categorical = x.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_categorical = x.select_dtypes(include=['object'])
# print(x_categorical)

for i in x_categorical.columns:
    print(i)
    items = x_categorical[i].unique()
    for item in items:
        name = i + '_' + item
        x[str(name)] = x[i].apply(lambda k: 1 if k == item else 0)
x.drop(columns= x_categorical.columns, inplace=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(x_train, y_train)

sort = model.feature_importances_.argsort()
importance = model.feature_importances_
print(importance)
print(x.columns)
result = pd.DataFrame({"Feature": list(x.columns) , "Importance": list(importance)})
result = result.sort_values(by = "Importance", ascending=True)
result = result.iloc[-20:,:]

import matplotlib.pyplot as plt
plt.barh(result['Feature'], result['Importance'],)
plt.title("Feature Importance")
plt.xlabel("Relative Importance")

plt.tight_layout()
plt.show()

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
F1_score = 2 * precision * recall / (precision + recall)

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",F1_score)