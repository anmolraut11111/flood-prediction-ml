import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
import time


data = pd.read_csv('flood_data.csv')


X = data.drop('flood', axis=1)
y = data['flood']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


start = time.time()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Normal Training Time:", round(time.time() - start, 4), "seconds")


start = time.time()
with parallel_backend('threading'):
    model.fit(X_train, y_train)
print("Parallel Training Time:", round(time.time() - start, 4), "seconds")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
