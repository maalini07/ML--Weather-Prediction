# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the weather station dataset and remove unnecessary columns and missing values.
2.Convert the categorical wind_direction column into numerical values and select the input and target variables.
3.Split the dataset into training and testing sets and train the Random Forest Regressor model.
4.Predict temperature, PM2.5, and energy values and evaluate the model using MSE and R² score.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: MAALINI B N
RegisterNumber:  212224060136
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

print("Dataset:")
print(data.head())

print("\nColumn Names:")
print(data.columns)

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

data = data.drop(columns=["time", "bat"])

data["wind_direction"] = data["wind_direction"].astype("category").cat.codes

data = data.dropna()

print("\nDataset after removing missing values:")
print(data.shape)

print("\nMissing Values after preprocessing:")
print(data.isnull().sum())

X = data.drop(columns=["tem", "pm2_5", "tsr"])
y = data[["tem", "pm2_5", "tsr"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

print("\nActual Values:")
print(y_test.head())

print("\nPredicted Values:")
print(y_pred[:5])

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print("\nPredicted Temperature:", prediction[0][0])
print("Predicted PM2.5:", prediction[0][1])
print("Predicted Energy:", prediction[0][2])

plt.figure(figsize=(7, 5))
plt.scatter(y_test["tem"], y_pred[:, 0])
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test["pm2_5"], y_pred[:, 1])
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(y_test["tsr"], y_pred[:, 2])
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted Energy")
plt.grid(True)
plt.show()
```

## Output:
<img width="778" height="319" alt="image" src="https://github.com/user-attachments/assets/e9bd4953-e3af-4382-b7cf-1a8300c77717" />

<img width="710" height="331" alt="image" src="https://github.com/user-attachments/assets/4b0014d8-2d4f-437d-841a-7087b9e4d52c" />

<img width="411" height="409" alt="image" src="https://github.com/user-attachments/assets/dd096b67-bb22-437c-bb56-09508496aa81" />

<img width="435" height="402" alt="image" src="https://github.com/user-attachments/assets/ccd3db59-5c7c-49db-853d-ba18bbdfb9b6" />

<img width="453" height="406" alt="image" src="https://github.com/user-attachments/assets/29684139-76dc-4b29-adca-4106d5bef606" />

<img width="609" height="468" alt="image" src="https://github.com/user-attachments/assets/9dc9e80b-ffa3-41b6-9a45-d4a5e795b3c1" />

<img width="618" height="468" alt="image" src="https://github.com/user-attachments/assets/ac581de6-3bde-45d9-b8c0-a99537112ef7" />

<img width="618" height="468" alt="image" src="https://github.com/user-attachments/assets/5c5c8e6e-1656-4b14-9bf7-ac5a4428974a" />

## Result:
Thus, the Random Forest Algorithm was successfully implemented to predict temperature, PM2.5 pollution level, and energy using environmental sensor data. The model was trained and tested successfully, and its performance was evaluated using Mean Squared Error and R² Score.
