# https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data?select=AEP_hourly.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
df = pd.read_csv("AEP_hourly.csv")

# convert datetime column
df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df.set_index('Datetime')
print(df.head())

df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
print(df.head())


train = df[df.index < '2017-01-01']
test = df[df.index >= '2017-01-01']

features = ['hour','day','month','year','dayofweek']
target = 'AEP_MW'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Train Model (XGBoost)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE :", rmse)
print("R2 Score :", r2)

# Plot Results
plt.figure(figsize=(12,6))

plt.plot(y_test.values[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")

plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Time")
plt.ylabel("Energy (MW)")
plt.legend()

plt.show()