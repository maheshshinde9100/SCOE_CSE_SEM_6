# Lab Assignment no 4
# Mahesh Shinde - TY-BT3-150
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/content/california_housing.csv")

print("First 5 rows of dataset")
print(df.head())

print("\nDataset Shape:", df.shape)

# -----------------------------
# 3. Data Cleaning
# -----------------------------
df["medianHouseValue"] = df["medianHouseValue"].astype(str).str.replace('"','')
df["medianHouseValue"] = df["medianHouseValue"].astype(float)
# -----------------------------
# 4. Define Features and Target
# -----------------------------
X = df.drop("medianHouseValue", axis=1)
y = df["medianHouseValue"]

# -----------------------------
# 5. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# -----------------------------
# 6. Create Base Model
# -----------------------------
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)
# =========================================================
# 7. Hyperparameter Tuning using GridSearchCV
# =========================================================
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=2,
    n_jobs=-1
)

print("\nRunning Grid Search...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearch:")
print(grid_search.best_params_)
# -----------------------------
# Best Model from GridSearch
# -----------------------------
best_grid_model = grid_search.best_estimator_

y_pred_grid = best_grid_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_grid)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_grid)

print("\nGridSearch Results")
print("RMSE:", rmse)
print("R2 Score:", r2)
# =========================================================
# 8. Hyperparameter Tuning using RandomizedSearchCV
# =========================================================

param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=6,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    verbose=2,
    n_jobs=-1
)

print("\nRunning Randomized Search...")
random_search.fit(X_train, y_train)

print("\nBest Parameters from RandomizedSearch:")
print(random_search.best_params_)
# -----------------------------
# Best Model from Random Search
# -----------------------------
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)

mse2 = mean_squared_error(y_test, y_pred_random)
rmse2 = mse2 ** 0.5
r22 = r2_score(y_test, y_pred_random)

print("\nRandomizedSearch Results")
print("RMSE:", rmse2)
print("R2 Score:", r22)
# =========================================================
# 9. Final Model Selection
# =========================================================

print("\nFinal Model Comparison")
if r2 > r22:
    print("GridSearch Model performs better.")
    final_model = best_grid_model
    y_pred = y_pred_grid
else:
    print("RandomizedSearch Model performs better.")
    final_model = best_random_model
    y_pred = y_pred_random

# =========================================================
# 10. Graph 1 : Actual vs Predicted Prices
# ========================================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
# =========================================================
# 11. Graph 2 : Feature Importance
# =========================================================
plt.figure(figsize=(8,6))
xgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.show()
# =========================================================
# 12. Graph 3 : Residual Error Distribution
# =========================================================
errors = y_test - y_pred

plt.figure(figsize=(6,4))
plt.hist(errors, bins=30)
plt.title("Residual Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()
