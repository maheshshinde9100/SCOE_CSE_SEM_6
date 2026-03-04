# ==============================================================
# Implementation and Optimization of Ensemble Models
# Case Study: XGBoost & LightGBM (Regression Problem)
# Dataset: California Housing Dataset
# ==============================================================

# ======================
# 1. Import Libraries
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")

# ======================
# 2. Load Dataset
# ======================

file_path = "./dataset/california_housing.csv"   
target_column = "medianHouseValue"

df = pd.read_csv(file_path)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# ======================
# 3. Separate Features & Target
# ======================

X = df.drop(columns=[target_column])
y = df[target_column]

# Handle missing values (if any)
X.fillna(X.median(), inplace=True)

# ======================
# 4. Train-Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ======================
# 5. XGBoost Regressor
# ======================

print("\n========== XGBOOST REGRESSOR ==========")

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_xgb)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ======================
# 6. Cross Validation
# ======================

print("\n========== CROSS VALIDATION ==========")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    xgb_model,
    X,
    y,
    cv=kf,
    scoring='r2'
)

print("Cross Validation R2 Scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())

# ======================
# 7. Hyperparameter Tuning
# ======================

print("\n========== GRID SEARCH OPTIMIZATION ==========")

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV R2 Score:", grid_search.best_score_)

# ======================
# 8. LightGBM Regressor
# ======================

print("\n========== LIGHTGBM REGRESSOR ==========")

lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)

print("LightGBM R2:", r2_score(y_test, y_pred_lgb))
print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lgb)))

# ======================
# 9. Feature Importance
# ======================

print("\n========== FEATURE IMPORTANCE ==========")

importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importance - XGBoost")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

print("\nAssignment Completed Successfully ✅")
