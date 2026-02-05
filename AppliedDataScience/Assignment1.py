import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/credit_card_fraud_dataset.csv')

print("Dataset loaded successfully")
print("Initial Shape:", df.shape)
print(df.head())

# data cleaning
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

print("\nMissing Values:")
print(df.isnull().sum())

# remove dup
df.drop_duplicates(inplace=True)

print("\nAfter removing duplicates:")
print(df.shape)

# feature engineering
df['TransactionDate'] = df['TransactionDate'].astype('int64') // 10**9

X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

# class distribution
print("\nOriginal class distribution:")
print(Counter(y))

# categorical encoding
X = pd.get_dummies(
    X,
    columns=['TransactionType', 'Location'],
    drop_first=True
)

print("\nAfter One-Hot Encoding:")
print(X.head())

# SMOTE balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE class distribution:")
print(Counter(y_resampled))

# balanced dataframe
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['IsFraud'] = y_resampled

print("\nBalanced dataset created successfully")
print(df_balanced.head())

# final split
X = df_balanced.drop(['IsFraud', 'TransactionID'], axis=1)
y = df_balanced['IsFraud']

print("\nFinal Feature & Target Shape:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# feature scaling
scaler = StandardScaler()

X_scaled = X.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nNumerical features scaled successfully")
print(X_scaled.head())

# correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(X_scaled.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap After Scaling")

plt.show()

print("\nAssignment 1 preprocessing completed successfully.")
