# Assignment 2 – Normalization & Standardization
# Dataset: Housing Prices (Kaggle)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("=" * 80)
print("Libraries Imported Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("Housing.csv")

print("Dataset Loaded Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

print("Column Names:")
print(df.columns.tolist())
print("=" * 80)

# ---------------------------------------------------------
# 4. Display Basic Information
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())
print("=" * 80)

print("Dataset Shape:", df.shape)
print("=" * 80)

print("Missing Values:")
print(df.isnull().sum())
print("=" * 80)

# ---------------------------------------------------------
# 5. Remove Duplicate Records
# ---------------------------------------------------------
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
final_rows = df.shape[0]

print(f"Duplicates Removed: {initial_rows - final_rows}")
print("=" * 80)

# ---------------------------------------------------------
# 6. Handle Missing Values
# ---------------------------------------------------------
# Fill numerical columns with mean
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 7. Select Only Numerical Columns for Scaling
# ---------------------------------------------------------
numeric_data = df.select_dtypes(include=np.number)

print("Numerical Columns Selected:")
print(numeric_data.columns.tolist())
print("=" * 80)

# ---------------------------------------------------------
# 8. Normalization (Min-Max Scaling)
# ---------------------------------------------------------
minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(numeric_data)

normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)

print("Normalized Data (First 5 Rows):")
print(normalized_df.head())
print("=" * 80)

# ---------------------------------------------------------
# 9. Standardization (Z-Score Scaling)
# ---------------------------------------------------------
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(numeric_data)

standardized_df = pd.DataFrame(standardized_data, columns=numeric_data.columns)

print("Standardized Data (First 5 Rows):")
print(standardized_df.head())
print("=" * 80)

print("Data Handling Completed Successfully ✅")
