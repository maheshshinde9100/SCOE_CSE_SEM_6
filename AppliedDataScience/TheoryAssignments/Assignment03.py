# =========================================================
# APPLIED DATA SCIENCE LAB
# Assignment 3 – Data Visualization
# Dataset: Global YouTube Statistics 2023
# Student Name : Mahesh Shinde
# =========================================================
# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------------------------------------
# 2. Load Dataset)
# ---------------------------------------------------------
df = pd.read_csv(
"./datasets/Global YouTube Statistics.csv",
encoding='latin1'
)
print("Dataset Loaded Successfully")
print("="*100)
# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns after cleaning:")
print(df.columns)
print("="*100)
# ---------------------------------------------------------
# 4. Basic Exploration
# ---------------------------------------------------------
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("="*100)
# ---------------------------------------------------------
# 5. Data Cleaning
# ---------------------------------------------------------
# Convert important numeric columns properly
numeric_columns = [
'subscribers', 'video_views',
'highest_yearly_earnings',
'video_views_for_the_last_30_days'
]
for col in numeric_columns:
df[col] = pd.to_numeric(df[col], errors='coerce')
# Fill missing numerical values with mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
df[col] = df[col].fillna(df[col].mode()[0])
print("Missing Values Handled Successfully")
print("="*100)
# ---------------------------------------------------------
# 6. HISTOGRAM – Subscribers Distribution
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(df['subscribers'], bins=30)
plt.title("Distribution of Subscribers")
plt.xlabel("Subscribers")
plt.ylabel("Frequency")
plt.show()
# ---------------------------------------------------------
# 7. BOXPLOT – Highest Yearly Earnings
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=df['highest_yearly_earnings'])
plt.title("Boxplot of Highest Yearly Earnings")
plt.show()
# ---------------------------------------------------------
# 8. SCATTER PLOT – Subscribers vs Video Views
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['subscribers'], df['video_views'])
plt.title("Subscribers vs Video Views")
plt.xlabel("Subscribers")
plt.ylabel("Video Views")
plt.show()
# ---------------------------------------------------------
# 9. BAR CHART – Top Channel Categories
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
df['category'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 YouTube Channel Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
print("Data Visualization Completed Successfully")
