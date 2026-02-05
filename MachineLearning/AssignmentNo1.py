import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_pits = pd.read_csv('./datasets/pit_stops.csv')
df_drivers = pd.read_csv('./datasets/drivers.csv')
df = pd.merge(df_pits, df_drivers, on='driverId')

# Handling Noisy data
df.replace(r'\N', np.nan, inplace=True)

df.drop_duplicates(inplace=True) #removing dup records

if 'forename' in df.columns and 'surname' in df.columns:
    df['DriverFull'] = df['forename'] + " " + df['surname']

# Handling Missing Values: Filling missing cars numbers with 0
if 'number' in df.columns:
    df['number'] = pd.to_numeric(df['number'], errors='coerce').fillna(0)

# 3. OUTLIER REMOVING USING IQR METHOD
Q1 = df['milliseconds'].quantile(0.25)
Q3 = df['milliseconds'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#data filterring
df_clean = df[(df['milliseconds'] >= lower_bound) & (df['milliseconds'] <= upper_bound)]
plt.figure(figsize=(15, 12))

# Bar graph - Nationality Distribution
plt.subplot(2, 2, 1)
df['nationality'].value_counts().head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Driver Nationalities")
plt.ylabel("Frequency")

# Histogram - Pit Stop Duration
plt.subplot(2, 2, 2)
plt.hist(df_clean['milliseconds'] / 1000, bins=15, color='orange', edgecolor='black')
plt.title("Distribution of Pit Stop Times (Cleaned)")
plt.xlabel("Seconds")

# Scatter Plot - lap vs duration
plt.subplot(2, 2, 3)
plt.scatter(df_clean['lap'], df_clean['milliseconds'] / 1000, alpha=0.4, color='blue' )

plt.title("Lap Number vs Duration")
plt.xlabel("Lap")
plt.ylabel("Seconds")

# Box Plot ->Detecting Outliers
plt.subplot(2, 2, 4)
plt.boxplot(df['milliseconds'].dropna() / 1000)
plt.title("Boxplot of Durations (with Outliers)")
plt.ylabel("Seconds")

plt.tight_layout()
plt.show()

# 5. Correlation Analysis
plt.figure(figsize=(8, 6))
numeric_df = df_clean.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Analysis Heatmap")
plt.show()

print("Preprocessing Completed...")