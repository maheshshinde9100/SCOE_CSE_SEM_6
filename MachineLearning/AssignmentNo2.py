import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv('./datasets/cleaned_group_death_claims.csv')

print("Initial Shape:", df.shape)
df.replace(['\\N', 'NA', 'null', 'None'], np.nan, inplace=True)

df.drop_duplicates(inplace=True)

# Convert numeric columns explicitly
numeric_cols = [
    'claims_pending_start_no', 'claims_pending_start_amt',
    'claims_intimated_no', 'claims_intimated_amt',
    'total_claims_no', 'total_claims_amt',
    'claims_paid_no', 'claims_paid_amt',
    'claims_repudiated_no', 'claims_repudiated_amt',
    'claims_rejected_no', 'claims_rejected_amt',
    'claims_unclaimed_no', 'claims_unclaimed_amt',
    'claims_pending_end_no', 'claims_pending_end_amt',
    'claims_paid_ratio_no', 'claims_paid_ratio_amt',
    'claims_repudiated_rejected_ratio_no',
    'claims_repudiated_rejected_ratio_amt',
    'claims_pending_ratio_no', 'claims_pending_ratio_amt'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

print("After Cleaning Shape:", df.shape)

#removing outliers using IQR method..
Q1 = df['claims_paid_amt'].quantile(0.25)
Q3 = df['claims_paid_amt'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[
    (df['claims_paid_amt'] >= lower_bound) &
    (df['claims_paid_amt'] <= upper_bound)
]

plt.figure(figsize=(15, 12))

# bar Chart – Total Claims Paid by Insurer
plt.subplot(2, 2, 1)
df.groupby('life_insurer')['claims_paid_amt'].sum().sort_values(ascending=False).plot(
    kind='bar', color='teal'
)
plt.title("Total Claims Paid Amount by Insurer")

plt.ylabel("Amount")
plt.xticks(rotation=45)

# histogram – Claims Paid Amount Distribution
plt.subplot(2, 2, 2)
plt.hist(df_clean['claims_paid_amt'], bins=15, edgecolor='black')
plt.title("Distribution of Claims Paid Amount (Cleaned)")
plt.xlabel("Claims Paid Amount")

# Plot 3: Scatter Plot – Claims Intimated vs Claims Paid
plt.subplot(2, 2, 3)
plt.scatter(
    df_clean['claims_intimated_no'],
    df_clean['claims_paid_no'],
    alpha=0.6
)
plt.title("Claims Intimated vs Claims Paid")
plt.xlabel("Claims Intimated (Number)")
plt.ylabel("Claims Paid (Number)")

# Plot 4: Box Plot – Claims Paid Amount
plt.subplot(2, 2, 4)
plt.boxplot(df['claims_paid_amt'])
plt.title("Boxplot of Claims Paid Amount")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()

# correlation analysis
plt.figure(figsize=(10, 8))
numeric_df = df_clean.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap – Insurance Claims Data")

plt.show()

print("insurance Claims Analysis Done Successfully...")


