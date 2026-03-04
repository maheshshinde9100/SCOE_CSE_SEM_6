# =====================================================
# Assignment 2: PCA + t-SNE (NaN Fixed Version)
# Name : Mahesh Shinde
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
data = pd.read_csv("/content/train.csv")

print("Dataset Shape:", data.shape)

# -----------------------------------------------------
# 2. Separate Features & Label
# -----------------------------------------------------
X = data.drop("label", axis=1)
y = data["label"]

# -----------------------------------------------------
# 3. Handle Missing Values (IMPORTANT FIX)
# -----------------------------------------------------
print("Total Missing Values:", X.isnull().sum().sum())

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# -----------------------------------------------------
# 4. Scale Features
# -----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------------------------------------
# 5. PCA
# -----------------------------------------------------
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

print("PCA Reduced Shape:", X_pca.shape)
print("Explained Variance (50 components):",
      np.sum(pca.explained_variance_ratio_))

# Plot cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

# -----------------------------------------------------
# 6. t-SNE (Use smaller sample for speed)
# -----------------------------------------------------
sample_size = min(2000, X_pca.shape[0])

X_sample = X_pca[:sample_size]
y_sample = y[:sample_size]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

print("t-SNE Output Shape:", X_tsne.shape)

# -----------------------------------------------------
# 7. Visualization
# -----------------------------------------------------
plt.figure(figsize=(10,8))
sns.scatterplot(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    hue=y_sample,
    palette="tab10",
    legend="full",
    s=50
)

plt.title("t-SNE Visualization of MNIST Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Digit")
plt.show()

# =====================================================
# END
# =====================================================
