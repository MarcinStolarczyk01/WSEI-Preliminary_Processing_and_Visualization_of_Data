import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv("lab5/data/kc_house_data.csv")

df.describe()

features = df.drop(columns="price")


scaler = StandardScaler()
values = scaler.fit_transform(features)


pca = PCA()
pca_values = pca.fit_transform(values)
expl_variance = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_variance)  # Cumulated variance for all features


components = range(1, len(expl_variance) + 1)
plt.figure(figsize=[10, 7])
plt.plot(components, expl_variance, marker="o", label="Explained Variance")
plt.plot(components, cum_var, marker="s", label="Cumulative Variance")

plt.title("PCA Explained Variance")
plt.xlabel("Principal component")
plt.ylabel("Variance ratio")

plt.xticks(np.arange(0, len(components) + 1))
plt.yticks(np.linspace(0, 1, 11))

plt.axhline(y=0.9, color="red", label="90% variance")
plt.grid(True, alpha=0.5)
plt.legend()

plt.show()

pca_2d = PCA(n_components=2)
features_2d = pca_2d.fit_transform(values)
pca2d_df = pd.DataFrame(
    features_2d, columns=["Principal Component 1", "Principal Component 2"]
)
pca2d_df["Price"] = df.price

scatter = plt.scatter(
    x=pca2d_df["Principal Component 1"],
    y=pca2d_df["Principal Component 2"],
    c=pca2d_df["Price"],
)
plt.title("PCA with 2 principal components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
cbar = plt.colorbar(scatter)
cbar.set_label("Price")


pca_2d = PCA(n_components=3)
features_2d = pca_2d.fit_transform(values)
pca2d_df = pd.DataFrame(
    features_2d,
    columns=["Principal Component 1", "Principal Component 2", "Principal Component 3"],
)
pca2d_df["Price"] = df.price

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


scatter = ax.scatter(
    pca2d_df["Principal Component 1"],
    pca2d_df["Principal Component 2"],
    pca2d_df["Principal Component 3"],
    c=pca2d_df["Price"],
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

cbar = plt.colorbar(scatter)
cbar.set_label("Price")

plt.show()
