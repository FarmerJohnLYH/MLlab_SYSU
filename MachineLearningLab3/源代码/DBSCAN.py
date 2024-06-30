import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# Create dataset
blob_centers = np.array(
    [[0.6, 2.7],
     [-1.3, 2.5],
     [-2.6, 2.0],
     [-2.8, 2.8],
     [-2.4, 0.6],
     [-1, 1.4],
     [0.2, 1]])
blob_std = np.array([0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.3])

X, y = make_blobs(n_samples=200, centers=blob_centers, cluster_std=blob_std, random_state=7)

# Set radius to 0.6 and minimum sample size to 4
y_pred = DBSCAN(eps=0.6, min_samples=4).fit_predict(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot clustered data
ax1 = ax[0]
ax1.scatter(x=X[:, 0], y=X[:, 1], s=250, c=y_pred)
ax1.set_title('DBSCAN Clustering Result', fontsize=12)

# Plot true data labels
ax2 = ax[1]
ax2.scatter(x=X[:, 0], y=X[:, 1], s=250, c=y)
ax2.set_title('True Labels', fontsize=12)
plt.savefig("DBSCAN.png",dpi = 1000)
# plt.show()

# Calculate silhouette score
score = silhouette_score(X, y_pred)
print('Silhouette Score: {}'.format(score))  # Random comment
