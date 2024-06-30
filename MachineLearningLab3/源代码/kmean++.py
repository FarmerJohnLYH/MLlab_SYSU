import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_excel('city.xls', header=None)
longitude = data.iloc[1:3180, 3]
latitude = data.iloc[1:3180, 4]
samples = np.column_stack((longitude, latitude))
print("samples=",samples)
k = 3


# 使用KMeans类进行聚类，其中init参数为"k-means++"是默认设置
model = KMeans(n_clusters=k, random_state=0)  # 确保结果可复现性
model.fit(samples)

# 获取聚类标签和聚类中心
sampleTag = model.labels_
clusterCents = model.cluster_centers_
SSE = model.inertia_ # 获取SSE

# 可视化聚类结果
ired = np.array([219, 49, 36])/256  # 红色
plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag)
plt.scatter(clusterCents[:, 0], clusterCents[:, 1], marker='o', s=150, c=ired)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'K-Means++ Clustering Result (k={k}, SSE={SSE:.2f})')
plt.savefig('kmeans++_k=3_Result.png', dpi=800, bbox_inches='tight')
# plt.show()
