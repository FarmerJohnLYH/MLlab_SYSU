import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义欧氏距离计算函数
def L2(vecXi, vecXj):
    return np.linalg.norm(vecXi - vecXj)

# K-means算法实现
def kMeans(S, k, distMeas=L2):
    # 随机选择初始聚类中心
    clusterCents = S[np.random.choice(np.arange(len(S)), size=k, replace=False)]
    
    # 初始化样本标签和误差平方和
    sampleTag = np.zeros(len(S))
    SSE = float('inf')
    
    while True:
        # 更新样本所属的簇标签
        for i, sample in enumerate(S):
            min_dist_index = np.argmin([distMeas(sample, cent) for cent in clusterCents])
            sampleTag[i] = min_dist_index
        
        # 计算新的聚类中心
        new_clusterCents = []
        for i in range(k):
            # 获取属于第i个簇的所有样本
            cluster_samples = S[sampleTag == i]
            if len(cluster_samples) > 0:
                # 新聚类中心为该簇所有样本坐标的平均值
                new_clusterCents.append(np.mean(cluster_samples, axis=0))
            else:
                # 若某个簇没有样本，则保持原聚类中心不变（这种情况一般不会出现）
                new_clusterCents.append(clusterCents[i])
        clusterCents = np.array(new_clusterCents)
        
        # 计算新的误差平方和
        oldSSE = SSE
        # 小心非整数sampleTag[i]
        SSE = sum([distMeas(sample, clusterCents[int(sampleTag[i])]) ** 2 for i, sample in enumerate(S)])
        # 如果聚类中心不再改变或达到最大迭代次数，则结束循环
        # 在这里，我们仅使用了简单停止条件（连续两次SSE相同），实际应用中可能需要设定迭代次数限制
        if SSE == oldSSE:
            break

    return sampleTag, clusterCents, SSE

# 示例数据
# samples = np.array([
#     [116.4, 39.9],
#     [116.38, 39.9],
#     [116.42, 39.93],
#     [121.15, 22.75],
#     [121.6, 23.98],
#     [119.58, 23.58],
#     # ... 更多城市经纬度数据
# ])

# 设置K值
k = 3

# 运行K-means算法
# sampleTag, clusterCents, SSE = kMeans(samples, k)

# 加载城市经纬度数据
data = pd.read_excel('city.xls', header=None)
longitude = data.iloc[1:3180, 3]
latitude = data.iloc[1:3180, 4]
samples = np.column_stack((longitude, latitude))
print("samples=",samples)
k = 20
# exit()

# 运行K-means算法
sampleTag, clusterCents, SSE = kMeans(samples, k)

# 可视化聚类结果
ired = np.array([219, 49, 36])/256  # 红色
plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag)
plt.scatter(clusterCents[:, 0], clusterCents[:, 1], marker='o', s=150, c=ired)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'K-Means Clustering Result (k={k}, SSE={SSE:.2f})')
plt.savefig('k=20_Result.png', dpi=800, bbox_inches='tight')
plt.show()