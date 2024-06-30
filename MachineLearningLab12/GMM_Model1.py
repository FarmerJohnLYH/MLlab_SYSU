# 导⼊必要的库
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

# 忽略警告
import warnings

warnings.filterwarnings("ignore")

# 设置随机种⼦以确保结果可重复
np.random.seed(42)

# ⽣成第⼀个⾼斯分布的数据
mean1 = [0, 0, 0, 0, 0]  # 均值向量
cov1 = np.eye(5)  # 协⽅差矩阵（单位矩阵）
data1 = np.random.multivariate_normal(mean1, cov1, 100)  # ⽣成100个样本

# ⽣成第⼆个⾼斯分布的数据
mean2 = [5, 5, 5, 5, 5]
cov2 = np.eye(5)
data2 = np.random.multivariate_normal(mean2, cov2, 100)

# ⽣成第三个⾼斯分布的数据
mean3 = [0, 5, 0, 5, 0]
cov3 = np.eye(5)
data3 = np.random.multivariate_normal(mean3, cov3, 100)
# ⽣成第四个⾼斯分布的数据

mean4 = [5, 0, 5, 0, 5]
cov4 = np.eye(5)
data4 = np.random.multivariate_normal(mean4, cov4, 100)

# ⽣成第五个⾼斯分布的数据
mean5 = [2.5, 2.5, 2.5, 2.5, 2.5]
cov5 = np.eye(5)
data5 = np.random.multivariate_normal(mean5, cov5, 100)

# 合并所有⽣成的数据
data = np.vstack((data1, data2, data3, data4, data5))
np.random.shuffle(data)  # 打乱数据顺序

# 输出⽣成的数据形状
print("Data shape:", data.shape)

# 2 数据标准化处理
scaler = StandardScaler()
X_normalized = scaler.fit_transform(data)
print("Input Data mean: {}".format(scaler.mean_))
print("Input Data var: {}".format(np.sqrt(scaler.var_)))

# 3 数据降维
pca = PCA(n_components=2)
train_data = pca.fit_transform(X_normalized)

# 输出降维后的数据形状
print("Dimensionality reduction lfw_home shape:", train_data.shape)
# 可视化降维后的数据
plt.scatter(train_data[:, 0], train_data[:, 1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM train data")
plt.show()

# 4 ⾼斯混合模型初始化和训练
GMM_model = GaussianMixture(n_components=5)
GMM_model.fit(train_data)
y_pred = GMM_model.predict(train_data)
y_prob = GMM_model.predict_proba(train_data)

# 输出⾼斯分布的均值、协⽅差矩阵和混合系数
print("GMM Model mean:\n{}".format(GMM_model.means_))
print("GMM Model covariance:\n{}".format(GMM_model.covariances_))
print("GMM Model weight:\n{}".format(GMM_model.weights_))


# 可视化展示聚类结果
def plot_result(GMM_model, train_data):
    fig, ax = plt.subplots()
    # 绘制数据点
    plt.scatter(train_data[:, 0], train_data[:, 1], s=5)

    # 定义椭圆颜色列表
    color_name = 'colorlist'
    color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    mycolor = LinearSegmentedColormap.from_list(color_name, color_list, N=5)

    # 绘制椭圆
    for i, params in enumerate(zip(GMM_model.means_, GMM_model.covariances_, GMM_model.weights_)):
        pos, cov, w = params
        u, s, vt = np.linalg.svd(cov)
        angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
        width, height = 3 * np.sqrt(s)
        ax.add_patch(Ellipse(pos, width, height, angle, alpha=w, color=mycolor(i / 5)))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("GMM Clustering Results (2D Projection)")

    plt.show()


# 5 结果可视化
plot_result(GMM_model, train_data)
