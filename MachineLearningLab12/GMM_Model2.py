# 导⼊必要的库
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 忽略警告
import warnings

warnings.filterwarnings("ignore")

# 设置随机种⼦以确保结果可重复
np.random.seed(42)

# 1 下载LFW⼈脸数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data  # 提取图像数据
n_samples, h, w = lfw_people.images.shape  # 获取图像的尺⼨

# 输出数据形状
print("Number of samples:", n_samples)
print("Image height:", h)
print("Image width:", w)

# 2 数据标准化处理
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# 输出标准化后的数据形状
print("Normalized lfw_home shape:", X_normalized.shape)

# 3 特征降维
pca = PCA(n_components=150)
train_data = pca.fit_transform(X_normalized)

# 输出降维后的数据形状
print("Dimensionality reduction lfw_home shape:", train_data.shape)

# 4 ⾼斯混合模型初始化和训练
GMM_model = GaussianMixture(n_components=7)
GMM_model.fit(train_data)

# 输出⾼斯分布的均值、协⽅差矩阵和混合系数
print("GMM Model mean:\n{}".format(GMM_model.means_))
print("GMM Model covariance:\n{}".format(GMM_model.covariances_))
print("GMM Model weight:\n{}".format(GMM_model.weights_))


# 绘制面部图像的网格
def show_reconstructed_images(model_means):
    # 逆变换GMM的均值回到原始特征空间
    original_dim_centers = pca.inverse_transform(model_means)

    # 逆标准化
    original_dim_centers_unscaled = scaler.inverse_transform(original_dim_centers)

    # 重构图像
    images_reconstructed = original_dim_centers_unscaled.reshape(-1, h, w)

    # 可视化聚类中心代表的人脸图像
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))  # 7个聚类，展示一个2x4的网格，最后一个位置留空
    for i, (image, ax) in enumerate(zip(images_reconstructed, axes.flatten())):
        if i < 7:  # 确保索引不会超出范围
            ax.imshow(image, cmap='gray')  # 使用灰度图显示
            ax.axis('off')  # 不显示坐标轴
            ax.set_title(f'Cluster {i}')

    fig.delaxes(axes[1][3])  # 手动删除第(2,4)位置的子图
    plt.tight_layout()
    plt.show()


# 5 结果可视化
show_reconstructed_images(GMM_model.means_)
