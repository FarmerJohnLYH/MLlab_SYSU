import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果的可重复性
np.random.seed(42)

# 模拟数据集
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加x0 = 1 到所有实例
X_b = np.c_[np.ones((100, 1)), X]

# MiniBatch 梯度下降参数
n_iterations = 50
minibatch_size = 16
n_epochs = 50
t0, t1 = 200, 1000  # 学习计划的超参数

def learning_schedule(t):
    return t0 / (t + t1)

theta_path_mgd = []
m = len(X_b)

theta = np.random.randn(2,1)  # 随机初始化参数

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t = epoch * m + i
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# 将路径转换为数组以便绘图
theta_path_mgd = np.array(theta_path_mgd)

# 绘制θ路径
plt.figure(figsize=(7,7))
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="MiniBatch")
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.legend(loc="upper right", fontsize=14)
plt.title("MiniBatch Gradient Descent Path")
plt.savefig("taskA.png")
plt.show()
#batch数量如何选择？