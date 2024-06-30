import numpy as np
import matplotlib.pyplot as plt

random_seed = 42
np.random.seed(random_seed)

m = 100
X = 6*np.random.rand(m,1) - 3
X = np.sort(X, axis=0)
y = 0.5*X**2+X+np.random.randn(m,1)
print(X)
print(y)
# exit()
# 定义多项式度数
degrees = [1, 2, 100]
cnt = 0
col = ['r','g','b']


fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for degree, ax in zip(degrees, axs):    # 生成多项式特征
    poly_features = np.vander(np.ravel(X), degree + 1, increasing=True)
    # 使用正规方程求解参数theta
    theta_best = np.linalg.inv(poly_features.T @ poly_features) @ poly_features.T @ y
    # 预测值
    y_pred = poly_features @ theta_best
    # 绘图
    
    ax.scatter(X, y, label="Data",color = 'g')

    ax.plot(X, y_pred, label=f"Degree {degree}",color = 'b')

    ax.set_title(f"Polynomial Regression (Degree={degree})")

    ax.legend()
    #设置纵轴范围到 y_min~y_max 
    ax.axis([X.min(), X.max(), y.min(), y.max()])
plt.tight_layout()
plt.legend()
plt.savefig("task2.png")
plt.show()

