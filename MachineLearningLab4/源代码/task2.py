# 数据集Folds5x2_pp.csv共有9568个样本数据，每个数据有5列，分别是:AT（温度）, V（压⼒）, AP（湿度）, RH（压
# 强）, PE（输出电⼒）。请以AT、V、AP、RH这4列作为样本特征，PE作为样本输出标签，将样本数据按3:1随机划分成训
# 练集和测试集，分别使⽤最⼩⼆乘法和梯度下降法进⾏线性回归求解，求出线性回归系数，并可视化结果。
# 数据形如：
# AT,V,AP,RH,PE
# 8.34,40.77,1010.84,90.01,480.48
# 23.64,58.49,1011.4,74.2,445.75
# 29.74,56.9,1007.15,41.91,438.76
# 19.07,49.69,1007.22,76.79,453.09
# 11.8,40.66,1017.13,97.2,464.43
# 13.97,39.16,1016.05,84.6,470.96
# ......

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('Folds5x2_pp.csv')

# 提取特征和标签
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

# 归一化
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 43) 

# 最小二乘法线性回归
model1 = LinearRegression()
import matplotlib.pyplot as plt

model1.fit(X_train, y_train)
print('最小二乘法线性回归系数:', model1.coef_)

# 梯度下降法线性回归
from sklearn.linear_model import SGDRegressor
model2 = SGDRegressor(max_iter=1000, tol=1e-3)
model2.fit(X_train, y_train)
print('梯度下降法线性回归系数:', model2.coef_)
import numpy as np

ired = np.array([219, 49, 36])/256  # 红色
iyel = np.array([255,223,146])/256 # 奶黄色
iblue = np.array([144,190,224])/256 # 淡蓝色

# 可视化结果
plt.figure(figsize=(8, 8))
plt.scatter(model1.predict(X_test), y_test, color=iblue) #使得横坐标为样本序号
# 绘制y=x的黑色虚线 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, color=ired)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend()
plt.title('Least Squares Linear Regression')
plt.savefig("task2_1.png", dpi=600)
plt.clf()

# 可视化结果
plt.figure(figsize=(8, 8))
plt.scatter(model2.predict(X_test), y_test, color=iblue) #使得横坐标为样本序号
# 绘制y=x的黑色虚线 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, color=ired)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend()
plt.title('Gradient Descent Linear Regression')
plt.savefig("task2_2.png", dpi=600)


# 最小二乘法线性回归系数: [-0.86330901 -0.17472427  0.01888633 -0.13320319]
# 梯度下降法线性回归系数: [-0.86047673 -0.18260796  0.02182599 -0.13348844]

from sklearn.metrics import mean_absolute_error, mean_squared_error
print('\n\n')
# 最小二乘法线性回归
print('最小二乘法线性回归系数:', model1.coef_)
print('MAE:', mean_absolute_error(y_test, model1.predict(X_test)))
print('MSE:', mean_squared_error(y_test, model1.predict(X_test)))
print('Score:', model1.score(X_test, y_test))

# 梯度下降法线性回归
print('梯度下降法线性回归系数:', model2.coef_)
print('MAE:', mean_absolute_error(y_test, model2.predict(X_test)))
print('MSE:', mean_squared_error(y_test, model2.predict(X_test)))
print('Score:', model2.score(X_test, y_test))

# 输出代码所用环境 
import sys
print("python",sys.version)
# print(sys.version_info)
# print(sys.path)
# 输出所用的 numpy，plotly，pandas，sklearn，matplotlib，torch，scipy，scikit-learn 的版本
import numpy
print("numpy==",numpy.__version__)
import pandas
print("pandas==",pandas.__version__)
import sklearn
print("sklearn==",sklearn.__version__)
import matplotlib
print("matplotlib==",matplotlib.__version__)
import torch
print("torch==",torch.__version__)
import scipy
print("scipy==",scipy.__version__)
