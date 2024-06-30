
import torch
import numpy as np
from torch.autograd import Variable
torch.manual_seed(43)
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
[9.779], [6.182], [7.59], [2.167], [7.042],
[10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
[3.366], [2.596], [2.53], [1.221], [2.827],
[3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 画出图像
import matplotlib.pyplot as plt

# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True) # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True) # 使用  0 进行初始化

# 构建一元线性回归模型
def linear_model(x,w,b):
    return x * w + b
# 输入数据转换成 Tensor 再转换为Variable
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = Variable(x_train)
y_train = Variable(y_train)

#误差函数
def get_loss(y_, y_train):
    return torch.mean((y_ - y_train) ** 2)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# 定义学习率列表
learning_rates = [0.001, 0.01, 0.1, 1]

for a in learning_rates:
    losses = []
    y_preds = []
    w = Variable(torch.randn(1), requires_grad=True) # 随机初始化
    b = Variable(torch.zeros(1), requires_grad=True) # 使用  0 进行初始化

    for e in range(200):
        y_ = linear_model(x_train,w,b)
        # y_ = x_train * w + b
        loss = get_loss(y_, y_train)
        loss.backward()
        w.data = w.data - a * w.grad.data # 更新 w
        b.data = b.data - a * b.grad.data # 更新 b
        w.grad.zero_() # 记得归零梯度
        b.grad.zero_() # 记得归零梯度
        losses.append(loss.item())
        y_preds.append(y_.detach().numpy())


    # 绘制损失函数图
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss with learning rate {}'.format(a))

    # 绘制预测效果图
    plt.subplot(1, 2, 2)
    plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
    plt.plot(x_train.data.numpy(), y_preds[-1], 'ro', label='estimated')
    plt.title('Prediction with learning rate {}'.format(a))
    plt.legend()
    # plt.show()
    plt.savefig('result_lr_{}.png'.format(a),dpi = 1000)

    # 计算均方误差
    mse = mean_squared_error(y_train.data.numpy(), y_preds[-1])
    print('MSE with learning rate {}: {}'.format(a, mse))



