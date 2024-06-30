import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import sys
import logging
learning_rate = 0.1
# 创建一个logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
import time
timecode = time.strftime("%m-%d-%H-%M", time.localtime())
# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(message)s')
timecode = "./results/" + timecode
fh = logging.FileHandler( timecode + '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)
sys.path.append("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability    
def load_process_data(batch_size, num_workers):
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())
    logger.info(f'Train size: {len(mnist_train)}, Test size: {len(mnist_test)}')

    train_dataloader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    test_dataloader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    #minibatch -> batch_size = 128
    return train_dataloader, test_dataloader

class SoftmaxRegression(nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(28*28, 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1) # [128, 28*28]
        x = self.linear(x)
        # x = self.softmax(x)
        return x

def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs, patience):
    model.to(device)  # Move model to GPU if available
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_test_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    logger.info('Start training...')
    logger.info(f'Using device {device}')
    logger.info(f'Number of epochs: {num_epochs}, Patience: {patience} , Learning rate: {learning_rate}')
    logger.info(model)
    logger.info(optimizer)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_train_loss)
        avg_test_loss, accuracy = evaluate(test_dataloader, model, loss_fn)
        test_loss.append(avg_test_loss)
        test_accuracy.append(accuracy)
        logger.info(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
    model.load_state_dict(best_model)
    return train_loss, test_loss, test_accuracy

def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
import numpy as np
def visualize(train_loss, test_loss, test_accuracy, num_epochs):
    ired = np.array([219, 49, 36])/256  # 红色
    iyel = np.array([255,223,146])/256 # 奶黄色
    iblue = np.array([144,190,224])/256 # 淡蓝色
    idarkblue = np.array([75,116,178])/256 # 蓝色
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    lens = min(len(train_loss), len(test_loss))
    plt.plot(range(1, lens+1), train_loss, label='Train Loss', color=iyel)
    plt.plot(range(1, lens+1), test_loss, label='Test Loss', color=iblue)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.ylim(0.2, 0.8)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, lens+1), test_accuracy, label='Test Accuracy', color=ired)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.ylim(0.7, 0.9)
    plt.savefig(timecode+'.svg', dpi=600, bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    batch_size = 4096
    num_workers = 4
    train_dataloader, test_dataloader = load_process_data(batch_size, num_workers)
    model = SoftmaxRegression()
    loss_function = nn.CrossEntropyLoss() # Softmax + CrossEntropy
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 1500
    patience = 200
    train_loss, test_loss, test_accuracy = train_and_test(train_dataloader, test_dataloader, model, loss_function, optimizer, num_epochs, patience)
    visualize(train_loss, test_loss, test_accuracy, num_epochs)