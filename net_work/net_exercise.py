import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# artificial data
x = torch.linspace(-1, 1, 100).view(-1, 1)
y = x**3 + 2*x**2 + 4*x + torch.rand(x.size())
# 定义神经网络


class Net(nn.Module):

    def __init__(self,n_features,n_hidden1,n_hidden2,n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_features,n_hidden1)  # 第一层隐藏层 z1 = w1*x
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)   # 第二层隐藏层
        self.predict = nn.Linear(n_hidden2,n_output)    # 第二层隐藏层到输出层

    def forward(self, x):   # 向前传递
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


# 将定义的神经网络实例化
net = Net(n_features=1, n_hidden1=10, n_hidden2=10, n_output=1)
# 设置优化器，采用随机梯度下降SGD，传入我们神经网络模型的参数，设置学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = nn.MSELoss()
for i in range(500):
    optimizer.zero_grad()  # 将之前储存的梯度清零
    prediction = net(x)
    loss = loss_func(prediction,y)
    loss.backward()  # 误差反向传播
    optimizer.step()    # 更新权重
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data, y.data)
        plt.plot(x.data, prediction.data, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    if loss <= 0.0050:
        break
plt.show()

