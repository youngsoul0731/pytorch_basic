import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 创建数据
x0 = torch.normal(2*torch.ones(100, 2), 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*torch.ones(100, 2), 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)


class Net(nn.Module):

    def __init__(self,n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(2, 10, 2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for i in range(500):
    optimizer.zero_grad()
    out = net(x)
    loss = loss_func(out,y)
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.squeeze()
        target_y = y.data
        plt.scatter(x.data[:, 0], x.data[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = torch.sum(torch.tensor(pred_y == target_y)) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
