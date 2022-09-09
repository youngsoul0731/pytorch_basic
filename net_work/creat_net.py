import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x)
y = x ** 2 + 0.2 * torch.rand(x.size())


class Net(nn.Module):

    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_features=1,n_hidden = 10,n_output = 1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
for i in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data, y.data)
        plt.plot(x.data, prediction.data, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    if loss <= 0.0050:
        break
plt.show()
