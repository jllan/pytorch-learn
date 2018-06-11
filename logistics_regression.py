# !/usr/bin/python3

# coding:utf8
# @Author: Jlan
# @Time: 18-6-11 下午11:40

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

n_data = torch.ones(100, 2)  # 100行2列 每个值都为1
# print(n_data)
x0 = torch.normal(2*n_data, 1)  # 数据 包含横纵坐标
# print(x0)
y0 = torch.zeros(100)  # class0 的数据标签
x1 = torch.normal(-2*n_data, 1)  # 数据
y1 = torch.ones(100)  # class1 的数据标签

x = torch.cat((x0, x1), 0).type(torch.FloatTensor) #cat的作用是将矩阵进行合并 0，1代表按行还是列
y = torch.cat((y0, y1), ).type(torch.FloatTensor) #Long 64bit； Float  32bit

x, y = Variable(x), Variable(y)

#创建NN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #创建神经网络单元
        self.lr = torch.nn.Linear(2, 1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


net = Net()
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.BCELoss()

plt.ion()   # 画图
plt.show()

for t in range(100):
    out = net(x)
    loss = criterion(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        print_loss = loss.data[0]
        mask = out.ge(0.5).float()
        correct = (mask == y).sum()
        acc = correct.data[0] / x.size(0)

        plt.cla()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(),
                    s=100, lw=0, cmap='RdYlGn')
        w0, w1 = net.lr.weight[0]
        w0 = w0.data[0]
        w1 = w1.data[0]
        b = net.lr.bias.data[0]
        plot_x = np.arange(-4, 4, 0.1)
        plot_y = (-w0*plot_x-b)/w1
        plt.plot(plot_x, plot_y)
        plt.text(1.5, -4, 'epoch={}, acc={:.4f}, loss={:.4f}'.format(t, acc, print_loss),
                 fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()

