# !/usr/bin/python3

# coding:utf8
# @Author: Jlan
# @Time: 18-6-11 下午11:05

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)  # 100行2列 每个值都为1
# print(n_data)
x0 = torch.normal(2*n_data, 1)  # 数据 包含横纵坐标
print(x0)
y0 = torch.zeros(100)  # class0 的数据标签
x1 = torch.normal(-2*n_data, 1)  # 数据
y1 = torch.ones(100)  # class1 的数据标签

x = torch.cat((x0, x1), 0).type(torch.FloatTensor) #cat的作用是将矩阵进行合并 0，1代表按行还是列
y = torch.cat((y0, y1), ).type(torch.LongTensor) #Long 64bit； Float  32bit

x, y = Variable(x), Variable(y)
#可视化
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(),
            s=100, lw=0, cmap='RdYlGn')
plt.show()

#创建NN
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        #创建神经网络单元
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
print(net)

#优化
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
criterion = torch.nn.CrossEntropyLoss() #分类用损失函数 softmax [0.1 0.2 0.7]=1

plt.ion()   # 画图
plt.show()

for t in range(100):
    out = net(x)
    loss = criterion(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        print(prediction)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'epoch={}, accuracy={:.4f}'.format(t, accuracy), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()