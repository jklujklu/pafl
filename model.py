# -*- coding: utf-8 -*-
# @Time : 2022/2/28 12:40
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : models.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,30,30) -> (32,28,28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        # (32,28,28) -> (32,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (32,14,14) -> (64,12,12)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        # (64,12,12) -> (64,6,6)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (64,6,6) -> (64,4,4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 43)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 30, 30)
        tensor = F.relu(self.conv1(tensor))
        # print(tensor.shape)
        tensor = self.pool1(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        # print(tensor.shape)
        tensor = F.relu(self.conv2(tensor))
        # print(tensor.shape)
        tensor = self.pool2(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        tensor = F.relu(self.conv3(tensor))
        # tensor = self.fl(tensor
        #                  )
        # print(tensor.shape)
        tensor = tensor.view(-1, 4 * 4 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    net = CNN()
    inputs = torch.randn(8, 3, 30, 30)
    net(inputs)
    print('aaa')
