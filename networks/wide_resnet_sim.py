import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import prune

import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    # (F.unfold(x, kernel_size=3, padding=1, stride=1).transpose(1, 2).matmul(self.conv1.weight.view(self.conv1.weight.size(0), -1).t()) + self.conv1.bias).transpose(1, 2).view(1,16,32,32)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        prune.random_unstructured(self.conv1, 'weight', amount=0.8)
        prune.random_unstructured(self.conv2, 'weight', amount=0.8)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_sim(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_sim, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        prune.random_unstructured(self.conv1, 'weight', amount=0.8)
        prune.random_unstructured(self.linear, 'weight', amount=0.8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        unfold = F.unfold(x, kernel_size=3, padding=1, stride=1)

        similarities = torch.einsum('bkd,bcd->kdc', out.reshape(128, 16, -1),
                                    unfold.reshape(128, 27, -1)).detach().numpy()

        K = int(similarities.size * 0.05)

        max_similarities = torch.einsum('abc,ac->abc', torch.Tensor(similarities), torch.Tensor(
            1 - (self.conv1.weight.reshape(16, 27).detach().numpy() != 0).astype("int"))).detach().numpy()
        max_partition = get_indexes_of_k_smallest(max_similarities, -K)
        a_list, b_list, c_list = np.unravel_index(max_partition, similarities.shape)
        C_list = np.unravel_index(c_list, self.conv1.weight[0].shape)
        self.to_randn = (a_list, *C_list)

        min_similarities = torch.einsum('abc,ac->abc', torch.Tensor(similarities), torch.Tensor(
            (self.conv1.weight.reshape(16, 27).detach().numpy() != 0).astype("int"))).detach().numpy()
        min_partition = get_indexes_of_k_smallest(min_similarities, K)
        a_list, b_list, c_list = np.unravel_index(min_partition, similarities.shape)
        C_list = np.unravel_index(c_list, self.conv1.weight[0].shape)
        self.to_zero = (a_list, *C_list)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        # print(len(out))

        return out

    def set_w(self):
        with torch.no_grad():
            self.conv1.weight[self.to_zero] = 0
            self.conv1.weight[self.to_randn] = torch.randn(self.conv1.weight[self.to_randn].shape)


def get_indexes_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return idx[range(min(k, 0), max(k, 0))].T


if __name__ == '__main__':
    net = Wide_ResNet_sim(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(128, 3, 32, 32)))

    print(y.size())
