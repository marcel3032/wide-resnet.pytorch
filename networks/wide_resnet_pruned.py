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


class wide_basic_pruned(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic_pruned, self).__init__()
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

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNetPruned(nn.Module):
    sparsity = 0
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, sparsity):
        super(Wide_ResNetPruned, self).__init__()
        self.in_planes = 16
        self.sparsity = sparsity

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet pruned %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic_pruned, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic_pruned, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic_pruned, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        prune.random_unstructured(self.linear, 'weight', amount=0.8)

    def log(self, m, writer):
        pass

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def update_weights(self, writer):
        pass

    def log(self, m, writer):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            import time
            writer.add_scalar(classname, torch.count_nonzero(m.weight) / torch.numel(m.weight), int(time.time()*1000))
            writer.add_scalar(classname + " mask", torch.count_nonzero(m.weight_mask) / torch.numel(m.weight_mask), int(time.time()*1000))


    @staticmethod
    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            init.constant_(m.bias, 0)
            prune.random_unstructured(m, 'weight', amount=Wide_ResNetPruned.sparsity)
        elif classname.find('BatchNorm') != -1:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

if __name__ == '__main__':
    net=Wide_ResNetPruned(28, 10, 0.3, 10)
    net.apply(net.conv_init)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
