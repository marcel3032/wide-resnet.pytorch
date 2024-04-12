import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import prune

import sys
import numpy as np

parameters_to_prune = []

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic_l1_pruned_global(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic_l1_pruned_global, self).__init__()
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
    
    def get_prunable_params(self):
        return [
          (self.conv1, 'weight'),
          (self.conv2, 'weight')
        ]

class Wide_ResNetL1PrunedGlobalRetrain(nn.Module):
    
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, sparsity):
        super(Wide_ResNetL1PrunedGlobalRetrain, self).__init__()
        self.in_planes = 16
        self.sparsity = sparsity

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet-L1-pruned-global %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1, prune1 = self._wide_layer(wide_basic_l1_pruned_global, nStages[1], n, dropout_rate, stride=1)
        self.layer2, prune2 = self._wide_layer(wide_basic_l1_pruned_global, nStages[2], n, dropout_rate, stride=2)
        self.layer3, prune3 = self._wide_layer(wide_basic_l1_pruned_global, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        parameters_to_prune.append((self.linear, 'weight'))

        self.apply(self.add_conv_to_parameters_to_prune)
        
        print("spravny subor")

        # self.prune()

    def prune(self):
        prune.global_unstructured(
          parameters_to_prune,
          pruning_method=prune.L1Unstructured,
          amount=self.sparsity,
        )

    @staticmethod
    def add_conv_to_parameters_to_prune(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            parameters_to_prune.append((m, 'weight'))

    def log(self, m, writer):
        pass

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        prune = []
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            prune += layers[-1].get_prunable_params()
            self.in_planes = planes

        return nn.Sequential(*layers), prune

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

    @staticmethod
    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            init.constant_(m.bias, 0)
            # parameters_to_prune.append((m, 'weight'))
        elif classname.find('BatchNorm') != -1:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def update_weights(self, w):
        pass


if __name__ == '__main__':
    net=Wide_ResNetL1PrunedGlobal(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
