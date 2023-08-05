import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import prune

import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    # (F.unfold(x, kernel_size=3, padding=1, stride=1).transpose(1, 2).matmul(self.conv1.weight.view(self.conv1.weight.size(0), -1).t()) + self.conv1.bias).transpose(1, 2).view(1,16,32,32)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
        prune.random_unstructured(m, 'weight', amount=0.8)
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

        # prune.random_unstructured(self.conv1, 'weight', amount=0.8)
        # prune.random_unstructured(self.conv2, 'weight', amount=0.8)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_sim(nn.Module):
    def __init__(self, writer: SummaryWriter, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_sim, self).__init__()
        self.writer = writer
        self.global_step = 0
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

        # prune.random_unstructured(self.conv1, 'weight', amount=0.8)
        prune.random_unstructured(self.linear, 'weight', amount=0.8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.conv1(x)
        self.that_weight_magic(x, out, self.conv1, batch_size, "self.conv1")
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def set_w(self):
        with torch.no_grad():
            conv, to_randn, to_zero, log_name = self.to_deal
            conv.weight[to_zero] = 0
            conv.weight[to_randn] = torch.randn(conv.weight[to_randn].shape)
            self.writer.add_histogram(log_name, conv.weight, self.global_step)
            self.writer.add_scalar(log_name+" nonzero weights", np.count_nonzero(conv.weight), self.global_step)
            self.writer.add_scalar(log_name+" weight count", conv.weight.numel(), self.global_step)
            self.writer.add_scalar(log_name+" sparsity", 100*(1-np.count_nonzero(conv.weight)/conv.weight.numel()), self.global_step)
        self.global_step += 1
        self.writer.flush()



    def that_weight_magic(self, x, out, conv, batch_size, log_name):
        unfold = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
        that_out_size = conv.in_channels * np.prod(conv.kernel_size)

        similarities = torch.einsum('bkd,bcd->kdc', out.reshape(batch_size, conv.out_channels, -1),
                                    unfold.reshape(batch_size, that_out_size, -1))

        K = int(conv.weight.numel() * 0.05)  # TODO constant

        nonzero_weights_tensor = (conv.weight.reshape(conv.out_channels, that_out_size) != 0).int()

        max_similarities = torch.einsum('abc,ac->abc', similarities, 1 - nonzero_weights_tensor).detach().numpy().max(
            axis=1)
        to_randn = get_index_list(get_indexes_of_k_smallest(max_similarities, -K), max_similarities, conv)

        min_similarities = torch.einsum('abc,ac->abc', similarities, nonzero_weights_tensor).detach().numpy().min(
            axis=1)
        to_zero = get_index_list(get_indexes_of_k_smallest(min_similarities, K), min_similarities, conv)

        self.to_deal = conv, to_randn, to_zero, log_name


def get_index_list(partition, similarities, conv):
    a_list, c_list = np.unravel_index(partition, similarities.shape)
    C_list = np.unravel_index(c_list, conv.weight[0].shape)
    return a_list, *C_list


def get_indexes_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return idx[range(min(k, 0), max(k, 0))].T


if __name__ == '__main__':
    net = Wide_ResNet_sim(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(128, 3, 32, 32)))

    print(y.size())
