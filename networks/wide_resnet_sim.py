import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import prune

import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import faiss

to_deal = []


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
    def __init__(self, in_planes, planes, dropout_rate, stride=1, name=None):
        super(wide_basic, self).__init__()
        self.name = name
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
        out = F.relu(self.bn1(x))

        inp = out
        out = self.conv1(out)
        if torch.is_grad_enabled():
            to_deal.append(that_weight_magic(inp, out, self.conv1, out.shape[0], self.name + " self.conv1"))

        out = self.dropout(out)
        out = F.relu(self.bn2(out))

        inp = out
        out = self.conv2(out)
        if torch.is_grad_enabled():
            to_deal.append(that_weight_magic(inp, out, self.conv2, out.shape[0], self.name + " self.conv2"))

        out += self.shortcut(x)

        return out


class Wide_ResNet_sim(nn.Module):
    def __init__(self, writer: SummaryWriter, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_sim, self).__init__()
        self.global_step = 0
        self.in_planes = 16
        self.conv_to_log = set()

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, name="self.layer1")
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, name="self.layer2")
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, name="self.layer3")
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        # prune.random_unstructured(self.conv1, 'weight', amount=0.8)
        prune.random_unstructured(self.linear, 'weight', amount=0.8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, name):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, name + ";stride" + str(stride)))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.conv1(x)
        if torch.is_grad_enabled():
            to_deal.append(that_weight_magic(x, out, self.conv1, batch_size, "self.conv1"))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def set_w(self, writer: SummaryWriter):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global to_deal
        with torch.no_grad():
            for conv, to_randn, to_zero, log_name in to_deal:
                # print(conv, to_randn, to_zero, log_name)
                conv.weight[to_zero] = 0
                conv.weight[to_randn] = torch.randn(conv.weight[to_randn].shape).to(device)
                self.conv_to_log.add((conv, log_name))
        self.global_step += 1
        writer.flush()
        to_deal = []


def that_weight_magic_faiss(x, out, conv, batch_size, log_name):
    K = int(conv.weight.numel() * 0.05)  # TODO constant

    unfold = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
    that_out_size = conv.in_channels * np.prod(conv.kernel_size)

    unfold = unfold.mean(axis=2).detach().cpu().numpy().T
    out = out.mean(axis=(2, 3)).detach().cpu().numpy().T

    index = faiss.IndexFlatL2(batch_size)
    index.add(unfold)

    other_dim = int(that_out_size) // 10

    D, I_rand = index.search(out, other_dim)
    to_rand_idx = np.array((np.repeat(np.arange(conv.out_channels), other_dim).reshape(conv.out_channels, other_dim),
                            *np.unravel_index(I_rand, conv.weight.shape[1:]))).reshape(len(conv.weight.shape), -1)
    conv_w = conv.weight[to_rand_idx].detach().cpu().numpy()

    to_rand = D.reshape(-1)
    to_rand[conv_w != 0] = np.inf

    D, I_zero = index.search(-1 * out, other_dim)
    to_zero_idx = np.array((np.repeat(np.arange(conv.out_channels), other_dim).reshape(conv.out_channels, other_dim),
                            *np.unravel_index(I_zero, conv.weight.shape[1:]))).reshape(len(conv.weight.shape), -1)
    conv_w = conv.weight[to_zero_idx].detach().cpu().numpy()

    to_zero = D.reshape(-1)
    to_zero[conv_w == 0] = np.inf

    to_rand_idx = to_rand_idx.T[np.argpartition(to_rand, K)[:K]]
    to_zero_idx = to_zero_idx.T[np.argpartition(to_zero, K)[:K]]

    return conv, to_rand_idx.T, to_zero_idx.T, log_name


def that_weight_magic(x, out, conv, batch_size, log_name):
    K = int(conv.weight.numel() * 0.05)  # TODO constant

    unfold = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
    that_out_size = conv.in_channels * np.prod(conv.kernel_size)

    unfold = unfold.mean(axis=2).detach().cpu().numpy().T
    out = out.mean(axis=(2, 3)).detach().cpu().numpy().T

    similarities = out.dot(unfold.T)

    nonzero_weights_tensor = (conv.weight.reshape(conv.out_channels, that_out_size) != 0).int().detach().cpu().numpy()

    max_similarities = similarities * (1 - nonzero_weights_tensor)
    to_randn = get_index_list(get_indexes_of_k_smallest(max_similarities, -K), conv)

    min_similarities = similarities * nonzero_weights_tensor
    to_zero = get_index_list(get_indexes_of_k_smallest(min_similarities, K), conv)

    return conv, to_randn, to_zero, log_name


def get_index_list(indices, conv):
    return np.unravel_index(indices, conv.weight.shape)


def get_indexes_of_k_smallest(arr, k):
    return np.argpartition(arr.reshape(-1), k)


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    net = Wide_ResNet_sim(writer, 28, 10, 0.3, 10)
    net.apply(conv_init)
    y = net(Variable(torch.randn(128, 3, 32, 32)))

    print(y.size())
