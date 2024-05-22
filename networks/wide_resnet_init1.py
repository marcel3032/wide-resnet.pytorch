import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

K = 5


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def weights_init(m):
    with torch.no_grad():
        if type(m) == nn.Linear:
            h = m.weight.shape[0]
            w = m.weight.shape[1]
            weights = np.random.randn(h, K).dot(np.random.randn(K, w)) + np.random.randn(h, w) * (
                        np.random.rand(h, w) > 0.8).astype(int)
            weights -= np.mean(weights)
            weights /= np.std(weights)
            print(np.mean(weights), np.std(weights))
            m.weight = nn.Parameter(torch.Tensor(weights))


class wide_basic_init1(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic_init1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1a = nn.Conv2d(in_planes, K, kernel_size=3, padding=1, bias=True)
        self.conv1b = nn.Conv2d(K, planes, kernel_size=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2a = nn.Conv2d(planes, K, kernel_size=3, padding=1, bias=True)
        self.conv2b = nn.Conv2d(K, planes, kernel_size=1, stride=stride, padding=0, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, K, kernel_size=1, padding=0, bias=True),
                nn.Conv2d(K, planes, kernel_size=1, stride=stride, padding=0, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1b(self.conv1a(F.relu(self.bn1(x)))))
        out = self.conv2b(self.conv2a(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out


class Wide_ResNet_init1(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_init1, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1a = conv3x3(3, K, kernel_size=3)
        self.conv1b = conv3x3(K, nStages[0], kernel_size=1)
        self.layer1 = self._wide_layer(wide_basic_init1, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic_init1, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic_init1, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        # plot_hist(self.linear.weight.detach().numpy(), plot_zeros=True)

        # self.linear.weight.data = torch.Tensor(np.random.randn(self.linear.in_features, 4).dot(np.random.randn(4, self.linear.out_features)))

        # plot_hist(torch.masked_select(self.linear.weight, self.linear.weight != 0).detach().numpy(), plot_zeros=True)

        self.apply(weights_init)

        # exit(0)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1a(x)
        out = self.conv1b(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    net = Wide_ResNet_init1(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())
