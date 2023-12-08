import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import prune
import config as cf

import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import faiss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def _backward_hook(module, grad_output):
    with torch.no_grad():
        if not hasattr(module, "grad_output"):
            module.grad_output = torch.zeros([cf.batch_size] + list(grad_output[0].shape)[1:], device=device)
        if module.grad_output.shape[0] == grad_output[0].shape[0]:
            module.grad_output += grad_output[0]


def _forward_hook(module, input, output):
    with torch.no_grad():
        if not hasattr(module, "input"):
            module.input = torch.zeros([cf.batch_size] + list(input[0].shape)[1:], device=device)
        if module.input.shape[0] == input[0].shape[0]:
            module.input += input[0]


class wide_basic_correct(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, K, k_similar, name=None):
        super(wide_basic_correct, self).__init__()
        self.name = name
        self.K = K
        self.k_similar = k_similar
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
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        return out + self.shortcut(x)


class Wide_ResNet_sim_correct(nn.Module):
    def __init__(self, writer: SummaryWriter, depth, widen_factor, dropout_rate, num_classes, K, k_similar, update_method, bits: int):
        super(Wide_ResNet_sim_correct, self).__init__()
        self.global_step = 0
        self.in_planes = 16
        self.conv_to_log = set()
        self.update_method = update_method
        self.bits = bits

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic_correct, nStages[1], n, dropout_rate, 1, K, k_similar, name="self.layer1")
        self.layer2 = self._wide_layer(wide_basic_correct, nStages[2], n, dropout_rate, 2, K, k_similar, name="self.layer2")
        self.layer3 = self._wide_layer(wide_basic_correct, nStages[3], n, dropout_rate, 2, K, k_similar, name="self.layer3")
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.K = K
        self.k_similar = k_similar

        prune.random_unstructured(self.linear, 'weight', amount=0.9)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, K, k_similar, name):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, K, k_similar, name + ";stride" + str(stride)))
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

    def update_weights(self, writer: SummaryWriter):
        self.apply(self.update_weights_in_module)
        # self.apply(lambda m : self.log(m, writer))

    def log(self, m, writer):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            import time
            writer.add_scalar(classname, torch.count_nonzero(m.weight) / torch.numel(m.weight), int(time.time()))
            writer.add_scalar(classname + " mask", torch.count_nonzero(m.weight_mask) / torch.numel(m.weight_mask), int(time.time()))

    def update_weights_in_module(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # import time
            # print(torch.count_nonzero(m.weight)/torch.numel(m.weight), int(time.time()))
            # print(torch.count_nonzero(m.weight_mask)/torch.numel(m.weight_mask), int(time.time()))
            remove_smallest_weights(m, self.K)
            if self.update_method == 'faiss':
                weight_magic_faiss(m, self.K, self.k_similar, self.bits)
            elif self.update_method == 'random':
                weight_magic_random(m, self.K, self.k_similar)
            elif self.update_method == 'bruteforce':
                weight_magic(m, self.K, self.k_similar)
            else:
                raise RuntimeError(f"unknown update method: {self.update_method}")

    @staticmethod
    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            init.constant_(m.bias, 0)
            prune.random_unstructured(m, 'weight', amount=0.9)

            m.register_forward_hook(_forward_hook)
            m.register_full_backward_pre_hook(_backward_hook)
        elif classname.find('BatchNorm') != -1:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


def remove_smallest_weights(conv, K):
    K = int(conv.weight.numel() * K)
    weights = np.abs(conv.weight.detach().cpu().numpy().reshape(-1))
    weights[conv.weight_mask.detach().cpu().numpy().reshape(-1) == 0] = np.inf
    indices = np.unravel_index(np.argpartition(weights, K)[:K], conv.weight.shape)
    conv.weight[indices] = 0
    conv.weight_mask[indices] = 0


def get_distances(index, conv, out, k):
    shape = conv.weight.shape
    # index.nprobe = 8
    D, I = index.search(out, k)
    idx = np.array(
        (np.repeat(np.arange(conv.out_channels), k).reshape(conv.out_channels, k),
         *np.unravel_index(I, shape[1:]))).reshape(len(shape), -1)

    distances = D.reshape(-1)

    return idx, distances


def weight_magic_faiss(conv, K, k_similar, bits):
    # print("faiss")
    x = conv.input
    batch_size = x.shape[0]
    out = conv.grad_output
    delattr(conv, "input")
    delattr(conv, "grad_output")

    # sposob B.
    K = int(conv.weight.numel() * K)

    unfold = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
    out_size = conv.in_channels * np.prod(conv.kernel_size)

    unfold = unfold.mean(axis=2).detach().cpu().numpy().T
    out = out.mean(axis=(2, 3)).detach().cpu().numpy().T

    # unfold /= np.linalg.norm(unfold, axis=1).reshape(-1, 1)
    # out /= np.linalg.norm(out, axis=1).reshape(-1, 1)
    # print(len(unfold))
    # if False: # or len(unfold)>5000:
    #     quantizer = faiss.IndexFlatIP(batch_size)
    #     index = faiss.IndexLSH(quantizer, batch_size, int(len(unfold)**0.25))
    #     index.train(unfold)
    # else:
    index = faiss.IndexLSH(batch_size, int(bits*batch_size))

    index.add(unfold)

    k_similar = int(out_size * k_similar)

    to_rand_idx, to_rand = get_distances(index, conv, out, k_similar)
    to_rand = np.abs(to_rand)
    to_rand[conv.weight_mask[to_rand_idx].detach().cpu().numpy() == 1] = -np.inf
    to_rand_idx = to_rand_idx.T[np.argpartition(to_rand, -K)[-K:]]
    to_randn = to_rand_idx.T

    conv.weight[to_randn] = 0
    conv.weight_mask[to_randn] = 1


def weight_magic_random(conv, K, k_similar):
    # print("random")
    delattr(conv, "input")
    delattr(conv, "grad_output")

    K = int(conv.weight.numel() * K)

    indices = torch.nonzero(torch.where(conv.weight_mask == 0, 1, 0))
    p = torch.ones(indices.shape[0], device=device)
    to_randn = tuple(indices[p.multinomial(K, False)].T)

    conv.weight[to_randn] = 0
    conv.weight_mask[to_randn] = 1


def weight_magic(conv, K, k_similar):
    # print("bruteforce")
    x = conv.input
    out = conv.grad_output
    delattr(conv, "input")
    delattr(conv, "grad_output")

    # sposob C.
    def indices_of_smallest(arr, k, conv):
        if k > 0:
            return np.unravel_index(np.argpartition(arr.reshape(-1), k)[:k], conv.weight.shape)
        else:
            return np.unravel_index(np.argpartition(arr.reshape(-1), k)[k:], conv.weight.shape)

    def dist(X, Y):
        sx = np.sum(X ** 2, axis=1, keepdims=True)
        sy = np.sum(Y ** 2, axis=1, keepdims=True)
        return -2 * X.dot(Y.T) + sx + sy.T

    K = int(conv.weight.numel() * K)

    unfold = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
    out_size = conv.in_channels * np.prod(conv.kernel_size)

    unfold = unfold.mean(axis=2).detach().cpu().numpy().T
    out = out.mean(axis=(2, 3)).detach().cpu().numpy().T

    weight_mask = conv.weight_mask.reshape(conv.out_channels, out_size).detach().cpu().numpy()

    distances = out.dot(unfold.T)

    max_distances = np.abs(distances)
    max_distances[weight_mask == 1] = -np.inf
    to_randn = indices_of_smallest(max_distances, -K, conv)

    conv.weight[to_randn] = 0
    conv.weight_mask[to_randn] = 1


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    net = Wide_ResNet_sim_correct(writer, 28, 5, 0.3, 10, 0.01, 1)
    net.apply(net.conv_init)

    y = net(Variable(torch.randn(128, 3, 32, 32)))
    loss = nn.CrossEntropyLoss()(y, Variable(torch.randn(128, 10)))
    loss.backward()

    net.update_weights(writer)

    print(y.size())
