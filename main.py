from __future__ import print_function

import argparse
import os
import sys
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch

import config as cf
from networks import *


deterministic = False

if deterministic:
    # deterministic behaviour
    import torch
    torch.manual_seed(0)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--dt', default=1, type=int, help='how many steps between weight update')
parser.add_argument('--K', default=2, type=float, help='frac of weight updates')
parser.add_argument('--k-similar', default=0, type=float, help='frac of similar weight searched')
parser.add_argument('--bits', default=0, type=float, help='bits per input')
parser.add_argument('--sparsity', default=0.8, type=float, help='sparsity of layers')
parser.add_argument('--M', default=0, type=int, help='parameter M of IndexPQ')
parser.add_argument('--stop_epoch', default=10*cf.num_epochs, type=int, help='last epoch with connection update')
parser.add_argument('--no_logs', action='store_true', help="mark tensorboard SummaryWriter as '-to-delete'")
parser.add_argument('--connection-update', help='what connection update we are going to use [faiss/bruteforce/random]')
parser.add_argument('--comment', help='There you can write any comment you like.')
args = parser.parse_args()

if args.no_logs:
    writer = SummaryWriter(comment='-to-delete')
else:
    writer = SummaryWriter(comment='-'+' '.join(sys.argv))
global_step = 0

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

if deterministic:
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    g = torch.Generator()
    g.manual_seed(0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              worker_init_fn=seed_worker, generator=g)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-pruned'):
        net = Wide_ResNetPruned(args.depth, args.widen_factor, args.dropout, num_classes, args.sparsity)
        file_name = 'wide-resnet-pruned-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'wide-resnet-pruned-global'):
        net = Wide_ResNetPrunedGlobal(args.depth, args.widen_factor, args.dropout, num_classes, args.sparsity)
        file_name = 'wide-resnet-pruned-global'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-l1-pruned'):
        net = Wide_ResNetL1Pruned(args.depth, args.widen_factor, args.dropout, num_classes, args.sparsity)
        file_name = 'wide-resnet-l1-pruned-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'wide-resnet-l1-pruned-global'):
        net = Wide_ResNetL1PrunedGlobal(args.depth, args.widen_factor, args.dropout, num_classes, args.sparsity)
        file_name = 'wide-resnet-l1-pruned-global'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-init1'):
        net = Wide_ResNet_init1(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-init1'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-init2'):
        net = Wide_ResNet_init2(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-init2'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-sim'):
        net = Wide_ResNet_sim(writer, args.depth, args.widen_factor, args.dropout, num_classes, args.K, args.k_similar)
        file_name = 'wide-resnet-sim'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'wide-resnet-sim-correct'):
        net = Wide_ResNet_sim_correct(writer, args.depth, args.widen_factor, args.dropout, num_classes, args.K, args.k_similar, args.connection_update, args.bits, args.M, args.sparsity)
        file_name = 'wide-resnet-sim-correct'+str(args.depth)+'x'+str(args.widen_factor)
    
    elif (args.net_type == 'wide-resnet-sim-correct-l1'):
        net = Wide_ResNet_sim_correct_l1(writer, args.depth, args.widen_factor, args.dropout, num_classes, args.K, args.k_similar, args.connection_update, args.bits, args.M, args.sparsity)
        file_name = 'wide-resnet-sim-correct-l1'+str(args.depth)+'x'+str(args.widen_factor)

    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet [pruned] / Wide_ResNet L1 [pruned]')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(net.conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update


        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        len(trainloader)

        writer.add_scalar("loss", loss.item(), (epoch-1)*len(trainloader)+batch_idx+1, new_style=True)
        writer.add_scalar("acc", 100.*correct/total, (epoch-1)*len(trainloader)+batch_idx+1, new_style=True)
        writer.flush()

        acc = 100.*correct/total
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), acc))
        sys.stdout.flush()

        if ((epoch-1)*len(trainloader)+batch_idx+1) % args.dt == 0:
            if epoch <= args.stop_epoch:
                # print("update")
                with torch.no_grad():
                    if isinstance(net, torch.nn.DataParallel):
                        net.module.update_weights(writer)
                    else:
                        net.update_weights(writer)
            else:
                # print("after stop_epoch")
                pass

        if False:
            if isinstance(net, torch.nn.DataParallel):
                net.apply(lambda m: net.module.log(m, writer))
            else:
                net.apply(lambda m: net.log(m, writer))


    writer.add_scalar("loss by epoch", loss.item(), epoch, new_style=True)
    writer.add_scalar("acc by epoch", acc, epoch, new_style=True)
    writer.flush()

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        # try:
        #     for conv, log_name in (net.module if use_cuda else net).conv_to_log:
        #         writer.add_histogram(log_name, conv.weight, epoch)
        #         writer.add_scalar(log_name + " nonzero weights", torch.count_nonzero(conv.weight), epoch)
        #         writer.add_scalar(log_name + " weight count", conv.weight.numel(), epoch)
        #         writer.add_scalar(log_name + " sparsity", 100 * (1 - torch.count_nonzero(conv.weight) / conv.weight.numel()), epoch)
        # except:
        #     pass

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        writer.add_scalar("validation loss by epoch", loss.item(), epoch, new_style=True)
        writer.add_scalar("validation acc by epoch", acc, epoch, new_style=True)
        writer.flush()

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
