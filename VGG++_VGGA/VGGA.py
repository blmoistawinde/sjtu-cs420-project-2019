import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as tfs
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *


parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
#use_cuda = False
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250
alpha = 0.5

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = tfs.Compose([
    ## 增加更多数据增强
    tfs.ColorJitter(brightness=0.3, contrast=0.3),
    tfs.RandomRotation(10),
    tfs.RandomCrop(cut_size),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
])

transform_test = tfs.Compose([
    tfs.TenCrop(cut_size),
    tfs.Lambda(lambda crops: torch.stack([tfs.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# Model
net = VGG('VGG19')

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda(1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    err0 = 0.005        # for FGSM
    
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(1), targets.cuda(1)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)
        loss = alpha*criterion(outputs, targets)
        loss.backward()
        train_loss += loss.data.item()
        del loss
        inputs_prime = inputs + err0*torch.sign(inputs.grad)
        del inputs
        outputs_prime = net(inputs_prime)
        loss2 = (1-alpha)*criterion(outputs_prime, targets)
        loss2.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss2.data.item()
        del loss2
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().numpy()

    utils.progress_bar(len(trainloader), len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(1), targets.cuda(1)
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().numpy()

    utils.progress_bar(len(PublicTestloader), len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(1), targets.cuda(1)
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().numpy()

    utils.progress_bar(len(PublicTestloader), len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
