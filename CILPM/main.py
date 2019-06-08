#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np 
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from torchvision import datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# hyperparameters
batch_size = 128
epoches = 200
lr = 3e-4
tolerance = 6         # stop training when val performance don't improve for $tolerance epoches
early_stopping = tolerance

# Model, opotimizer and criterion
# We use colored, resize images here, so classical Imagenet-pretrained model can be leveraged here.
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 7)
net = net.cuda()
opt = optim.Adam(net.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

# data_loading utilities
def get_train_loader(dir0="train", batch_size=32):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         # data augmentation
         transforms.ColorJitter(brightness=0.3, contrast=0.3),
         transforms.RandomRotation(10),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root="./" + dir0, transform=transform)
    #print(data.class_to_idx)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

def get_test_loader(dir0="test", batch_size=32):
    transform = transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor()])
    data = datasets.ImageFolder(root="./" + dir0, transform=transform)
    #print(data.class_to_idx)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

train_loader = get_train_loader(dir0="train", batch_size=batch_size)
val_loader = get_test_loader(dir0="val", batch_size=batch_size)
test_loader = get_test_loader(dir0="test", batch_size=batch_size)

len_train = len(train_loader.dataset)
len_val = len(val_loader.dataset)
len_test = len(test_loader.dataset)

# train and test procedure
def train(net, train_loader):
    net.train()
    avg_loss, n_batches = 0, 0
    for i, (data, label) in enumerate(train_loader):
        opt.zero_grad()
        data, label = data.cuda(), label.cuda()
        output = net(data)
        loss = criterion(output, label)
        loss.backward()
        opt.step()
        n_batches += 1
        avg_loss += loss.data.item()
    avg_loss /= n_batches
    return avg_loss

def test(net, test_loader):
    net.eval()
    correct = 0
    avg_loss, n_batches = 0, 0
    for i, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data, label = data.cuda(), label.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += torch.eq(pred,label).sum().data.item()
            loss = criterion(output, label)
            n_batches += 1
            avg_loss += loss.data.item()
    avg_loss /= n_batches
    return correct, avg_loss

# main program
if __name__ == "__main__":
    path = "./model"
    best_val_acc = 0
    best_test_acc = 0
    best_test_epoch = 0

    for e in range(epoches):
        t0 = time.time()
        avg_loss = train(net, train_loader)
        print("epoch:%d, train_avg_loss: %.4f" % (e, avg_loss))
        correct, avg_loss = test(net, val_loader)
        val_acc = correct / len_val
        print("val_avg_loss: %.4f, acc: %.4f (%d / %d)" % (avg_loss, val_acc, correct, len_val))
        if val_acc > best_val_acc:
            print('Saving..')
            print("Best_Val_Acc: %0.4f" % val_acc)
            early_stopping = tolerance
            best_val_acc = val_acc
            best_val_epoch = e
            state = {
                'net': net.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_epoch': best_val_epoch,
            }
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'colored_model.pt'))
        else:
            early_stopping -= 1
            if early_stopping <= 0:
                break
            print('Not imporved, tolerance remaining: %d' % early_stopping)
        print("Time cost: %.4f" % (time.time()-t0))
    # Load the model with the best performace on the val set and use it to test
    state = torch.load(os.path.join(path,'colored_model.pt'))
    net.load_state_dict(state['net'])
    correct, avg_loss = test(net, test_loader)
    test_acc = correct / len_test
    print("Test_avg_loss: %.4f, acc: %.4f (%d / %d)" % (avg_loss, test_acc, correct, len_test))
            



