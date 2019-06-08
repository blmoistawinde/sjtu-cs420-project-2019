'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as tfs
import transforms as transforms
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from torchvision.models import resnext101_32x8d

use_cuda = torch.cuda.is_available()

cut_size = 44
alpha = 0.5

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=64, shuffle=False, num_workers=1)
criterion = nn.CrossEntropyLoss()

def PrivateTest_adv(net):
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    err0 = 0.005        # for FGSM
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        loss.backward()
        inputs_prime = inputs + err0*torch.sign(inputs.grad)  # Adversirial example
        outputs = net(inputs_prime)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().numpy()
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/float(total)
    return PrivateTest_acc

# net = VGG('VGG19')
# path = "FER2013_VGG19_aug"
# checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))
# net.load_state_dict(checkpoint['net'])
# net.cuda()
# print("PrivateTest_acc[not adversarial trained]: %0.3f" % PrivateTest_adv(net))
# del net

net_adv = VGG('VGG19')
path = "FER2013_VGG19_adv"
checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'), map_location={'cuda:1':'cuda:0'})
net_adv.load_state_dict(checkpoint['net'])
net_adv.cuda()
print("PrivateTest_acc[adversarial trained]: %0.3f" % PrivateTest_adv(net_adv))
