import math
import numpy as np
import copy
import sys
import os
import timeit
import csv
import dill
from tqdm import tqdm ## better progressbar
from math import exp
from utils import find_classes


import numpy as np
from numpy import genfromtxt


## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn
from torchvision import datasets #, transforms

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
         
import torch.utils.data as data
import torchvision.datasets as datasets
 
import transforms 
from copy import deepcopy
from sys import getsizeof

def imageNet_loader(train_size, valid_size, test_size, crop_size):
    # http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder('./data/kaggle/train', transforms.Compose([transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize, ])), batch_size=train_size, shuffle=True, pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(datasets.ImageFolder('./data/kaggle/valid', transforms.Compose([transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])), batch_size=valid_size, shuffle=True, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('./data/image_net/small_classes/', transforms.Compose([transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize,])), batch_size=test_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def fashion_mnist_loader(train_size, test_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/MNIST-FASHION', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=train_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/MNIST-FASHION', train=False, transform=transforms.ToTensor()),
        batch_size=test_size)
    return train_loader, test_loader


def mnist_loader(train_size, test_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=train_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),  
        batch_size=test_size)
    return train_loader, test_loader

"""
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
"""

def fashion_mnist_aug_loader(train_size, test_size, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.4914]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = datasets.FashionMNIST('./data/MNIST-FASHION', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True)

    testset = datasets.FashionMNIST(root='./data/MNIST-FASHION', train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False)
    return train_loader, test_loader

def mnist_aug_loader(train_size, test_size, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        # https://github.com/hwalsuklee/tensorflow-mnist-cnn/blob/master/mnist_data.py
        #transforms.RandomAffine(translate=0.12),
        transforms.RandomRotation((-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        #transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.4914]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True)

    testset = datasets.MNIST(root='./data/MNIST', train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False)
    return train_loader, test_loader

# Take as many D0 as we want
def process_d0(batches, num):
    full_data = []
    for cnt, (data, y) in enumerate(batches):
        data_g, y_g = Variable(data).cuda(), Variable(y).cuda()
        full_data.append((data_g, y_g))
        if cnt >= num:
            break
    return(full_data)


def model_eval(net, data_loader, epoch=0, if_print=1):
    net.eval() 
    correct = 0 
    total = 0 
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():         
            images, labels = images.cuda(), labels.cuda()  
        outputs = net.forward(images)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
    if if_print:
        print 'Epoch {} Test set accuracy: {:0.2f}%'.format(\
            epoch, 100.0 * correct / len(data_loader.dataset))
    return(100.0 * correct / len(data_loader.dataset))

def bayes_mv(net, data_loader, bmv, epoch, counter=1, final_eval=False, decay=0):
    net.eval()
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = (torch.exp(net.forward(images))).data
        if counter == 1:
            bmv.append(outputs)
        else:
            # optimal decay size from Chen, C, NIPS 2015
            #outputs = outputs * ((counter * 1.0) ** (decay))
            outputs = outputs * net.ensemble_w
            bmv[cnt] += outputs

    if final_eval == True:
        correct, total = 0, 0
        for cnt, (images, labels) in enumerate(data_loader):
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
            prediction = bmv[cnt].max(1)[1]
            correct += prediction.eq(labels.data).sum().item()
        print 'Epoch {} Bayes model avg: {:0.2f}%\n'.format(epoch, 100.0 * correct / len(data_loader.dataset))



def save_or_pretrain(net, num_epochs, model_name):
    if num_epochs > 0:
        print('Save model')
        #torch.save(net, model_name, pickle_module=dill)
        torch.save(net.state_dict(), model_name)
    else:
        print('Use preTrained model')
        #net = torch.load(model_name)
        net.load_state_dict(torch.load(model_name))
    return net
    
