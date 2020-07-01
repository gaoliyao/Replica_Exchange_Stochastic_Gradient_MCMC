"""
Code for Replica Exchange Stochastic Gradient MCMC on supervised learning
Wei Deng
July 1, 2020

You can cite this paper 'Non-convex Learning via Replica Exchange Stochastic Gradient MCMC (ICML 2020)' if you find it useful.

Note that in Bayesian settings, the lr 2e-6 and weight decay 25 are equivalent to lr 0.1 and weight decay 5e-4 in standard setups.
"""

#!/usr/bin/python

import math
import copy
import sys
import os
import timeit
import csv
import argparse
from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
import pickle
## import pytorch modules
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

from tools import model_eval, save_or_pretrain, bayes_mv
from tools import loader, process_d0
from models.model_zoo import CNN, CNN1, BayesCNN, BayesCNN1
from trainer import trainer

import models.fashion as fmnist_models
import models.cifar as cifar_models

'''
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
'''


def main():
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-aug', default=1, type=float, help='Data augmentation or not')
    parser.add_argument('-split', default=2, type=int, help='Bayes avg every split epochs')
    # numper of optimization/ sampling epochs
    parser.add_argument('-sn', default=1000, type=int, help='Sampling Epochs')
    parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay')
    parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate')
    parser.add_argument('-T', default=0.01, type=float, help='Inverse temperature for low temperature chain')
    parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')
    parser.add_argument('-burn', default=200, type=float, help='burn in iterations for sampling when sn = 1000')
    parser.add_argument('-ifstop', default=1, type=int, help='stop iteration if acc is too low')

    # Parallel Tempering hyperparameters
    parser.add_argument('-chains', default=2, type=int, help='Total number of chains')
    parser.add_argument('-types', default='swap', type=str, help='swap type: greedy (low T copy high T), swap (low high T swap)')
    parser.add_argument('-Tgap', default=0.2, type=float, help='Temperature gap between chains')
    parser.add_argument('-LRgap', default=0.66, type=float, help='Learning rate gap between chains')
    parser.add_argument('-anneal', default=1.02, type=float, help='temperature annealing factor (default for 500 epochs)')
    parser.add_argument('-lr_anneal', default=0.984, type=float, help='lr annealing factor (default for 500 epochs)')
    parser.add_argument('-F_jump', default=0.7, type=float, help='F jump factor')

    # other settings
    parser.add_argument('-ck', default=False, type=bool, help='Check if we need overwriting check')
    parser.add_argument('-data', default='cifar100', dest='data', help='MNIST/ Fashion MNIST/ CIFAR10/ CIFAR100')
    parser.add_argument('-model', default='resnet', type=str, help='resnet / preact / WRN')
    parser.add_argument('-depth', type=int, default=20, help='Model depth.')
    parser.add_argument('-total', default=50000, type=int, help='Total data points')
    parser.add_argument('-train', default=1000, type=int, help='Training batch size')
    parser.add_argument('-test', default=500, type=int, help='Testing batch size')
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')
    parser.add_argument('-multi', default=0, type=int, help='Multiple GPUs')
    parser.add_argument('-var', default=5, type=int, help='estimate variance piecewise, positive value means estimate every [var] epochs')
    parser.add_argument('-windows', default=20, type=int, help='Moving average of corrections')
    parser.add_argument('-alpha', default=0.3, type=float, help='forgetting rate')
    parser.add_argument('-bias_F', default=2000, type=float, help='correction factor F')
    parser.add_argument('-repeats', default=50, type=int, help='number of samples to estimate sample std')


    pars = parser.parse_args()
    """ Step 0: Numpy printing setup and set GPU and Seeds """
    print(pars)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    try:
        torch.cuda.set_device(pars.gpu)
    except: # in case the device has only one GPU
        torch.cuda.set_device(0) 
    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic=True

    """ Step 1: Preprocessing """
    if pars.ck:
        if raw_input('Are you sure to overwrite the pretrained model? [y/n]') not in ['y', 'Y']:
            sys.exit('Fail in overwriting')
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")
    if pars.model == 'resnet':
        if pars.data == 'fmnist':
            net = fmnist_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()
        elif pars.data == 'cifar10':
            net = cifar_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()
        elif pars.data == 'cifar100':
            net = cifar_models.__dict__['resnet'](num_classes=100, depth=pars.depth).cuda()
    elif pars.model == 'wrn':
        if pars.data == 'fmnist':
            net = fmnist_models.__dict__['wrn'](num_classes=10, depth=16, widen_factor=8, dropRate=0).cuda()
        if pars.data == 'cifar10':
            net = cifar_models.__dict__['wrn'](num_classes=10, depth=16, widen_factor=8, dropRate=0).cuda()
        elif pars.data == 'cifar100':
            net = cifar_models.__dict__['wrn'](num_classes=100, depth=16, widen_factor=8, dropRate=0).cuda()
    elif pars.model == 'wrn28':
        if pars.data == 'fmnist':
            net = fmnist_models.__dict__['wrn'](num_classes=10, depth=28, widen_factor=10, dropRate=0).cuda()
        if pars.data == 'cifar10':
            net = cifar_models.__dict__['wrn'](num_classes=10, depth=28, widen_factor=10, dropRate=0).cuda()
        elif pars.data == 'cifar100':
            net = cifar_models.__dict__['wrn'](num_classes=100, depth=28, widen_factor=10, dropRate=0).cuda()
        

    # parallelized over multiple GPUs in the batch dimension
    if pars.multi:
        net = torch.nn.DataParallel(net).cuda()
    nets = [net]
    for _ in range(1, pars.chains):
        nets.append(pickle.loads(pickle.dumps(net)))
    
    """ Step 2: Load Data """
    train_loader, test_loader, targetloader = loader(pars.train, pars.test, pars)
    
    """ Step 3: Bayesian Sampling """
    trainer(nets, train_loader, test_loader, pars)
    

if __name__ == "__main__":
    main()
