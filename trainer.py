"""
Code for Replica Exchange Stochastic Gradient MCMC on supervised learning
(c) Wei Deng, Liyao Gao
July 1, 2020

You can cite this paper 'Non-convex Learning via Replica Exchange Stochastic Gradient MCMC (ICML 2020)' if you find it useful.

Note that in Bayesian settings, the lr 2e-6 and weight decay 25 are equivalent to lr 0.1 and weight decay 5e-4 in standard setups.
"""

#!/usr/bin/python
import math
import copy
import sys
import os
import time
import csv
import dill
import argparse
import random
from random import shuffle
import pickle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## Import helper functions
from tools import model_eval, BayesEval, swap
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def trainer(nets, train_loader, test_loader, pars):
    criterion = nn.CrossEntropyLoss()
    init_T = pars.T
    init_lr = pars.lr
    samplers, bmas, corrections, mv_corrections = {}, [], [], []
    for idx in range(pars.chains-1, -1, -1):
        print('Chain {} Initial learning rate {:.2e} temperature {:.2e}'.format(idx, init_lr, init_T))
        sampler = Sampler(nets[idx], criterion, lr=init_lr, wdecay=pars.wdecay, T=init_T, total=pars.total)
        init_T /= pars.Tgap
        init_lr /= pars.LRgap
        samplers[idx] = sampler
        bmas.append(BayesEval())
        corrections.append([])
        mv_corrections.append(sys.float_info.max)
    losses = np.empty((0, pars.chains))
    counter = 1.
    adjusted_corrections = 0
    start = time.time()
    for epoch in range(pars.sn):
        if pars.var > 0 and epoch % pars.var == 0 and epoch >= 10:
            for idx in range(pars.chains):
                stage_losses = []
                nets[idx].eval()
                for _ in range(10):
                    for i, (images, labels) in enumerate(train_loader):
                        images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                        labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                        nets[idx].zero_grad()
                        """ scaled losses """
                        loss = criterion(nets[idx](images), labels).item() * pars.total
                        stage_losses.append(loss)
                        if len(stage_losses) >= pars.repeats:
                            break
                    if len(stage_losses) >= pars.repeats:
                        break
                std_epoch = np.std(stage_losses, ddof=1)
                """ moving window average """
                corrections[idx].append(0.5 * std_epoch**2)
                """ exponential smoothing average """
                if mv_corrections[idx] == sys.float_info.max:
                    mv_corrections[idx] = 0.5 * std_epoch**2
                else:
                    mv_corrections[idx] = (1 - pars.alpha) * mv_corrections[idx] + pars.alpha * 0.5 * std_epoch ** 2
                print('Epoch {} Chain {} loss std {:.2e} variance {:.2e} smothing variance {:.2e} scaled smoothing variance {:.2e}'.format(\
                        epoch, idx, std_epoch, 0.5 * std_epoch**2, mv_corrections[idx], mv_corrections[idx] / pars.bias_F))

        for idx in range(pars.chains):
            nets[idx].train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            counter += 1.
            loss_chains = []
            for idx in range(pars.chains):
                loss = samplers[idx].step(images, labels)
                loss_chains.append(loss)
            losses = np.vstack((losses, loss_chains))
            if losses.shape[0] > 60000 / pars.train:
                losses = np.delete(losses, 0, 0) # delete the first row

            """ Swap (quasi-buddle sort included) """
            for ii in range(pars.chains - 1):
                for idx in range(pars.chains - ii - 1):
                    """ exponential average smoothing """
                    delta_invT = 1. / samplers[idx].T - 1. / samplers[idx+1].T
                    adjusted_corrections = delta_invT * (mv_corrections[idx] + mv_corrections[idx+1]) / pars.bias_F
                    if np.log(np.random.uniform(0, 1)) < delta_invT * (losses[-1, idx] - losses[-1, (idx+1)] - adjusted_corrections):
                        if pars.types == 'greedy':
                            samplers[idx+1].net.load_state_dict(samplers[idx].net.state_dict())
                            print('Epoch {} Copy chain {} to chain {}'.format(epoch, idx, idx+1))
                        elif pars.types == 'swap':
                            temporary = pickle.loads(pickle.dumps(samplers[idx+1].net))
                            samplers[idx+1].net.load_state_dict(samplers[idx].net.state_dict())
                            samplers[idx].net.load_state_dict(temporary.state_dict())
                            print('Epoch {} Swap (with jump F) chain {} with chain {} and increase '.format(epoch, idx, idx+1))
                            pars.bias_F *= pars.F_jump
                        else:
                            sys.exit('Unknown swapping types.')
        """ Anneaing """
        pars.bias_F *= pars.anneal
        for idx in range(pars.chains):
            if epoch > (0.4 * pars.sn) and pars.lr_anneal <= 1.:
                samplers[idx].eta *= pars.lr_anneal
            samplers[idx].set_T(pars.anneal)
            bmas[idx].eval(nets[idx], test_loader, bma=True, burnIn=pars.burn)
            print('Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} Temperature: {:.2E}  Loss: {:0.3f} Corrections: {:0.3f}'\
                    .format(epoch, idx, bmas[idx].cur_acc, bmas[idx].bma_acc, bmas[idx].best_cur_acc, \
                    bmas[idx].best_bma_acc, samplers[idx].T, np.array(losses[-1, idx]).sum(), adjusted_corrections))
            if pars.ifstop and bmas[idx].cur_acc < 15 and epoch > 10:
                exit('Sampling lr may be too large')
        print('')
    end = time.time()
    print('Time used {:.2f}s'.format(end - start))
