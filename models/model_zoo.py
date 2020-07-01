#!/usr/bin/python
# from __future__ import print_function
import math
import copy
import sys
import os
import timeit
import csv
from tqdm import tqdm ## better progressbar
from math import exp
from utils import find_classes

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn

from numpy import genfromtxt
from copy import deepcopy

"""
Network general framework
        NN     Bayes
       /  \    /  \
     CNN1  CNN2  CNN3 ...
       \    |    /
        BayesCNN
"""
class NN(nn.Module):
    def __init__(self, pars):
        super(NN, self).__init__()

        self.t, self.sd, self.theta, self.scales = [None] * 4
        self.frame = torch.cuda if torch.cuda.is_available() else torch
        self.double = pars.double
        print('================================================')
        print('Model={}-{} Data={} tn={} sn={} Augmentation={}'.format(pars.model, pars.c, pars.data, pars.tn, pars.sn, pars.aug))
        print('Scale MLE={} Double={} eta={} Tempreture={}'.format(pars.mle, pars.double, pars.lr, pars.tempre))
        print('Decay={},{},{}'.format(pars.dc, pars.da, pars.dalpha))
        print('Parts={} MinE={} Div={} zeta={} D0={} Ginit={}'.format(pars.part, pars.bias, pars.div, pars.zeta, pars.d0, pars.g))

    
    def layers(self, states):   
        print('------- Layer name and parameter number -------')
        for num, name in enumerate(states):
            print('   {}, Weights: {}'.format(name, np.prod(states[name].size())))


    def calc_std(self):
        for num, name in enumerate(self.state_dict()):
            param = self.state_dict()[name]
            print('{}, std: {:0.4f}, max: {:0.4f}, min: {:0.4f}'.format(name, param.std(), param.max(), param.min()))

    # empty function to be overridden
    def convs(self, x):
        return(x)

    def clf(self, x):
        return(x)

    def forward(self, x, dropout=False):
        x = self.convs(x)
        if dropout:
            x = Func.dropout(x, p=0.5, training=self.training)
        return(self.clf(x))

    def cal_nllloss(self, x, true_y, dropout=True): # negative log likelihood 
        nllloss = nn.NLLLoss(size_average=False)
        loss = nllloss.forward(self.forward(x, dropout), true_y)
        return(loss)

class Bayes(NN):
    def __init__(self, pars):
        super(Bayes, self).__init__(pars)
        self.tempre = pars.tempre
        # decay coefficients
        self.dcoef = {'c': pars.dc, 'A': pars.da, 't': 1.0, 'alpha': pars.dalpha}
        # batch size
        self.batch = pars.train
        # update covariance
        self.counter = 1
        # scale MLE with N / n
        self.mle = pars.mle
        self.ensemble_w = 1

    def cal_nlpos(self, x, true_y, update=True, dropout=True): # no sparse setting, use dropout instead
        nlloss = self.cal_nllloss(x, true_y, dropout)
        nlloss *= 600000 / self.batch
        return(nlloss)

    def update_covariance(self):
        scales = [par.grad.data.std().item() + 1e-6 for i, par in enumerate(self.parameters()) if i in self.idx_name]
        scales = np.array([scale / max(scales) for scale in scales]) ** self.tempre
        self.scales = scales
        self.counter += 1

"""
small models in our framework
"""
class CNN(NN):
    def __init__(self, pars): 
        super(CNN, self).__init__(pars)       
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) 
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) 
        self.fc1 = nn.Linear(64*7*7, 200) 
        self.fc2 = nn.Linear(200, 10)     

        self.idx_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.p = np.prod(self.fc1.weight.data.size()) 
        self.b = self.p 
        self.layers(self.state_dict())

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2)
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2) 
        return(x)

    def clf(self, x): 
        x = x.view(-1, 64*7*7)    
        x = Func.relu(self.fc1.forward(x))  
        x = self.fc2.forward(x) 
        return(Func.log_softmax(x, dim=1))

"""
Used for large models
https://github.com/hwalsuklee/tensorflow-mnist-cnn
The authors claim that they can get 99.6, 99.7 at best.
"""
'''
class CNN2(NN):
    def __init__(self, pars):
        super(CNN2, self).__init__(pars)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0)
        self.fc1 = nn.Linear(64*5*5, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.idx_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.p = np.prod(self.fc1.weight.data.size())
        self.b = self.p
        self.layers(self.state_dict())

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2)
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2)
        return(x)

    def clf(self, x):
        x = x.view(-1, 64*5*5)
        x = Func.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return(Func.log_softmax(x, dim=1))
'''

"""
Mid MNIST/F-MNIST models
github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb
"""
class CNN1(NN):
    def __init__(self, pars):
        super(CNN1, self).__init__(pars)
        self.bn_conv1 = nn.BatchNorm2d(1) 
        self.conv1 = nn.Conv2d(1, 64, 4, padding=2)    
        self.conv2 = nn.Conv2d(64, 64, 4, padding=0)  
        self.fc1 = nn.Linear(64*5*5, 256)   
        self.fc2 = nn.Linear(256, 64) 
        self.bn_fc3 = nn.BatchNorm1d(64)         
        self.fc3 = nn.Linear(64, 10) 

        self.p = np.prod(self.fc1.weight.data.size()) 
        self.b = self.p 
        # the 1st two BN layers has just 1 parameter
        self.idx_name = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.layers(self.state_dict())

    def convs(self, x):
        x = Func.relu(self.conv1.forward(self.bn_conv1(x))) # BatchNorm
        x = Func.max_pool2d(x, 2) 
        x = Func.dropout(x, p=0.1, training=self.training)
        x = Func.relu(self.conv2.forward(x))  
        x = Func.max_pool2d(x, 2)    
        x = Func.dropout(x, p=0.3, training=self.training)  
        x = x.view(-1, 64*5*5) 
        x = Func.relu(self.fc1.forward(x))            
        return(x)

    def clf(self, x):
        x = Func.relu(self.bn_fc3(self.fc2.forward(x))) # BatchNorm  
        x = self.fc3.forward(x)  
        return(Func.log_softmax(x, dim=1))


class CNN2(NN):
    def __init__(self, pars):
        super(CNN2, self).__init__(pars)
        self.conv1 = nn.Conv2d(1, 64, 4, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=0)
        self.fc1 = nn.Linear(64*5*5, 500)
        self.fc2 = nn.Linear(500, 10)

        self.idx_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.p = np.prod(self.fc1.weight.data.size())
        self.b = self.p
        self.layers(self.state_dict())

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2) 
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2) 
        return(x)

    def clf(self, x):    
        x = x.view(-1, 64*5*5)
        x = Func.relu(self.fc1.forward(x))   
        x = self.fc2.forward(x) 
        return(Func.log_softmax(x, dim=1)) 



"""
small models
"""
class BayesCNN(CNN, Bayes):
    def __init__(self, pars):
        super(BayesCNN, self).__init__(pars)

"""
mid models
"""
class BayesCNN1(CNN1, Bayes):
    def __init__(self, pars):
        super(BayesCNN1, self).__init__(pars)
"""
large models
"""
class BayesCNN2(CNN2, Bayes):
    def __init__(self, pars):
        super(BayesCNN2, self).__init__(pars)
        
    

'''
    if pars.c in ['dropout', 'alpha', 'vanilla']:
        feature_set, label_set, predictions = adversarial_images(net, pars.eps, 100)
        repeats = 30
        accuracy = 0
        for _ in range(repeats):
            adv_acc, _,  _ = adversarial_test(net, feature_set, label_set, predictions, if_print=0)
            accuracy += adv_acc
            print 'Adversarial Accuracy:', accuracy / repeats
'''


"""
Implementation of alpha-divergence models
"""

def log_sum_exp(tensor, dim=None):
    xmax, _ = torch.max(tensor, dim=dim, keepdim=True)
    xmax_, _ = torch.max(tensor, dim=dim)
    return xmax_ + torch.log(torch.sum(torch.exp(tensor - xmax), dim=dim))

class MC_Loss(nn.Module):
    '''   
    bbalpha softmax cross entropy with mc_logits
    ''' 
    def __init__(self, pars):
        super(MC_Loss, self).__init__()
        self.alpha = pars.alpha
        if alpha == 0:
            print "alpha = 0"
        self.k_mc = pars.mc

    def forward(self, mc_logits, y_true):
        y_true = y_true.expand(self.k_mc, -1, -1).contiguous().permute(1, 0, 2)
        if self.alpha != 0:
            # log(p_ij), p_ij = softmax(logit_ij)
            #assert mc_logits.ndim == 3
            temp, _ = torch.max(mc_logits, dim=2, keepdim=True)
            mc_log_softmax = mc_logits - temp
            mc_log_softmax = mc_log_softmax - torch.log(torch.sum(torch.exp(mc_log_softmax), dim=2, keepdim=True))
            mc_ll = torch.sum(y_true * mc_log_softmax, dim = -1)  # N x K
            # print mc_ll.size()
            out = - 1. / self.alpha * (log_sum_exp(self.alpha * mc_ll, 1) + np.log(1.0 / self.k_mc))
            # print out.size()
            # sys.exit()
            return torch.sum(out)
        else:
            predictions = Func.log_softmax(mc_logits, dim=2)
            # print y_true, predictions
            out = - torch.sum(torch.mean(y_true * predictions, dim=1))
            return out

#class AlphaCNN(CNN):
class AlphaCNN(nn.Module):
    def __init__(self, pars, wd = 1e-6):   
        super(AlphaCNN, self).__init__()  
        
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # input is 28x28    
        self.wd = wd 
        self.k_mc = pars.mc         
        self.alpha = pars.alpha 
        self.bbalpha_loss = MC_Loss(pars.alpha, pars.mc)     

    def lin(self, x):
        '''
        x = self.convs(x)
        x = Func.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = Func.relu(self.fc1.forward(x))
        out = self.fc2.forward(x)
        #out = self.clf(x)
        '''
        #########################
        x = Func.relu(self.conv1.forward(x))
        x = Func.max_pool2d(x, 2)
        x = Func.relu(self.conv2.forward(x))
        x = Func.max_pool2d(x, 2)
        x = Func.dropout(x, p = 0.5, training=self.training)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = Func.relu(self.fc1.forward(x))
        out = self.fc2.forward(x)
        
        return out

    def generate_MC_samples(self, x):    
        if self.k_mc == 1:    
            out = self.lin(x) 
            mc_logits = out.view(len(out), 1, -1) # nb_batch x K_mc x nb_classes
            return mc_logits   
        else:     
            output_list = []             
            for _ in xrange(self.k_mc): 
                output_list += [self.lin(x)] # time consuming
            output = torch.stack(output_list) # K_mc x nb_batch x nb_classes 
            mc_logits = output.permute(1, 0, 2) # nb_batch x K_mc x nb_classes 
            return mc_logits        

    def forward(self, x): 
        lin_out = self.generate_MC_samples(x)         
        out = Func.softmax(lin_out, dim=-1) 
        out = torch.mean(out, dim=1).squeeze()        
        return out   

    def cal_bbalpha_loss(self, x, true_y): # negative log likelihood    
        mc_logits = self.generate_MC_samples(x) 
        loss = self.bbalpha_loss.forward(mc_logits, true_y) 
        # negative log-likelihood     
        return loss    

    def cal_priors(self): # dropout may cause two forwards different    
        prior = 0 
        for param in self.parameters():          
            prior = prior + self.wd * torch.sum(param**2) 
        return prior  

    def cal_npos(self, x, true_y): # negative log posterior     
        return self.cal_bbalpha_loss(x, true_y) + self.cal_priors() 





"""
For basic ImageNet dataset
"""

class CNN_ImageNet(CNN): # extend MNIST structure to ImageNet
    def __init__(self):    
        super(CNN_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)   
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 18 * 18, 64)            
        self.fc2 = nn.Linear(64, 2)           

        self.p = np.prod(self.fc1.weight.data.size())

    def print_name(self): 
        print('You are working on Naive CNN for ImageNet')

    def convs(self, x):    
        x = Func.relu(self.conv1.forward(x))   
        x = Func.max_pool2d(x, 2) 
        x = Func.relu(self.conv2.forward(x))       
        x = Func.max_pool2d(x, 2) 
        x = Func.relu(self.conv3.forward(x))   
        x = Func.max_pool2d(x, 2)  
        x = x.view(-1, 64 * 18 * 18)   # reshape Variable 
        x = Func.relu(self.fc1.forward(x))      
        return x 

class BayesCNN_ImageNet(CNN_ImageNet):
    def __init__(self): 
        super(BayesCNN_ImageNet, self).__init__()

    def print_name(self):
        print('You are working on Bayesian CNN for ImageNet')

    def forward(self, x): # No dropout
        return Func.log_softmax(self.fc2.forward(self.convs(x)), dim=1)

    def forward_testing(self, x):#, w):
        x = self.convs(x)
        new_drop = nn.Dropout(p=0.5)          
        x = new_drop(x) # * torch.autograd.Variable(w, requires_grad=False)       
        x = self.fc2.forward(x)        
        out = Func.log_softmax(x, dim=1)                  
        return out  

