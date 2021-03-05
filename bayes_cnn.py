#!/usr/bin/python


"""
Created on Mar 03, 2021
@author: Wei Deng
Code for Accelerating Convergence of Replica Exchange Stochastic Gradient MCMC via Variance Reduction. ICLR'21
"""

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

from tools import loader
from trainer import trainer

import models.cifar as cifar_models


def main():
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-sn', default=1000, type=int, help='Sampling Epochs')
    parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay (equivalent to 5e-4 in non-Bayes settings)')
    parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate (equivalent to 0.1 in non-Bayes settings)')
    parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')
    parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')

    # Parallel Tempering hyperparameters
    parser.add_argument('-chains', default=1, type=int, help='Total number of chains')
    parser.add_argument('-var_reduce', default=0, type=int, help='n>0 means update variance reduction every n epochs; n divides 10')
    parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
    parser.add_argument('-T', default=0.05, type=float, help='Temperature for high temperature chain')
    parser.add_argument('-Tgap', default=0.2, type=float, help='Temperature gap between chains')
    parser.add_argument('-LRgap', default=0.66, type=float, help='Learning rate gap between chains')
    parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
    parser.add_argument('-LRanneal', default=0.984, type=float, help='lr annealing factor')
    parser.add_argument('-adapt_c', default=0, type=float, help='adapt_c=1 is equivalent to running Alg. 2 in the appendix')
    parser.add_argument('-cool', default=1, type=int, help='No swaps happen during the cooling time after a swap')

    # other settings
    parser.add_argument('-data', default='cifar100', dest='data', help='CIFAR10/ CIFAR100')
    parser.add_argument('-depth', type=int, default=20, help='ResNet depth')
    parser.add_argument('-total', default=50000, type=int, help='Total data points')
    parser.add_argument('-batch', default=256, type=int, help='Batch size')
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')
    parser.add_argument('-alpha', default=0.3, type=float, help='forgetting rate')
    parser.add_argument('-bias_F', default=1.5e5, type=float, help='correction factor F')
    parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')


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
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")
    if pars.data == 'cifar10':
        net = cifar_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()
    elif pars.data == 'cifar100':
        net = cifar_models.__dict__['resnet'](num_classes=100, depth=pars.depth).cuda()

    nets = [net]
    for _ in range(1, pars.chains):
        nets.append(pickle.loads(pickle.dumps(net)))
    
    """ Step 2: Load Data """
    train_loader, test_loader = loader(pars.batch, pars.batch, pars)
    
    """ Step 3: Bayesian Sampling """
    trainer(nets, train_loader, test_loader, pars)
    

if __name__ == "__main__":
    main()
