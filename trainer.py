"""
Created on Mar 03, 2021
@author: Wei Deng
Code for Accelerating Convergence of Replica Exchange Stochastic Gradient MCMC via Variance Reduction. ICLR'21

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
import collections
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
from tools import BayesEval
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def trainer(nets, train_loader, test_loader, pars):
    criterion = nn.CrossEntropyLoss()
    init_T, init_lr = pars.T, pars.lr
    samplers, BMAS, myVars, cooling_time, lr_set = {}, [], [], [], []
    for idx in range(pars.chains-1, -1, -1):
        print('Chain {} Initial learning rate {:.2e} temperature {:.2e}'.format(idx, init_lr, init_T))
        sampler = Sampler(nets[idx], criterion, lr=init_lr, wdecay=pars.wdecay, T=init_T, total=pars.total)
        lr_set.insert(0, init_lr)
        init_T /= pars.Tgap
        init_lr /= pars.LRgap
        samplers[idx] = sampler
        BMAS.append(BayesEval())
        myVars.append(sys.float_info.max)
    counter, warm_up, adjusted_corrections = 1., 10, 0
    start = time.time()
    """ Initialization for variance reduction """
    last_full_losses, last_VRnets, corr = [0] * pars.chains, [], [-1] * pars.chains
    for idx in range(pars.chains):
        last_VRnets.append(pickle.loads(pickle.dumps(nets[idx])))

    for epoch in range(pars.sn):
        """ update adaptive variance and variance reduction every [period] epochs """
        if pars.period > 0 and epoch % pars.period == 0 and epoch > warm_up:
            cur_full_losses = [0] * pars.chains
            for idx in range(pars.chains):
                stage_losses, cv_losses = [], []
                nets[idx].eval()
                for i, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    nets[idx].zero_grad()
                    avg_loss = criterion(nets[idx](images), labels).item()
                    cur_full_losses[idx] += avg_loss * pars.batch
                    stage_losses.append(avg_loss * pars.total)
                    if pars.var_reduce:
                        cv_losses.append(criterion(last_VRnets[idx](images), labels).item() * pars.total)

                if pars.adapt_c:
                    adaptive_corr = -np.cov(stage_losses, cv_losses, ddof=1)[0][1] / np.var(cv_losses, ddof=1)
                    corr[idx] = (1 - pars.alpha) * corr[idx] + pars.alpha * adaptive_corr

                if pars.var_reduce:
                    for i in range(len(stage_losses)):
                        stage_losses[i] = stage_losses[i] + corr[idx] * (cv_losses[i] - np.mean(cv_losses))
                std_epoch = np.std(stage_losses, ddof=1)
                myVars[idx] = 0.5 * std_epoch**2 if myVars[idx] == sys.float_info.max else ((1 - pars.alpha) * myVars[idx] + pars.alpha * 0.5 * std_epoch ** 2)
                print('Epoch {} Chain {} loss std {:.2e} variance {:.2e} smooth variance {:.2e} adaptive c {:.2f}'.format(epoch, idx, std_epoch, 0.5 * std_epoch**2, myVars[idx], corr[idx]))
                last_VRnets[idx] = pickle.loads(pickle.dumps(nets[idx]))
                last_full_losses[idx] = cur_full_losses[idx]
        
        for idx in range(pars.chains):
            nets[idx].train()

        if pars.cycle >= 2:
            sub_sn = pars.sn / pars.cycle
            cur_beta = (epoch % sub_sn) * 1.0 / sub_sn
            for idx in range(pars.chains):
                samplers[idx].set_eta(lr_set[idx] / 2 * (np.cos(np.pi * cur_beta) + 1))
                if (epoch % sub_sn) * 1.0 / sub_sn == 0:
                    print('Chain {} Cooling down for optimization'.format(idx))
                    samplers[idx].set_T(1e10)
                elif epoch % sub_sn == int(pars.burn * sub_sn):
                    print('Chain {} Heating up for sampling'.format(idx))
                    samplers[idx].set_T(1e-10)

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            counter += 1.
            loss_chains = []
            for idx in range(pars.chains):
                loss = samplers[idx].step(images, labels)
                """ variance-reduced negative log posterior """
                if pars.var_reduce and epoch > warm_up:
                    control_variate_loss = criterion(last_VRnets[idx](images), labels).item() * pars.total
                    loss = loss + corr[idx] * (control_variate_loss - last_full_losses[idx])
                loss_chains.append(loss)

            """ Swap """
            for idx in range(pars.chains - 1):
                """ exponential average smoothing """
                delta_invT = 1. / samplers[idx].T - 1. / samplers[idx+1].T
                adjusted_corrections = delta_invT * (myVars[idx] + myVars[idx+1]) / pars.bias_F
                if np.log(np.random.uniform(0, 1)) < delta_invT * (loss_chains[idx] - loss_chains[idx+1] - adjusted_corrections):
                    if epoch not in cooling_time:
                        temporary = pickle.loads(pickle.dumps(samplers[idx+1].net))
                        samplers[idx+1].net.load_state_dict(samplers[idx].net.state_dict())
                        samplers[idx].net.load_state_dict(temporary.state_dict())
                        print('Epoch {} Swap chain {} with chain {} and increased F {:0.2e}'.format(epoch, idx, idx+1, pars.bias_F))
                        cooling_time = range(epoch, epoch+pars.cool)
                    else:
                        print('Epoch {} Cooling period'.format(epoch))
        """ Anneaing """
        if epoch < pars.burn * pars.sn:
            pars.bias_F *= pars.Tanneal
        for idx in range(pars.chains):
            if pars.cycle == 1:
                if epoch > 0.4 * pars.sn and pars.LRanneal <= 1.:
                    samplers[idx].eta *= pars.LRanneal
                if epoch < pars.burn * pars.sn:
                    samplers[idx].set_T(pars.Tanneal)
                
            if (pars.cycle == 1 and epoch >= int(pars.burn * pars.sn)) or (pars.cycle > 1 and cur_beta >= pars.burn):
                BMAS[idx].eval(nets[idx], test_loader, bma=True)
            elif (pars.cycle == 1 and epoch < int(pars.burn * pars.sn)) or (pars.cycle > 1 and cur_beta < pars.burn):
                BMAS[idx].eval(nets[idx], test_loader, bma=False)
            print('Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} T: {:.2E}  Loss: {:0.2f} Corrections: {:0.2f}'\
                    .format(epoch, idx, BMAS[idx].cur_acc, BMAS[idx].bma_acc, BMAS[idx].best_cur_acc, BMAS[idx].best_bma_acc, \
                    samplers[idx].T, np.array(loss_chains[idx]).sum(), abs(adjusted_corrections)))
        print('')
    end = time.time()
    print('Time used {:.2f}s'.format(end - start))
