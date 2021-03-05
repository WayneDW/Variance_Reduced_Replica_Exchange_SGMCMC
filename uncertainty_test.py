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
import models.cifar as cifar_models


parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-c', default='VR_reSGHMC', type=str, help='Other algorithms include cSGHMC')
parser.add_argument('-T_scale', default=1.0, type=float, help='Uncertainty calibration')
pars = parser.parse_args()


net = cifar_models.__dict__['resnet'](num_classes=10, depth=20).cuda()

dataloader = datasets.CIFAR10

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = dataloader(root='../data/CIFAR10', train=False, download=True, transform=transform_test)
test_loader = data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

notcifar = datasets.SVHN(root='../data/SVHN', split='test', download=True, transform=transform_test)
target_loader = data.DataLoader(notcifar, batch_size=256, shuffle=False, num_workers=0)


""" Step 3: Load Model """
torch.set_printoptions(precision=3)

def number_digits(x): return str(x)[:6]
softmax = nn.Softmax(dim=1)


""" Step 3.1: Ensemble outputs and then transform to prob """

Brier_seen, Brier_unseen = 0, 0
entropy_seen, entropy_unseen = 0, 0
output_ensemble_seen, output_ensemble_unseen = [], []
prob_ensemble_seen, prob_ensemble_unseen = [], []

idx = 1
cycle = 4 if pars.c == 'cSGHMC' else 1
sub_sn = 500 / cycle

DIR = 'snapshot_models/' + pars.c
for seed in os.listdir(DIR):
    for filename in sorted(os.listdir(DIR + '/' + seed)):
        if filename[-1] not in ['5']:
            continue
        file_idx = float(filename.split('_')[-1])
        cur_beta = (file_idx % sub_sn) * 1.0 / sub_sn
        if cur_beta < 0.8:
                continue
        net.load_state_dict(torch.load(DIR + '/' + seed + '/' + filename))
        net.eval()

        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data / pars.T_scale
            prob = softmax(outputs)
            if idx == 1:
                output_ensemble_seen.append(outputs)
                prob_ensemble_seen.append(prob)
            else:
                output_ensemble_seen[cnt] = (1. - 1. / idx) * output_ensemble_seen[cnt] + (1. / idx) * outputs
                prob_ensemble_seen[cnt] = (1. - 1. / idx) * prob_ensemble_seen[cnt] + (1. / idx) * prob

        for cnt, (images, labels) in enumerate(target_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data / pars.T_scale
            prob = softmax(outputs)
            if idx == 1:
                output_ensemble_unseen.append(outputs)
                prob_ensemble_unseen.append(prob)
            else:
                output_ensemble_unseen[cnt] = (1. - 1. / idx) * output_ensemble_unseen[cnt] + (1. / idx) * outputs
                prob_ensemble_unseen[cnt] = (1. - 1. / idx) * prob_ensemble_unseen[cnt] + (1. / idx) * prob
        
        idx += 1

        Brier_seen, counts_seen = 0, 0
        # entropy ranges from 0 to 2.5 roughly with each unit of width 0.05
        hist_brier_seen = [0] * 300000
        hist_entropy_seen = [0] * 50
        hist_entropy_unseen = [0] * 50
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_seen = prob_ensemble_seen[cnt]
            one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            counts_seen += prob_seen.shape[0]
            Brier_seen += torch.mean((prob_seen - one_hot)**2,dim=1).sum().item()
            prob_seen_reg = prob_seen + 1e-20
            entropy_idx = (torch.sum(-prob_seen_reg * torch.log(prob_seen_reg), dim=1) / 0.05).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_seen[idx_] += 1
    
        Brier_unseen = 0
        counts_unseen = 0
        for cnt, (images, labels) in enumerate(target_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_unseen = prob_ensemble_unseen[cnt] 
            counts_unseen += prob_unseen.shape[0]
            Brier_unseen += torch.mean((prob_unseen)**2,dim=1).sum().item()
            prob_unseen_reg = prob_unseen + 1e-20
            entropy_idx = (torch.sum(-prob_unseen_reg * torch.log(prob_unseen_reg), dim=1) / 0.05).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_unseen[idx_] += 1
        print('===' * 100)
        print('Seed {} {} cur_beta {:.2f} Seen / Unseen / Total Brier score {:.4f} / {:.3f} / {:.3f}'.format(seed, filename, cur_beta, \
                Brier_seen/counts_seen, Brier_unseen/counts_unseen, (Brier_seen+Brier_unseen)/(counts_seen+counts_seen)))

        print("Entropy seen (from low to high)")
        print(hist_entropy_seen)
        print("Entropy unseen (from high to low)")
        print(hist_entropy_unseen[::-1])
