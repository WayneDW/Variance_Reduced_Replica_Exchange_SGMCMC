#!/usr/bin/python 
import random 
import os
import time
import sys
 
secure_random = random.SystemRandom()


if len(sys.argv) == 2:
    gpu = sys.argv[1]
elif len(sys.argv) > 2:
    sys.exit('Unknown input')
else:
    gpu = '0'

#os.system('sleep 1.2h')

for _ in range(2):
    seed = str(random.randint(1, 10**5))
    VR = secure_random.choice(['0'])
    adapt_c = secure_random.choice(['0', '0'])
    depth = secure_random.choice(['20', '32'])
    F, T, Tanneal = secure_random.choice([('1.5e5', '0.01', '1.02')])
    os.system('python bayes_cnn.py -sn 600 -chains 2 -depth ' + depth + ' -lr 2e-6 -LRanneal 0.984 -T ' + T + ' -Tanneal ' + Tanneal +  ' -var_reduce ' + VR + ' -period 2 -gpu ' + gpu + ' -adapt_c ' + adapt_c + ' -bias_F ' + F + ' -burn 0.6 > ./output/cifar100_resnet' + depth + '_batch_256_chain_2_T_' + T + '_Tanneal_' + Tanneal + '_LRgap_0.66_Tgap_0.2_F_' + F + '_VR_' + VR + '_p_2_burn_0.6_alpha_0.3_cycle_1_adapt_c_' + adapt_c + '_seed_' + seed)
