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

for _ in range(3):
    seed = str(random.randint(1, 10**5))
    chains = secure_random.choice(['1'])
    Tgap = secure_random.choice(['5'])
    lr = secure_random.choice(['0.15'])
    LRgap = secure_random.choice(['0.66'])
    model = secure_random.choice(['resnet'])
    depth = secure_random.choice(['20', '56'])
    data = secure_random.choice(['cifar10', 'cifar100'])
    batch = secure_random.choice(['250', '500', '1000', '2000'])

    if model == 'resnet' and depth == '20':
        if chains == '1':
            invT = secure_random.choice(['2.5e7', '5e7'])
        else:
            invT = secure_random.choice(['5e6', '1e7'])
    elif model == 'resnet' and depth == '56':
        if chains == '1':
            invT = secure_random.choice(['2.5e7'])
        else:
            invT = secure_random.choice(['5e6'])
    elif model == 'wrn':
        depth = '0'
        if chains == '1':
            invT = secure_random.choice(['5e6', '1e7', '5e7'])
        else:
            invT = secure_random.choice(['5e6', '2.5e7', '5e7'])
    else:
        sys.exit('Unknown syntax')

    if batch in ['250', '500']:
        sn = '500'
        #Egap = secure_random.choice(['1.0', '1.05', '1.1', '1.15', '1.2', '1.2', '1.25', '1.25', '1.25', '1.3', '1.35'])
        Egap = secure_random.choice(['1.25'])
        lr_anneal, anneal = '0.984', '1.02'
    elif batch in ['1000', '2000']:
        sn = '2000'
        Egap = secure_random.choice(['1.15'])
        lr_anneal, anneal = '0.996', '1.005'

    if chains == '1':
        lr = secure_random.choice(['0.1'])
        os.system('python bayes_cnn.py -sn ' + sn + ' -data ' + data + ' -model ' + model + ' -depth ' + depth + ' -train ' + batch + ' -gpu ' + gpu + ' -chains ' + chains + ' -invT ' + invT + ' -lr ' + lr + ' -lr_anneal ' + lr_anneal + ' -anneal ' + anneal + ' -seed ' + seed + ' > ./output/adaptive_var_' + data + '_' + model + depth + '_batch_' + batch + '_chain_' + chains + '_invT_' + invT + '_lr_' + lr + '_lr_anneal_' + lr_anneal + '_anneal_' + anneal + '_seed_' + seed)
    else:
        os.system('python bayes_cnn.py -sn ' + sn + ' -data ' + data + ' -model ' + model + ' -depth ' + depth + ' -train ' + batch  + ' -lr ' + lr + ' -LRgap ' + LRgap + ' -Tgap ' + Tgap + ' -Egap ' + Egap + ' -gpu ' + gpu + ' -chains ' + chains + ' -invT ' + invT + ' -lr_anneal ' + lr_anneal + ' -anneal ' + anneal + ' -seed ' + seed + ' > ./output/' + data + '_' + model + depth + '_batch_' + batch + '_chain_' + chains + '_lr_' + lr + '_invT_' + invT + '_lr_anneal_' + lr_anneal + '_anneal_' + anneal + '_LRgap_' + LRgap + '_Tgap_' + Tgap + '_Egap_' + Egap + '_seed_' + seed)
