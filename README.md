# Replica Exchange Stochastic Gradient MCMC

Experiment code for "[Non-convex Learning via Replica Exchange Stochastic Gradient MCMC](https://arxiv.org/pdf/2008.05367.pdf)". This is a scalable replica exchange (also known as parallel tempering) stochastic gradient MCMC algorithm with clear acceleration guarantees. This algorithm proposes **corrected swaps** to connect the high-temperature process for **exploration** and the low-temperature process for **exploitation**. 

<img src="/figures/path_v5.png" width="300"> <img src="/figures/simulation.png" width="300">


```
@inproceedings{reSGMCMC,
  title={Non-convex Learning via Replica Exchange Stochastic Gradient MCMC},
  author={Wei Deng and Qi Feng* and Liyao Gao* and Faming Liang and Guang Lin},
  booktitle =   {Proceedings of the 37th International Conference on Machine Learning},
  pages =   {2474--2483},
  year =   {2020},
  volume =   {119}
}
```

# Simulation of Gaussian mixture distributions

## Environment

1. R

2. numDeriv (library)

3. ggplot2 (library)

Please check the file in the **simulation** folder




# Optimization of Supervised Learning on CIFAR100


## Environment

1. Python2.7

2. PyTorch >= 1.1

3. Numpy

## How to run code on CIFAR100 using Resnet20

Setup: batch size 256 and 500 epochs. Simulated annealing is used by default.

- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `SGHMC` Set the default learning rate (lr) to 2e-6 and the temperature (T) to 0.01
```bash
$ python bayes_cnn.py -data cifar100 -model resnet -depth 20 -sn 500 -train 256 -lr 2e-6 -T 0.01 -chains 1
```

- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `reSGHMC`  The low-temperature chain has the same setting as SGHMC; The high-temperature chain has a higher lr=3e-6 (2e-6/LRgap) and a higher T=0.05 (0.01/Tgap); The initial F is 3e5. 
```bash
$ python bayes_cnn.py -data cifar100 -model resnet -depth 20 -sn 500 -train 256 -chains 2 -LRgap 0.66 -Tgap 0.2 -F_jump 0.8 -bias_F 3e5
```

- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Naive reSGHMC`  Simply set bias_F=1e300 and F_jump=1 as follows
```bash
$ python bayes_cnn.py -data cifar100 -model resnet -depth 20 -sn 500 -train 256 -chains 2 -F_jump 1 -bias_F 1e300
```

To use a large batch size 1024, you need a slower annealing rate and 2000 epochs to keep the same iterations.
```bash
$ python bayes_cnn.py -data cifar100 -model resnet -depth 20 -sn 2000 -train 1024 -chains 1 -lr_anneal 0.996 -anneal 1.005
$ python bayes_cnn.py -data cifar100 -model resnet -depth 20 -sn 2000 -train 1024 -chains 2 -lr_anneal 0.996 -anneal 1.005 -F_jump 0.8
```

Remark: If you do Bayesian model average every epoch and there are two swaps in the same epoch, the **acceleration may be neutralized**. To handle this issue, you need to consider a cooling time.

To run the WRN models (WRN-16-8 and wrn-28-10) , you can try the following
```bash
$ python bayes_cnn.py -data cifar100 -model wrn -sn 500 -train 256 -chains 2 -F_jump 0.8 -cool 20 -bias_F 3e5
$ python bayes_cnn.py -data cifar100 -model wrn28 -sn 500 -train 256 -chains 2 -F_jump 0.8 -cool 20 -bias_F 3e5
```
Note that in WRN models, we need to include the extra **cooling time** because cases of two consecutive swaps during the same epoch happens a lot and cancel the acceleration effects.

To reduce the hyperparameter tuning cost, you can try **greedy** instead of swap to break the detailed balance. This strategy has the same optimization performance as the swap type. For example
```bash
$ python bayes_cnn.py -data cifar100 -model wrn -types greedy -sn 500 -train 256 -chains 2 -cool 20 -bias_F 3e5
```



# Semi-supervised Learning via Bayesian GAN
## Environment

1. Python2.7

2. Tensorflow == 1.0.0 (version number might be critical)

3. Numpy

## How to run code on CIFAR10 using Replica Exchange Stochastic Gradient MCMC
```bash
python ./bayesian_gan_hmc.py --dataset cifar --numz 10 --num_mcmc 2 --data_path ./output --out_dir ./output --train_iter 15000 --N 4000 --lr 0.00045 -LRgap 0.66 -Tgap 100 --semi_supervised --n_save 100 --gen_observed 4000 --fileName cifar10_4000_0.00045_0.66_100
```
For detailed instruction please check the README.md file inside semi_supervised_learning folder. 
