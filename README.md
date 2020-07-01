# Replica Exchange Stochastic Gradient MCMC

Experiment code for "[Non-convex Learning via Replica Exchange Stochastic Gradient MCMC](link to be updated)". This is a scalable replica exchange (also known as parallel tempering) stochastic gradient MCMC algorithm with clear acceleration guarantees. This algorithm proposes **corrected swaps** to connect the high-temperature process for **exploration** and the low-temperature process for **exploitation**. 

<img src="/figures/path_v5.png" width="300"> <img src="/figures/simulation.png" width="300">


```
@inproceedings{reSGMCMC,
  title={Non-convex Learning via Replica Exchange Stochastic Gradient MCMC},
  author={Wei Deng and Qi Feng* and Liyao Gao* and Faming Liang and Guang Lin},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```

# Simulation of Gaussian mixture distributions

## Environment

1. R

2. numDeriv (library)

3. ggplot2 (library)

Please check the file in the simulation folder




# Optimization of Supervised Learning on CIFAR 10 & 100


## Environment

1. Python2.7

2. PyTorch > 1.1

3. Numpy

## How to run code on CIFAR10 using Resnet20

Setup: batch size 256 and 500 epochs. Simulated annealing is used by default.

- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `SGHMC` Set the default learning rate to 2e-6 and the temperature to 0.01
```bash
$ python bayes_cnn.py -data cifar10 -model resnet -depth 20 -sn 500 -train 256 -lr 2e-6 -T 0.01 -chains 1
```

- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `reSGHMC`  The low-temperature chain has the same settings as SGHMC; The high-temperature chain has a learning rate 3e-6 (2e-6/0.66) and a temperature 0.05 (0.01/0.2). The initial correction factor is 3e5. 
```bash
$ python bayes_cnn.py -data cifar10 -model resnet -depth 20 -sn 500 -train 256 -chains 2 -LRgap 0.66 -Tgap 0.2 -F_jump 0.7 -bias_F 3e5
```

- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Naive reSGHMC`  Simply set bias_F=1e300 and F_jump=1 as follows
```bash
$ python bayes_cnn.py -data cifar10 -model resnet -depth 20 -sn 500 -train 256 -chains 2 -LRgap 0.66 -Tgap 0.2 -F_jump 1 -bias_F 1e300
```

To use a large batch size 1024, you need a slower annealing rate and 2000 epochs to keep the same iterations.
```bash
$ python bayes_cnn.py -data cifar10 -model resnet -depth 20 -sn 2000 -train 1024 -chains 1 -lr_anneal 0.996 -anneal 1.005 -F_anneal 1.005
```

To run the WRN models, you can use "-model wrn -depth 0" and "-model wrn28 -depth 0" to run WRN-16-8 and wrn-28-10 models, respectively. 

# Semi-supervised Learning via Bayesian GAN
