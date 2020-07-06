# Non-convex Learning via Replica Exchange Stochastic Gradient MCMC in Tensorflow
This repository contains the the Tensorflow implementation of Replica Exchange Stochastic Gradient MCMC (reSG-MCMC) by Wei Deng, Qi Feng*, Liyao Gao*, Faming Liang, and Guang Lin, focusing on experiment on Bayesian GANs for semi-supervised learning. We wish to mention that this implementation is largely based on previous work on [Bayesian GANs](https://github.com/andrewgordonwilson/bayesgan) by Yunus Saatchi and Andrew Gordon Wilson. Our paper is going to appear at ICML 2020. 

### Installation of related dependencies
The running environment of this repository is Python 2.7. Please view this [website](https://www.python.org/download/releases/2.7/) to install Python 2. Note here that running with Python 2 is crucial for this version. We would hopefully release version for Python 3 in future. 
#### Conda users
It would be handy for conda users to load the environment directly via *environment.yml*. After the entering of current folder, type these two lines in the terminal. 
```
conda env create -f environment.yml -n bgan
source activate bgan
```
#### Other users
The following list collected all dependencies that the authors used for experiments. Please use **pip** or other methods to download related packages. Note here the versions are only for suggestion (other version of these dependencies might also work).

- Tensorflow 1.0.0
- Numpy 
