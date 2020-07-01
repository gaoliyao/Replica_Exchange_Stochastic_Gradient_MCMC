import sys
import numpy as np
from copy import copy, deepcopy
from math import log

""" Eq. (15) in Chen et al in SGHMC by Tianqi Chen, ICML 2014
    eta: learing rate
    alpha: momentum decay
    V: fisher information matrix, for convenience, just set as I
"""

def SGHMC(clf, X_batch, y_batch, Priors, eta=0.00002, L=5, alpha=0.01, V=1):
    beta = 0.5 * V * eta
    p = Priors.reinitialize().mul(np.sqrt(eta))
    momentum = 1. - alpha

    if beta > alpha:
        sys.exit('eta is too large!')
    sigma = np.sqrt(2. * eta * (alpha - beta))

    for i in range(L):
        Priors = clf.calc_grad(X_batch, y_batch, all_priors=Priors)
        p = p.mul(momentum) + Priors.mul(-1 * eta, 'grad')
        p = p +  Priors.reinitialize().mul(sigma)
        Priors = Priors + p
        clf.update_weights(Priors)

    return Priors

        
        
