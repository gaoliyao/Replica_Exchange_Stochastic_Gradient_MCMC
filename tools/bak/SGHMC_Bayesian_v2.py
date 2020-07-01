import torch
import numpy as np


def sghmc(data, y, bayes_nn, eta=0.00002, L=5, alpha=0.01, V=1, v0=0.001, v1=0.1, bayes_type='em'):
    """ Eq. (15) in Chen et al in SGHMC by Tianqi Chen, ICML 2014
        eta: learing rate
        alpha: momentum decay
        V: fisher information matrix, for convenience, just set as I
    """
    set_scale = [parameter.data.std() for parameter in bayes_nn.parameters()]
    set_scale = [scale / max(set_scale) for scale in set_scale] # normalize

    frame = torch.cuda if torch.cuda.is_available() else torch

    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in bayes_nn.parameters()]

    for num, parameter in enumerate(parameters):
        p = frame.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta) * set_scale[num]
        if p.data_ptr() == bayes_nn.fc1.weight.data.data_ptr():
            if bayes_type == 'em':
                p *= torch.sqrt((1 - bayes_nn.p_star) * v0 ** 2 + bayes_nn.p_star * v1 ** 2)
                p /= np.sqrt((1 - bayes_nn.p_star.mean()) * (v0 / v1) ** 2 + bayes_nn.p_star.mean())
            else:
                p *= torch.sqrt((1 - bayes_nn.keepfc1) * v0 ** 2 + bayes_nn.keepfc1 * v1 ** 2)
                p /= np.sqrt((1 - bayes_nn.keepfc1.mean()) * (v0 / v1) ** 2 + bayes_nn.keepfc1.mean())
        current_ps.append(p)

    momentum = 1.0 - alpha
    if beta > alpha:
        sys.exit('Eta is too large')
    sigma = np.sqrt(2.0 * eta * (alpha - beta))

    for l in range(L):
        temp_U = bayes_nn.cal_nlpos(data, y)
        bayes_nn.zero_grad()
        temp_U.backward()

        for i in range(len(current_ps)):
            reinitilize = frame.FloatTensor(parameters[i].data.size()).normal_() * sigma * set_scale[i]
            if parameters[i].data_ptr() == bayes_nn.fc1.weight.data.data_ptr():
                if bayes_type == 'em':
                    reinitilize *= torch.sqrt((1 - bayes_nn.p_star) * v0 ** 2 + bayes_nn.p_star * v1 ** 2)
                    reinitilize /= np.sqrt((1 - bayes_nn.p_star.mean()) * (v0 / v1) ** 2 + bayes_nn.p_star.mean())
                else:
                    reinitilize *= torch.sqrt((1 - bayes_nn.keepfc1) * v0 ** 2 + bayes_nn.keepfc1 * v1 ** 2)
                    reinitilize /= np.sqrt((1 - bayes_nn.keepfc1.mean()) * (v0 / v1) ** 2 + bayes_nn.keepfc1.mean())

            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]

    current_paras = [para.data.cpu().numpy() for para in parameters]
    return current_paras, temp_U.view(-1).data.tolist()[0]
