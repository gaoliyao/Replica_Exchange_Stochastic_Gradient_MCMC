"""
Bayesian SGHMC, each layer is scaled by its std, fc1 make corresponding change
"""
import torch
import numpy as np

def sghmc(data, y, net, pars, epoch, eta=0.00002, L=5, alpha=0.01, V=1, sd_0=0.001, sd_1=0.1):
    """ Eq. (15) in Chen et al in SGHMC by Tianqi Chen, ICML 2014
        eta: learing rate
        alpha: momentum decay
        V: fisher information matrix, for convenience, just set as I
    """
    temp_U = net.cal_nlpos(data, y)
    net.zero_grad()
    temp_U.backward()
    
    if epoch == 0:
        net.set_scale = [param.data.std().item() + 1e-15 for param in net.parameters()]
    else:
        net.set_scale = [param.grad.data.std().item() + 1e-15 for param in net.parameters()]
    net.set_scale = [scale / max(net.set_scale) for scale in net.set_scale]
    net.set_scale = [(scale * 1.0) / (pars.tempre * 1.0) for scale in net.set_scale]
    

    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in net.parameters()]

    for num, parameter in enumerate(parameters):
        p = net.frame.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta) * net.set_scale[num]
        current_ps.append(p)

    momentum = 1.0 - alpha
    if beta > alpha:
        sys.exit('Eta is too large')
    sigma = np.sqrt(2.0 * eta * (alpha - beta))

    for l in range(L):
        temp_U = net.cal_nlpos(data, y, update=False)
        net.zero_grad()
        temp_U.backward()

        for i in range(len(current_ps)):
            reinitilize = net.frame.FloatTensor(parameters[i].data.size()).normal_() * sigma * net.set_scale[i]
            
            if parameters[i].data_ptr() == net.fc1.weight.data.data_ptr():
                reinitilize *= torch.sqrt((1 - net.p_star) * sd_0 ** 2 + net.p_star * sd_1 ** 2)
                reinitilize /= (np.sqrt((1 - net.p_star.mean()) * (sd_0 / sd_1) ** 2 + net.p_star.mean())).item()
            
            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]

    return temp_U.view(-1).data.tolist()[0]

"""
Bayesian SGHMC, each layer is scaled by its std, fc1 make corresponding change
scale can be exponentialized, 0 represent no scaling, the larger the scaling is higher
"""
def sghmc2(data, y, net, tempreture, eta=0.00002, L=5, alpha=0.01, V=1):
    temp_U = net.cal_nlpos(data, y)
    net.zero_grad()
    temp_U.backward()
        
    # scale for sparse layer
    std_all = net.fc1.weight.data.std().item()
    scale_large = net.fc1.weight.data[net.p_star > 0.5].std().item() / std_all
    scale_small = net.fc1.weight.data[net.p_star <= 0.5].std().item() / std_all

    frame = torch.cuda if torch.cuda.is_available() else torch
    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in net.parameters()]


    for i, parameter in enumerate(parameters):
        p = frame.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta) * set_scale[i]
        if parameters[i].data_ptr() == net.fc1.weight.data.data_ptr():
            p[net.p_star > 0.5] *= scale_large ** (tempreture * 1.0)
            p[net.p_star <= 0.5] *= scale_small ** (tempreture * 1.0)
        current_ps.append(p)

    momentum = 1.0 - alpha
    if beta > alpha:
        sys.exit('Eta is too large')
    sigma = np.sqrt(2.0 * eta * (alpha - beta))

    for l in range(L):
        temp_U = net.cal_nlpos(data, y)
        net.zero_grad()
        temp_U.backward()

        for i in range(len(current_ps)):
            reinitilize = frame.FloatTensor(parameters[i].data.size()).normal_() * sigma * set_scale[i]
            if parameters[i].data_ptr() == net.fc1.weight.data.data_ptr():
                reinitilize[net.p_star > 0.5] *= scale_large ** (tempreture * 1.0)
                reinitilize[net.p_star <= 0.5] *= scale_small ** (tempreture * 1.0)

            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]
    return temp_U.view(-1).data.tolist()[0]

"""
Rather than uniform scaling or sd^t scaling, use sd/t scaling
"""
def sghmc3(data, y, net, eta=0.00002, L=5, alpha=0.01, V=1):
    temp_U = net.cal_nlpos(data, y)
    net.zero_grad()
    temp_U.backward()
        
    frame = torch.cuda if torch.cuda.is_available() else torch
    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in net.parameters()]

    for i, parameter in enumerate(parameters):
        p = frame.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta) * net.set_scale[i]
        current_ps.append(p)

    momentum = 1.0 - alpha
    if beta > alpha:
        sys.exit('Eta is too large')
    sigma = np.sqrt(2.0 * eta * (alpha - beta))

    for l in range(L):
        temp_U = net.cal_nlpos(data, y)
        net.zero_grad()
        temp_U.backward()

        for i in range(len(current_ps)):
            reinitilize = frame.FloatTensor(parameters[i].data.size()).normal_() * sigma * net.set_scale[i]
            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]

    return temp_U.view(-1).data.tolist()[0]

"""
Pure tempreture scaling
"""
def sghmc4(data, y, net, tempreture, eta=0.00002, L=5, alpha=0.01, V=1):
    temp_U = net.cal_nlpos(data, y)
    net.zero_grad()
    temp_U.backward()

    frame = torch.cuda if torch.cuda.is_available() else torch
    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in net.parameters()]


    for i, parameter in enumerate(parameters):
        p = frame.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta) / (tempreture * 1.0)
        current_ps.append(p)

    momentum = 1.0 - alpha
    if beta > alpha:
        sys.exit('Eta is too large')
    sigma = np.sqrt(2.0 * eta * (alpha - beta))

    for l in range(L):
        temp_U = net.cal_nlpos(data, y)
        net.zero_grad()
        temp_U.backward()

        for i in range(len(current_ps)):
            reinitilize = frame.FloatTensor(parameters[i].data.size()).normal_() * sigma / (tempreture * 1.0)

            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]

    return temp_U.view(-1).data.tolist()[0]
