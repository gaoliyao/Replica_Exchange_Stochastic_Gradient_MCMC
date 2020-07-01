import torch
import numpy as np

def SGHMC(data, y, bayes_nn, eta=0.00002, L=5, alpha=0.01, V=1):
    current_ps = []
    beta = 0.5 * V * eta
    parameters = [parameter for parameter in bayes_nn.parameters()]

    for parameter in parameters:
        p = torch.cuda.FloatTensor(parameter.data.size()).normal_() * np.sqrt(eta)
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
            reinitilize = torch.cuda.FloatTensor(parameters[i].data.size()).normal_() * sigma
            current_ps[i] = momentum * current_ps[i] - eta * parameters[i].grad.data + reinitilize
            parameters[i].data  = parameters[i].data + current_ps[i]
    current_paras = [para.data.cpu().numpy() for para in parameters]
    return current_paras, temp_U.view(-1).data.tolist()[0]
