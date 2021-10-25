import torch


def kl_loss(var, mean):
    kld_element = mean.pow(2).add_(var).mul_(-1).add_(1).add_(torch.log(var))
    kld = torch.mean(kld_element).mul_(-0.5)
    return kld
