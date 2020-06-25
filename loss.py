import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
# Domain Confusion Loss
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def RSL(source,target,k=1):
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)

    # Compute singular values of layers output
    d, u,v = torch.svd(source)
    r, l,s = torch.svd(target)

    # Control sign of singular vectors in backprob.
    d,v,r,s= torch.abs(d),torch.abs(v),torch.abs(r),torch.abs(s)


    u_k = u[:-k]

    #BSS with Spectral loss
    u_n = torch.pow(u[-k:],2)
    u_new = torch.cat([u_k,u_n])

    # Compute Spectral loss
    loss = torch.norm(u_new-l)
    loss = loss / ns

    return loss

def BSP(feature_s,feature_t):
        _, s_s, _ = torch.svd(feature_s)
        _, s_t, _ = torch.svd(feature_t)
        sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return sigma

def compute_a_distance(method,input_list,ad_net,random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]

    if "CDAN" in method:
        if random_layer is None:
            op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
            ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        else:
            random_out = random_layer.forward([feature, softmax_output])
            ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    else:
        ad_out = ad_net(feature)

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    predict = (ad_out >0.5).type(torch.int)
    error = torch.sum((predict.float() != dc_target).type(torch.int)) / float(dc_target.size(0))
    if error > .5:# https://github.com/jindongwang/transferlearning/blob/master/code/distance/proxy_a_distance.py
        error = 1. - error
    a_distance = 2*(1-2*error)
    return a_distance
# log_str = "iter: {:05d}, evaluation_accuracy: {:.5f}, RSL: {:.5f}, domain_loss: {:.5f}, classifer_loss: {:.5f}".format(i, temp_acc,rsl_loss_value,transfer_loss_value,classifier_loss_value)
