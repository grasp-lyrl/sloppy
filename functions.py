# import re
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets

from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from torch.autograd import Variable, grad
from numpy.linalg import eig as eig
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import *
from models.fc import Network
import scipy
from scipy.linalg import eigh_tridiagonal
from dataset import create_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.wide_resnet_1 import WideResNet
import time

from backpack import extend, backpack
from backpack.extensions import (
    GGNMP,
    HMP,
    KFAC,
    KFLR,
    KFRA,
    PCHMP,
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    BatchDiagHessian,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SumGradSquared,
    Variance,
)




















# mode = 'none'  
# 'train': train and save models   
# 'lanczo': use lanczo method to calculate the eigen spectrum of hessian.
# 'scipy': use scipy.sparse.linalg.eigsh to calculate the eigen spectrum of fisher
# 'fisher': eigen spectrum of FIM 
# 'trad': calculate full hessian then do eigen value decomposition
# 'eva_posterior': estimate the first term of PAC-Bayes bound (both for CE loss and for 0-1 loss) by sampling from the posterior calculated in bound_1.py.














def train(model, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# def diff_norm(u_list, v_list):
#     '''
#     u_list, v_list: lists of tensors
#     return: difference of u_list and v_list
#     '''
#     diff = []
#     for (u, v) in zip(u_list, v_list):
#         diff.append(u-v)

#     diff_norm = norm_2_list(diff)

#     return(diff_norm)




def train_decay(model, model_init, decay, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        diff = diff_norm(list(model.parameters()), list(model_init.parameters()))

        loss = loss + decay * diff**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print((diff**2).item())











def train_LBFGS(model, device, train_loader, criterion, optimizer, epoch):

    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def closure():
            output = model(data)
            loss = criterion(output, target)    
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)












def val(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    
    model.eval()
    # with torch.no_grad():
    for i, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        # print(output)

        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * criterion(output, target).item()

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset)





def val_grad(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.tensor([]).to(device)
    model.eval()
    for i, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1)[1]
        sum_loss += len(data) * criterion(output, target)

    return sum_loss / len(val_loader.dataset)












def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p


def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):

    '''
    train_accur : training accuracy
    B_init: the second term of pac-bayes bound
    return: approximated pac bayes bound using inverse of kl
    eg: err = approximate_BPAC_bound(0.9716, 0.2292)
    '''
    B_RE = 2* B_init **2
    A = 1-train_accur
    B_next = B_init+A
    if B_next>1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next,A,B_RE)
    return B_next















def FIM(model, criterion, loader, num_params, device):

    '''
    empirical fisher using the data in loader
    '''

    grad_all = torch.empty((len(loader), num_params))

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    # print(grad_all.shape)
    # print(len(loader))
    FIM = (grad_all@grad_all.T)/len(loader)
    L, _ = torch.eig(FIM)

    return FIM, L







def FIM2(model, criterion, loader, device, k):

    '''
    calculates the empirical fisher using the data in loader
    model should be trained, fisher is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)

    FIM = FIM.cpu().detach().numpy()
    L, v = scipy.linalg.eigh(FIM, driver = 'evx', subset_by_index = [len(loader)-k, len(loader)-1])

    L = torch.tensor(L)
    v = torch.tensor(v)
    u = (grad_all.T / np.sqrt(len(loader))) @ v
    u = u * torch.sqrt(1/L)
    idx = list(np.flip(L.numpy().argsort()))
    L = L[idx]
    u = (u.T[idx]).T

    return FIM, L, u

















def FIM_true(model, criterion, loader, device, k):

    '''
    calculates the true fisher using the data in loader
    model should be trained, fisher is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''

    
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pr = F.softmax(output, dim=1)
        y = torch.multinomial(pr, num_samples=1)
        y = y.reshape(target.shape)


        loss = criterion(output, y)
        loss.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)

    FIM = FIM.cpu().detach().numpy()
    # FIM = (FIM + FIM.T) / 2

    L, v = scipy.linalg.eigh(FIM, driver = 'evx', subset_by_index = [len(loader)-k, len(loader)-1])

    L = torch.tensor(L)
    v = torch.tensor(v)
    u = (grad_all.T / np.sqrt(len(loader))) @ v
    u = u * torch.sqrt(1/L)
    idx = list(np.flip(L.numpy().argsort()))
    L = L[idx]
    u = (u.T[idx]).T

    return FIM, L, u



















def logit_jacobian(model, cl, criterion, loader, device, k):

    '''
    calculates the logit jacobian of a certain class cl
    logit jacobian is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        logit = output[0, cl]
        logit.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)

    FIM = FIM.cpu().detach().numpy()
    L, v = scipy.linalg.eigh(FIM, driver = 'evx', subset_by_index = [len(loader)-k, len(loader)-1])

    L = torch.tensor(L)
    v = torch.tensor(v)
    u = (grad_all.T / np.sqrt(len(loader))) @ v
    u = u * torch.sqrt(1/L)
    idx = list(np.flip(L.numpy().argsort()))
    L = L[idx]
    u = (u.T[idx]).T

    return FIM, L, u























def FIM_kfac(model, loader, device, mc, mode = "kfra", empirical = True):

    model_kf = model.classifier.to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)

    for (p1, p2) in zip(model_kf.parameters(), model.parameters()):
        p1.data = p2.data

    criterion1 = extend(criterion1)
    model_kf = extend(model_kf)
    sum_loss = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        data = data.view(data.size(0), -1)
        output = model_kf(data)
        

        if empirical == True:
            loss = criterion1(output, targets)
        else:
            pr = F.softmax(output, dim=1)
            y = torch.multinomial(pr, num_samples=1)
            y = y.reshape(targets.shape)
            loss = criterion1(output, y)
        
        sum_loss += loss

    with backpack(KFAC(mc_samples=mc), KFLR(), KFRA()):
        sum_loss.backward()

    kfac_list = []
    for n, p in model_kf.named_parameters():
        d = dict({"kfac":p.kfac, "kflr":p.kflr, "kfra":p.kfra})
        kfac_list.append(d[mode])

    return kfac_list
















def FIM2x(model, criterion, loader, device):

    '''
    calculates the empirical fisher using the data in loader
    model should be trained, fisher is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)       
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)
    FIM = FIM.cpu().detach().numpy()
    L = np.linalg.eigvalsh(FIM)
    L = L[::-1]

    return L



def FIM_truex(model, criterion, loader, device):

    '''
    calculates the true fisher using the data in loader
    model should be trained, fisher is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''

    
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pr = F.softmax(output, dim=1)
        y = torch.multinomial(pr, num_samples=1)
        y = y.reshape(target.shape)


        loss = criterion(output, y)
        loss.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)
    FIM = FIM.cpu().detach().numpy()

    L = np.linalg.eigvalsh(FIM)
    L = L[::-1]

    return L



def logit_jacobianx(model, cl, criterion, loader, device):

    '''
    calculates the logit jacobian of a certain class cl
    logit jacobian is calculated at the param in model.
    k: number of eigen values
    loader: the data loader used for FIM calculation, should have batch size 1, use train_loader_FIM. Use 'cpu if the model is large'
    return: torch.tensor of FIM (num_data, num_data), L (k, ), v (num_params, u)

    '''
    model = model.to(device)
    criterion = criterion.to(device)
    num_params = sum(param.numel() for param in model.parameters())

    grad_all = torch.empty((len(loader), num_params)).to(device)
    # check this
    model.eval()

    for i, (data, target) in enumerate(loader):
        # print(i)
        data, target = data.to(device), target.to(device)
        output = model(data)
        logit = output[0, cl]
        logit.backward()
        grad_list = []
        for param in model.parameters():
            m = param.grad.clone().detach()
            m = torch.reshape(m, (-1,))
            grad_list.append(m)

        for param in model.parameters():
            param.grad.data.zero_()

        grad = torch.cat(grad_list)
        
        grad_all[i] = grad

    
    FIM = (grad_all@grad_all.T)/len(loader)

    FIM = FIM.cpu().detach().numpy()
    L = np.linalg.eigvalsh(FIM)
    L = L[::-1]

    return L



































def eigspace_FIM_kron(kfac_list):

    eigspace_list =[]
    eigval_list = []
    for kfac in kfac_list:
        if len(kfac) == 2:
            outmat, inmat = kfac
            outmat = (outmat + outmat.T) / 2
            inmat = (inmat + inmat.T) / 2
            outmat, inmat = outmat.detach().cpu(), inmat.detach().cpu()
            eo, vo = np.linalg.eig(outmat)
            ei, vi = np.linalg.eig(inmat)
            eo, vo = torch.tensor(np.real(eo)), torch.tensor(np.real(vo))
            ei, vi = torch.tensor(np.real(ei)), torch.tensor(np.real(vi))
            # e = (torch.kron(eo.contiguous(), ei.contiguous())).reshape(len(outmat), len(inmat))

            eigspace_list.append((vo, vi))
            eigval_list.append((eo, ei))

        if len(kfac) == 1:
            mat = kfac[0]
            mat = (mat + mat.T) / 2
            mat = mat.detach().cpu()
            e, v = np.linalg.eig(mat)
            e, v = torch.tensor(np.real(e)), torch.tensor(np.real(v))

            eigspace_list.append((v,))
            eigval_list.append((e,))

    return eigspace_list, eigval_list




def trans_eigval(eigval_list):
    eigval_true_list = []
    for fac in eigval_list:
        if len(fac) == 2:
            eigval_true = torch.outer(fac[0], fac[1])
        if len(fac) == 1:
            eigval_true = fac[0]
        eigval_true_list.append(eigval_true)

    return eigval_true_list





def kfac_top_eigvec(kfac_list, model, k):
    num_params = sum(p.numel() for p in model.parameters())
    eigspace_list, eigval_list = eigspace_FIM_kron(kfac_list)
    eigval_true_list = trans_eigval(eigval_list)
    tag1 = list_to_vec([torch.ones(p.shape)*i for i, p in enumerate(model.parameters())])
    tag2 = list_to_vec([torch.arange(p.numel()) for p in model.parameters()])
    eigval = list_to_vec(eigval_true_list)

    idx = list(np.flip(eigval.numpy().argsort()))
    eigval, tag1, tag2 = eigval[idx], tag1[idx], tag2[idx]

    eig_vec_list = []

    for i in range(k):
        vec_list = [torch.zeros(p.shape) for p in model.parameters()]
        pos_list, pos_vec = int(tag1[i]), int(tag2[i])
        es = eigspace_list[pos_list]

        if len(es) == 2:
            vo, vi = es
            p = list(model.parameters())[pos_list].detach().cpu()
            h = torch.zeros(p.numel())
            h[pos_vec] = 1
            h = h.reshape(p.shape)
            vec = vo@h@vi.T

        if len(es) == 1:
            v = es[0]
            p = list(model.parameters())[pos_list].detach().cpu()
            h = torch.zeros(p.numel())
            h[pos_vec] = 1
            h = h.reshape(p.shape)
            vec = v@h

        vec_list[pos_list].data = vec

        eig_vec = list_to_vec(vec_list)
        eig_vec = eig_vec.reshape(len(eig_vec), 1)
        eig_vec_list.append(eig_vec)


    eig_vec = torch.cat(eig_vec_list, dim=1)
    eig_val = eigval[:k]
    return eig_val, eig_vec






















def diag_hess(model, loader, device):
    model_kf = model.classifier.to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)

    for (p1, p2) in zip(model_kf.parameters(), model.parameters()):
        p1.data = p2.data

    criterion1 = extend(criterion1)
    model_kf = extend(model_kf)

    sum_loss = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        data = data.view(data.size(0), -1)
        output = model_kf(data)
        sum_loss += criterion1(output, targets)

    

    with backpack(DiagHessian()):
        sum_loss.backward()

    hess_diag_list = []

    for p in model_kf.parameters():
        hess_diag_list.append(p.diag_h)

    return hess_diag_list
              








def overlap(A, B, k, device):

    A, B = torch.tensor(A).to(device), torch.tensor(B).to(device)
    over = torch.zeros(k).to(device)

    for i in range(k):
        print(i)
        a = A[:, :i+1]
        b = B[:, :i+1]
        # print(a.shape)
        overlap = torch.norm(a.T@b, p='fro')**2 / (i+1)
        # print(overlap)
        over[i] = overlap

    return over.detach().cpu()





def proj(vec, spa, k, device):

    vec = torch.tensor(vec).to(device)
    spa = torch.tensor(spa).to(device)

    frac_all = torch.zeros(k).to(device)


    for i in range(k):
        i=i+1
        print(i) 
        normp = torch.norm(spa[:, :i].T@vec, p=2)**2
        norm = torch.norm(vec, p=2)**2

        frac = normp / norm
        frac_all[i-1] = frac

    return frac_all








def proj_single(vec, spa, k, device):
    vec = torch.tensor(vec).to(device)
    spa = torch.tensor(spa).to(device)
    frac_all = torch.zeros(k).to(device)

    for i in range(k):
        print(i)
        normp = torch.sum(spa[:, i]*vec)**2
        norm = torch.norm(vec, p=2)**2
        frac = normp / norm
        frac_all[i] = frac

    return frac_all












def fnc_2(model, loader, param_list, criterion, device):

    '''
    param_list: list of tensors of parameters in model
    return: averaged loss of model
    '''

    sum_loss = 0

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        output = data.view(data.size(0), -1)
        output = F.linear(output, param_list[0], param_list[1])
        output = F.relu(output)
        output = F.linear(output, param_list[2], param_list[3])

        loss = criterion(output, target)
        sum_loss += len(data) * loss

    return sum_loss / len(loader.dataset) 






def norm_2(model):

    '''
    return: 2-norm of parameters in model
    gradient cannot pass throught this operation
    '''

    norm = 0
    for param in model.parameters():
        norm += torch.norm(param.data, p=2)**2

    
    return torch.sqrt(norm)






def norm_2_list(param_list):

    '''
    param_list: list of parameters of a model
    return: 2-norm of parameters in model
    gradient can pass through this operation
    '''

    norm = 0
    for param in param_list:
        norm += torch.norm(param, p=2)**2

    return torch.sqrt(norm)








def prod_list(u_list, v_list):

    '''
    u_list, v_list: lists of tensors
    return: dot product of u_list and v_list
    '''

    prod = 0
    for (u, v) in zip(u_list, v_list):
        prod += torch.sum(u*v)

    return prod


def diff_list(u_list, v_list):
    '''
    u_list, v_list: lists of tensors
    return: difference of u_list and v_list
    '''
    diff = []
    for (u, v) in zip(u_list, v_list):
        diff.append(u-v)

    return(diff)






def diff_norm(u_list, v_list):
    '''
    u_list, v_list: lists of tensors
    return: difference of u_list and v_list
    '''
    diff = []
    for (u, v) in zip(u_list, v_list):
        diff.append(u-v)

    diff_norm = norm_2_list(diff)

    return(diff_norm)





def bound(model, L, epsilon, a, n_true, delta):

    
    count = torch.zeros(len(L))
    count[L > epsilon/a] = 1
    d0 = torch.count_nonzero(count)

    print("d0",d0)
    
    b1 = (d0 / a).to(device)

    norm = norm_2(model)
    print("norm", norm)

    KL = 0.5 * (torch.sum(torch.log((a/epsilon)*L[:d0])) + norm*epsilon)
    print("KL", KL)

    K = torch.sqrt(2*(KL+np.log(2*n_true/delta))/(n_true-1))
    
    bound = b1 + torch.sqrt(b1*K) + K

    return bound 





def list_to_vec(param_list):

    '''
    transfer a iterable (can be tuple or list) of tensors to a tensor of shape (num_param, )
    
    gradient can pass through this operation.

    if param_list is leaf variable, p_vector is not a leaf variable
    '''

    cnt = 0
    for p in param_list:
        if cnt == 0:
            p_vector = p.contiguous().view(-1)
        else:
            p_vector = torch.cat([p_vector, p.contiguous().view(-1)])
        cnt += 1

    return p_vector





def vec_to_list(p_vector, model):

    '''
    transfer a tensor of shape (num_params, ) to a list of tensors according to the shape of parameters in a model

    gradient can pass through this operation.

    if param_list is leaf variable, p_vector is not a leaf variable
    '''

    p_list = []
    idx = 0
    for param in model.parameters():
        num = param.data.numel()
        a = p_vector[idx : idx + num].clone()
        p_list.append(a.view(param.data.shape))
        idx += num

    return p_list









def lr_scheduler(lr_list, div_list, optimizer, epoch):

    '''
    lr_list: list of learning rate
    div_list: list of stages to change the learning rate
    epoch: current epoch 
    '''

    for div, lr in zip(div_list, lr_list):
        if epoch >= div:
            for g in optimizer.param_groups:
                g['lr'] = lr

            












# def del_attr(obj, names):
#     if len(names) == 1:
#         delattr(obj, names[0])
#     else:
#         del_attr(getattr(obj, names[0]), names[1:])

# def set_attr(obj, names, val):
#     print(names)
#     if len(names) == 1:
#         setattr(obj, names[0], val)
#     else:
#         set_attr(getattr(obj, names[0]), names[1:], val)

# def make_functional(mod):
#     orig_params = tuple(mod.parameters())
#     # Remove all the parameters in the model
#     names = []
#     for name, p in list(mod.named_parameters()):
#         del_attr(mod, name.split("."))
#         names.append(name)

#     return orig_params, names

# def load_weights(mod, names, params):
#     for name, p in zip(names, params):
#         set_attr(mod, name.split("."), p)





def del_attr(obj, names):
    '''
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    
    delete the attribute obj.a.b.c
    '''
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])



def set_attr(obj, names, val):

    '''
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    
    set the attribute obj.a.b.c to val

    if obj.a.b.c is nn.Parameter, cannot directly use set_attr, need to first use del_attr
    '''
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)



def get_attr(obj, names):
    '''
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    
    return the value of attribute obj.a.b.c
    '''
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])




def get_names_params(mod):
    '''
    mod: model with nn.Parameters, cannot use functionalized model
    return:
        names_all: a list of all names of mod.paramters, [a1.b1.c1, a2.b2.c2, ...]
        orig_params: tuple of parameters of type nn.Parameter
    '''

    orig_params = tuple(mod.parameters())
    names_all = []
    for name, p in list(mod.named_parameters()):
        names_all.append(name)
    return orig_params, names_all


def make_functional(mod, names_all, param_iter):
    '''
    names_all: list of all names in mod
    param_iter: iterable of parameters, tensor or nn.Parameter

    load param_iter into mod, preserve the type of param_iter
    '''
    for name, p in zip(names_all, param_iter):
        del_attr(mod, name.split("."))
        set_attr(mod, name.split("."), p)


def load_weights(mod, names_all, params):
    for name, p in zip(names_all, params):
        set_attr(mod, name.split("."), p)







def functional2(model, data, targets, criterion, device, param_tuple, names_all):
    '''
    functional of loss w.r.t. param_tuple
    after calling this function, the model becomes functional, does not contain nn.Parameter
    '''
    data, targets = data.to(device), targets.to(device)

    make_functional(model, names_all, param_tuple)
    output = model(data)
    loss = criterion(output, targets)

    return loss







def functional(model, loader, criterion, device, param_tuple):
    param_tuple_ori, names = make_functional(model)
    load_weights(model, names, param_tuple)

    sum_loss = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        loss = criterion(output, targets)
        sum_loss += len(data)*loss

    avg_loss = sum_loss / len(loader.dataset)



    load_weights(model, names, param_tuple_ori)

    return avg_loss



# def functional2(model, data, targets, criterion, device, param_tuple):
#     data, targets = data.to(device), targets.to(device)
#     param_tuple_ori, names = make_functional(model)
#     load_weights(model, names, param_tuple)

#     output = model(data)
#     loss = criterion(output, targets)
#     load_weights(model, names, param_tuple_ori)

#     return loss




def vhp(model, loader, w0_tuple, v_tuple, criterion, device, half = False):

    '''
    model: model with nn.Parameter
    w0_tuple: tuple of weights at which hessian is calculated
    v_tuple: tuple of weights of direction
    if half = Ture, then using half type vectors to calculate the Hv. w0_tuple, v_tuple should have type half. Model, loader don't need to be in half in the input.
    
    return: loss(w0), Hess(w0)*v in format of tuple
            the returned value and u are all in float32 whether half=True or not
    after calling this function, model is still in the format of nn.Parameter
    '''


    model = model.to(device)
    criterion = criterion.to(device)

    if half == True:
        model = model.half()


    value_list, u_list = [], []
    param_tuple_ori, names_all = get_names_params(model)

    for (data, targets) in loader:
        if half == True:   
            data = data.half()
        def f(*param_tuple):           
            loss = functional2(model, data, targets, criterion, device, param_tuple, names_all)
            return loss
        value, u = torch.autograd.functional.vhp(f, w0_tuple, v_tuple)
        value_list.append(value)
        u_list.append(list_to_vec(list(u)))

    print("vhp iteration",value, norm_2_list(u))

    # load param_tuple_ori back to model, with type nn.Parameters
    load_weights(model, names_all, param_tuple_ori)
        

    value, u = sum(value_list) / len(value_list), sum(u_list) / len(u_list)
    value = value.float()
    u = u.float()

    u = tuple(vec_to_list(u, model))


    return value, u
    











def hess_lanczo(model, k, loader, criterion, device, half=False):

    '''
    calculate eigen values of hessian using Lanczo's method
    model: trained model with parameters at minima
    k: number of eigen values to calculate
    loader: data used to calculate hessian
    half: calculates vhp in half version, use input in normal version

    return: w: np.array of top eigen values sorted in descending order.
    '''

    model = model.to(device)
    criterion = criterion.to(device)
    beta, alpha = [0], [0]
    param_list = [_.data for _ in model.parameters()]
    direc_list = [torch.randn(_.shape).to(device) for _ in model.parameters()]
    q0 = [torch.zeros(_.shape).to(device) for _ in model.parameters()]
    q1 = [_ / norm_2_list(direc_list) for _ in direc_list]

    if half == True:
        for p, q in zip(param_list, q1):
            p, q = p.half(), q.half()

    for i in range(k):
        print(i)
        _, v = vhp(model, loader, tuple(param_list), tuple(q1), criterion, device)
        v = list(v)
        alpha.append(prod_list(q1, v))

        for (ele_v, ele_q0, ele_q1) in zip(v, q0, q1):
            ele_v.data = ele_v.data - beta[i]*ele_q0.data - alpha[i+1]*ele_q1.data

        beta.append(norm_2_list(v))
        q0 = copy.deepcopy(q1)
        w = [ele_v / beta[i+1] for ele_v in v]
        q1 = copy.deepcopy(w)

    # alpha, beta are diagonals of the tridiagonal matrix
    a = np.array([ele.item() for ele in alpha[1:]])
    b = np.array([ele.item() for ele in beta[1:]])
    w, _ = eigh_tridiagonal(a, b[:-1])
    w = -np.sort(-w)

    return w














def hess_scipy(model, k, loader, criterion, device):
    '''
    calculate eigen values of hessian using scipy.sparse.eigsh
    need to first import eigsh from scipy.sparse
    model: trained model with parameters at minima
    k: number of eigen values to calculate
    loader: data used to calculate hessian
    return: torch.tensor float of eigen values and vectors in cpu, both are sorted in descending order.
    '''

    model = model.to(device)
    criterion = criterion.to(device)
    param_list = [_.data for _ in model.parameters()]
    num_params = sum(param.numel() for param in model.parameters())

    def fnc_LO(q):
        '''
        q: numpy array of shape (num_params, )
        return: numpy array of shape (num_params, ), calculates Hq, H is the hessian.
        '''
        q_list = vec_to_list(torch.tensor(q).float().to(device), model)
        _, v_tuple = vhp(model, loader, tuple(param_list), tuple(q_list), criterion, device)
        v = list_to_vec(v_tuple)
        
        return v.cpu().detach().numpy()

    A = LinearOperator((num_params, num_params), matvec=fnc_LO)
    eigenvalues, eigenvectors = eigsh(A, k, which='LM')

    idx = list(np.flip(eigenvalues.argsort()))
    eigenvalues = eigenvalues[idx]
    eigenvectors = ((eigenvectors.T[idx]).T)

    eigenvalues = torch.tensor(eigenvalues).float()
    eigenvectors = torch.tensor(eigenvectors).float()

    return eigenvalues, eigenvectors













def hess_FIM(model, criterion, loader_FIM, loader_hess, k, device):

    '''
    approximate hessian eigen values and vectors using FIM eigen vectors
    model: trained model
    loader_FIM: loader for FIM calculation, batch size =1 
    loader_hess: loader for hess valculation
    k: number of eigen values
    device: the device for evaluating hess eigen values

    FIM is calculated in cpu
    
    return:
        eig_hess: torch.tensor (num_param, )
        u: torch.tensor (num_params, k)
        both in 'cpu'
    '''

    model = model.to(device)
    criterion = criterion.to(device)
    param_list = [_.data for _ in model.parameters()]
    # num_params = sum(param.numel() for param in model.parameters())

    FIM, L, u = FIM2(model, criterion, loader_FIM, 'cpu',k)

    print("=========== FIM finished!==========")

    list_hess = []

    for i in range(k):

        print('evaluate', i, 'th eigen vector')
        
        q1 = vec_to_list(u[:, i].to(device), model)
        _, v_tuple = vhp(model, loader_hess, tuple(param_list), tuple(q1), criterion, device)
        lbd = torch.norm(list_to_vec(list(v_tuple))).item()
        list_hess.append(lbd)

    eig_hess = torch.tensor(list_hess)
    idx = list(np.flip(eig_hess.numpy().argsort()))
    eig_hess = (eig_hess[idx]).to("cpu")
    u = ((u.T[idx]).T).to("cpu")

    return eig_hess, u

















def proj_FIM_kron(r, kfac, device):

    if len(kfac) == 2:
        outmat, inmat = kfac
        outmat = (outmat + outmat.T) / 2
        inmat = (inmat + inmat.T) / 2
        outmat, inmat = outmat.detach().cpu(), inmat.detach().cpu()

        eo, vo = scipy.linalg.eig(outmat)
        ei, vi = scipy.linalg.eig(inmat)
        eo, vo = torch.tensor(np.real(eo)).to(device), torch.tensor(np.real(vo)).to(device)
        ei, vi = torch.tensor(np.real(ei)).to(device), torch.tensor(np.real(vi)).to(device)

        # C = torch.kron(outmat, inmat)

        e = (torch.kron(eo.contiguous(), ei.contiguous())).reshape(len(outmat), len(inmat))

        outmat, inmat = outmat.to(device), inmat.to(device)
        r = r.to(device)
        rp = vo@r@vi.T

        
 
    if len(kfac) == 1:
        mat = kfac[0]
        mat = (mat + mat.T) / 2
        mat = mat.detach().cpu()
        # C = mat

        e, v = scipy.linalg.eig(mat)
        e, v = torch.tensor(np.real(e)).to(device), torch.tensor(np.real(v)).to(device)
        r = r.to(device)
        mat = mat.to(device)

        rp = v@r



    return rp, e

















# def norm_proj(diff_list, kfac_list):










































     




        







# # prepare dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = 'cpu'


# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# num_true = 50000  # number of samples with true labels used in training
# num_random = 0
# num_approx = 1000
# num_classes = 10
# dataset = "cifar10"
# num_channels = 1 if dataset == "mnist" else 3

# print(num_channels)

# model_name = "wr"
# # args = (10, 30, 0.1, num_classes, num_channels)
# args = (10, num_channels, num_classes, 8, 0.1)
# # args = (num_channels, num_classes, 1000)
# path = create_path(model_name, args, num_true, num_random, dataset)
# print(path)
# mkdir(path)



# model = WideResNet(*args).to(device)


# num_params = sum(param.numel() for param in model.parameters())
# print(num_params)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-3)
# scheduler = CosineAnnealingLR(optimizer,T_max=250, eta_min = 1e-5)






# train_set, train_set_combined, test_set = create_dataset(dataset, num_classes, num_true, num_random, False)

# train_set_approx, _, __ = create_dataset(dataset, num_classes, num_approx, num_random, False)






# train_loader = torch.utils.data.DataLoader(train_set,
#                                           batch_size=1000,
#                                           shuffle = True,
#                                           **kwargs)

# train_loader_combined = torch.utils.data.DataLoader(train_set_combined,
#                                           batch_size=500,
#                                           shuffle = True,
#                                           **kwargs)

# train_loader_approx = torch.utils.data.DataLoader(train_set_approx,
#                                           batch_size=500,
#                                           shuffle = True,
#                                           **kwargs)

# train_loader_FIM = torch.utils.data.DataLoader(train_set_approx,
#                                           batch_size=1,
#                                           shuffle = True,
#                                           **kwargs)

# test_loader = torch.utils.data.DataLoader(test_set,
#                                          batch_size=500,
#                                          shuffle = True,
#                                          **kwargs)







# model = torch.load(path + "model.pt").to(device)
# num_params = sum(param.numel() for param in model.parameters())

# param_tuple = tuple(p.detach() for p in model.parameters())



# v = torch.randn(num_params).to(device)

# v = v / torch.sqrt(torch.sum(v**2))

# v_tuple = tuple(vec_to_list(v, model))



# value, u = vhp(model, train_loader_approx, param_tuple, v_tuple, criterion, device)

# print(value, norm_2_list(u))

















# for (data, targets) in train_loader:

#     ts = time.time()

#     def f(*param_tuple):
        
#         loss = functional2(model, data, targets, criterion, device, param_tuple)

#         return loss

#     _, u = torch.autograd.functional.vhp(f, param_tuple, v_tuple)
#     print(_)
#     print(norm_2_list(u))
#     tn = time.time()

#     print(tn-ts)



# loss1 = f(*v_tuple)
# loss1.backward()
# print(v.grad)




# _, u = torch.autograd.functional.vhp(f, param_tuple, v_tuple)

# print(_)

# print(norm_2_list(u))
















































# class md(nn.Module):

#     def __init__ (self):
#         super().__init__()
#         self.fc = nn.Linear(3, 2, bias=False)

#     def forward(self, x):
#         output = self.fc(x)
#         output = torch.sum(output**2)
#         return output






# a = md()
# # for n,p in a.named_parameters():
# #     print(n, p)



# w0 = torch.tensor([[3,3,3],[3,3,3]]).float()
# x = torch.tensor([[1,1,1]]).float()
# v = torch.tensor([[1,2],[3,4]]).float()
# w = torch.randn((2, 3)).float()
# w.requires_grad = True
# w0_tuple = tuple([w0])
# w_tuple = tuple([w])




# def f(*w_tuple):

#     orig_params, names = make_functional(a)
#     load_weights(a, names, w_tuple)
#     res = a(x)

#     load_weights(a, names, orig_params)

#     return res

# print(f(w0))

# print("-----------------------------------------------")
# print("w_tuple", w_tuple)

# loss = f(w)
# print(loss)
# loss.backward()
# print(w.grad)


# _, u = torch.autograd.functional.vhp(f, w0, w)

# print(_, u)






















































# def pow_reducer(x):
#     return x.pow(3).sum().detach()
# inputs = torch.rand(2, 2)
# v = torch.ones(2, 2)
# _, u = torch.autograd.functional.vhp(pow_reducer, inputs, v)

# print(_, u)































# for (name, param) in model.named_parameters():
    # print(name.split("."))


# print(model.block1.layer.0.conv2)















# a = torch.tensor([0.1,0.2,0.3], requires_grad=True)

# b = nn.Parameter(a)

# c = torch.sum(b*2)
# c.backward()

# print(a.grad)
# print(b.grad)









# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# num_true = 50000  # number of samples with true labels used in training
# num_random = 0
# num_approx = 10000
# num_classes = 10
# dataset = "cifar10"
# num_channels = 1 if dataset == "mnist" else 3


# model_name = "wr"
# args = (10, num_channels, num_classes, 8, 0.1)
# path = create_path(model_name, args, num_true, num_random, dataset)
# print(path)


# model = torch.load(path + "model.pt")

# params, names = make_functional(model)


# def f(*new_params):
#     load_weights(m, names, new_params)
#     out = m(inputs)
#     loss = criterion(out, labels)
#     return loss







# params, names = make_functional(m)
# # Make params regular Tensors instead of nn.Parameter
# params = tuple(p.detach().requires_grad_() for p in params)



# def f(*new_params):
#     load_weights(m, names, new_params)
#     out = m(inputs)
#     loss = criterion(out, labels)
#     return loss










    











# model = torch.load(path + 'model.pt')
# tr_err_combined, tr_loss_combined = val(model, device, train_loader_combined, criterion)
# tr_err_true, tr_loss_true = val(model, device, train_loader, criterion)
# val_err, val_loss = val(model, device, test_loader, criterion)
# print('tr_loss_combined', tr_loss_combined, 'tr_loss_true', tr_loss_true, 'val_loss', val_loss)
# print('tr_err_combined', tr_err_combined, 'tr_err_true', tr_err_true, 'val_err', val_err)

# print(norm_2(model))
























    

        
    































    
    







































# if mode == 'trad':


#     model = torch.load('./results/model.pt')
#     loss = val_grad(model, device, train_loader, criterion)
#     Var = model.parameters()
#     grad1 = grad(loss, Var, create_graph=True)




#     def eval_hessian(loss_grad, model):
#         cnt = 0
#         for g in loss_grad:
#             g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
#             cnt = 1
#         l = g_vector.size(0)
#         print(l)
#         hessian = torch.zeros(l, l)
#         for idx in range(l):
#             grad2rd = grad(g_vector[idx], model.parameters(), create_graph=True)
#             cnt = 0
#             for g in grad2rd:
#                 g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
#                 cnt = 1
#             hessian[idx] = g2
#         return hessian.detach().numpy()

#     hess = eval_hessian(grad1, model)
#     print(hess)
#     torch.save(hess, './results/hess.pt')


#     u, v = eig(hess)
#     u = np.real(u)
#     torch.save(u, './results/eig_hess.pt')


#     u = torch.load('./results/eig_hess.pt')
#     u = -np.sort((-u))


#     plt.plot(np.log(np.abs(u[:500])))
#     plt.show()
#     plt.savefig('./results/eig_hess.png')











































# if mode == 'eva_posterior':


#     value, vector = torch.load('./results/eig_all_hess.pt')
#     value, vector = np.real(value), np.real(vector)
#     idx = np.flip(value.argsort())
#     value = torch.tensor(value[idx]).to(device)
#     vector = torch.tensor((vector.T[idx]).T).to(device)

#     lbd_bar, gap, sec = torch.load('./results/posterior.pt')
#     lbd_bar = torch.tensor(lbd_bar).to(device)*100
#     cov_half = vector@(torch.diag(torch.sqrt(1/lbd_bar))).float()

#     # print(lbd_bar)


#     model = torch.load("./results/model.pt")
#     num_params = sum(param.numel() for param in model.parameters())
#     p_list = [param for param in model.parameters()]
#     mean = list_to_vec(p_list)



#     tr_err_true_sum = 0
#     tr_loss_true_sum = 0
#     val_err_sum = 0
#     val_loss_sum = 0

#     n_total = 300

#     for i in range(n_total):
#         print(i)
#         v = cov_half@(torch.randn(num_params)).to(device)
#         param_tuple = tuple(vec_to_list(v + mean, model))
#         for (p1, p2) in zip(model.parameters(), param_tuple):
#             p1.data = copy.deepcopy(p2)

#         tr_err_true, tr_loss_true = val(model, device, train_loader, criterion)
#         val_err, val_loss = val(model, device, test_loader, criterion)

#         tr_err_true_sum += tr_err_true
#         tr_loss_true_sum += tr_loss_true
#         val_err_sum += val_err
#         val_loss_sum += val_loss
#         print('tr_loss_true', tr_loss_true, 'val_loss', val_loss)
#         print('tr_err_true', tr_err_true, 'val_err', val_err)



#     tr_err_true_avg = tr_err_true_sum / n_total
#     tr_loss_true_avg = tr_loss_true_sum / n_total
#     val_err_avg = val_err_sum / n_total
#     val_loss_avg = val_loss_sum / n_total


#     print('tr_loss_true_avg', tr_loss_true_avg, 'val_loss_avg', val_loss_avg)
#     print('tr_err_true_avg', tr_err_true_avg, 'val_err_avg', val_err_avg)
#     print(sec)































