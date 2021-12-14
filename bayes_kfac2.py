from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets
from models.fc import Network
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from torch.autograd import Variable, grad
from numpy.linalg import eig as eig
from torch.distributions.multivariate_normal import MultivariateNormal
from dataset import *
from functions import *
from models.wide_resnet_1 import WideResNet

from models.fc import Network
from utils import *
import time




# isotropic prior + moving FIM posterior


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_device", type=str,
                    default='cuda:0',
                    help="gpu device")

parser.add_argument("--num_neurons", type=int,
                    default=600,
                    help="number of neurons of fc net")

parser.add_argument("--num_layers", type=int,
                    default=2,
                    help="Number of layers of fc net")

arg = parser.parse_args()

# prepare dataset
device = torch.device(arg.gpu_device if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# dataset
num_true = 55000
num_prior = 5000
num_random = 0
num_approx = 10000
num_classes = 2
dataset = "mnist"
num_inplanes = 1 if dataset == "mnist" else 3

# fully connected
num_neurons = arg.num_neurons
num_layers = arg.num_layers

model_name = "fc"
args = (num_classes, num_layers, num_neurons)
path = create_path(model_name, args, num_true, num_random, dataset)
print(path)
mkdir(path)








train_set, train_set_prior, train_set_approx, test_set = create_mnist(num_classes, num_true, num_prior, num_approx)


train_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=1100,
                                          shuffle = True,
                                          **kwargs)

train_loader_approx = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=len(train_set_approx.data),
                                          shuffle = True,
                                          **kwargs)

train_loader_FIM = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=1,
                                          shuffle = True,
                                          **kwargs)

train_loader_prior = torch.utils.data.DataLoader(train_set_prior,
                                          batch_size=500,
                                          shuffle = True,
                                          **kwargs)

test_loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=500,
                                         shuffle = True,
                                         **kwargs)


































##########################
##### Bayes class ########
##########################






















class bayesian_nn(nn.Module):

    def __init__(self, c, eigspace_list, args, ns=150):
        super().__init__()
        self.w = c(*args).to(device)
        self.mu_std = nn.ModuleList([c(*args).to(device), c(*args).to(device)])
        self.ns, self.args = ns, args
        orig_params_w, names_all_w = get_names_params(self.w)
        self.names_all_w = names_all_w
        self.eigspace_list = eigspace_list
        self.c = c

        # for es in self.eigspace_list:
        #     for a in es:
        #         a.data = a.to(device)



    def forward(self, x):

        ys = []
        for _ in range(self.ns):
            for name, m, v, es in zip(self.names_all_w, list(self.mu_std[0].parameters()), list(self.mu_std[1].parameters()), self.eigspace_list):
                r = torch.randn_like(m).mul(torch.sqrt(torch.exp(2*v)))
                rp = proj_FIM_kron(r, es)
                del_attr(self.w, name.split("."))
                set_attr(self.w, name.split("."), rp+m)  


            y = self.w(x)
            ys.append(y)

        self.w = self.c(*args).to(device)

        return torch.stack(ys)















###########################
####### functions #########
###########################





def faclist_to_device(fac_list):
    for fac in fac_list:
        for f in fac:
            f.data = f.to(device)





def proj_FIM_kron(r, eigspace):

    '''
    input: 
        r: tensor(dout, din), 
        eigspace: kfac eig space of a layer of size (dout, din)
    return: eigspace of kron @ r
    '''

    if len(eigspace) == 2:
        vo, vi = eigspace
        rp = vo@r@vi.T 

    if len(eigspace) == 1:
        v = eigspace[0]
        rp = v@r
    
    return rp




def proj_norm(d, eigspace, eigval, tuple = False):
    '''
    input:
        d: tensor(dout, din)
        eigspace: kfac eig space of a layer of size (dout, din), list of tuples if tuple  = True, tnesor of the same size as parameter if tuple = False
        eigval: kfac eigval of a layer of size (dout, din), list of tuples
    return:
        d@(A kron B)@d.T
    '''
    if tuple:
        if len(eigspace) == 2:
            vo, vi = eigspace
            eo, ei = eigval
            A = vo*eo@vo.T
            B = vi*ei@vi.T
            dpn = torch.sum(d*(A@d@B))
        if len(eigspace) == 1:
            v = eigspace[0]
            e = eigval[0]
            A = v*e@v.T
            dpn = torch.sum(d*(A@d))

    if not tuple:
        if len(eigspace) == 2:
            vo, vi = eigspace
            dp = (vo.T@d@vi)*torch.sqrt(eigval)
            dpn = torch.sum(dp**2)
        if len(eigspace) == 1:
            v = eigspace[0]
            dp = (v.T@d)*torch.sqrt(eigval)
            dpn = torch.sum(dp**2)

    return dpn





def ev_trans(ev_tuple_list):
    ev_list = []
    for ev_tuple in ev_tuple_list:
        ev = torch.outer(ev_tuple[0], ev_tuple[1]) if len(ev_tuple) == 2 else ev_tuple[0]
        ev_list.append(ev)

    return ev_list








def sec(model, model_init, rho, num_samples, device, b = 10, c = 0.05, delta = 0.025):

    epsilon = torch.exp(2*rho)
    pi = torch.tensor(np.pi)
    kl_1, kl_2 = 0, 0

    for m0, m, xi in zip(
        model_init.parameters(), 
        model.mu_std[0].parameters(), 
        model.mu_std[1].parameters(), 
    ):

        q = torch.exp(2*xi)
        p = epsilon
        kl_1 += (1/p) * torch.sum((m0-m)**2)
        kl_2 += torch.sum(q / p) + torch.sum(torch.log(p / q))
        kl_2 += -m.numel()

    kl = (kl_1 + kl_2) / 2
    penalty = 2*torch.log(2*torch.abs(b*torch.log(c / epsilon))) + torch.log(pi**2*num_samples / 6*delta)


    sec = torch.sqrt((kl + penalty) / (2*(num_samples - 1)))

    return sec, kl, kl_1, kl_2, penalty



    
    
    
    
    
    
    
    
    
    
    
    
def train(model,model_init, num_samples, device, train_loader, criterion, optimizer, rho, num_classes):

    model.train()
    for (data, targets) in train_loader:

        loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho ,num_samples, device)
        

        data, targets = data.to(device), targets.to(device)
        output = model(data)
        output = output.reshape(model.ns * len(data), num_classes)
        targets = targets.repeat(model.ns)
        loss = criterion(output, targets) * (1/np.log(2))

        optimizer.zero_grad()
        (loss2 + loss).backward()

        # loss.backward()
        # break
        optimizer.step()

    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())





def val(model, device, val_loader, criterion, num_classes):
    
    model.eval()
    sum_loss, sum_corr = 0, 0

    

    for (data, targets) in val_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        output = output.reshape(model.ns * len(data), num_classes)
        targets = targets.repeat(model.ns)
        loss = criterion(output, targets)
        pred = output.max(1)[1]
        sum_loss += loss.item()
        sum_corr += pred.eq(targets).sum().item() / len(targets)

    err_avg = 1 - (sum_corr/len(val_loader))
    loss_avg = sum_loss / len(val_loader)
    


    return err_avg, loss_avg






def val_d(model, device, val_loader, criterion, num_classes):
    
    model.eval()
    sum_loss, sum_corr = 0, 0

    

    for (data, targets) in val_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        loss = criterion(output, targets)
        pred = output.max(1)[1]
        sum_loss += loss.item()
        sum_corr += pred.eq(targets).sum().item() / len(targets)

    err_avg = 1 - (sum_corr/len(val_loader))
    loss_avg = sum_loss / len(val_loader)
    
    return err_avg, loss_avg





def initial(model, model_trained):

    state_dict = model_trained.state_dict()
    model.mu_std[0].load_state_dict(state_dict)
    model.w.load_state_dict(state_dict)

    for p in model.mu_std[1].parameters():
        p.data = 0.5*torch.log(torch.ones(p.shape) / 50).to(device)




def initial1(model, model_trained, ev_list, rho):

    epsilon = torch.exp(2*rho)

    state_dict = model_trained.state_dict()
    model.mu_std[0].load_state_dict(state_dict)
    model.w.load_state_dict(state_dict)
    for p, ev_tuple in zip(model.mu_std[1].parameters(), ev_list):
        ev = torch.outer(ev_tuple[0], ev_tuple[1]) if len(ev_tuple) == 2 else ev_tuple[0]
        p.data = 0.5*torch.log((ev + epsilon) / 10).to(device)





































#########################
###### trainining #######
#########################




def main():


    c = Network


    rho = torch.tensor(-3).to(device).float()

    kfacp_list, esp_list, evp_list = torch.load(path + "kfac_all_init.pt")
    faclist_to_device(esp_list)
    faclist_to_device(evp_list)



    model = bayesian_nn(c, esp_list ,args)
    model_trained = Network(*args)
    model_trained.load_state_dict(torch.load(path + "model.pt"))
    model_trained = model_trained.to(device)
    model_init = Network(*args)
    model_init.load_state_dict(torch.load(path + "model_init.pt"))
    model_init = model_init.to(device)

    num_params = sum(p.numel() for p in model_trained.parameters())
    print(num_params)

    # initialization

    initial1(model, model_trained, evp_list, rho)

    # model_state, rho = torch.load(path + "model_bayes_kfac2.pt")
    # model.load_state_dict(model_state)
    # model = model.to(device)
    # rho = torch.tensor(rho).to(device)



    epochs = 250
    rho.requires_grad = True
    param = list(model.parameters()) + [rho]
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(param, lr = 1e-3, weight_decay=0)



    kfac_list = FIM_kfac(model.mu_std[0], train_loader_approx, device, mc = 10, mode="kfra", empirical=True)
    es_list_new, _ = eigspace_FIM_kron(kfac_list)
    faclist_to_device(es_list_new)
    model.eigspace_list = es_list_new



    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)



    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)


    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho", rho.item())


    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)






    # training loop


    for epoch in range(epochs):
        if epoch >= 100:
            if epoch%5 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.95

        time_start = time.time()

        kfac_list = FIM_kfac(model.mu_std[0], train_loader_approx, device, mc = 10, mode="kfra", empirical=True)
        es_list_new, _ = eigspace_FIM_kron(kfac_list)
        faclist_to_device(es_list_new)
        model.eigspace_list = es_list_new
        train(model,model_init, num_true, device, train_loader, criterion, optimizer, rho, num_classes)
        
        time_end = time.time()

        if epoch%20 == 0:

            val_err, val_loss = val(model,device, test_loader, criterion, num_classes)
            train_err, train_loss = val(model,device, train_loader, criterion, num_classes)

            loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
            bd1 = train_err + loss2
            bd2 = train_loss * (1/np.log(2)) + loss2



            print('epoch', epoch)
            print('train_err, train_loss', train_err, train_loss)
            print('val_err, val_loss', val_err, val_loss)
            print('bd1, bd2, rho', bd1.item(), bd2.item(), rho.item())
            print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
            for g in optimizer.param_groups:
                    print(g['lr'])
            print('time', time_end - time_start)

            if epoch != 0:
                torch.save((model.state_dict(), rho.item()), path + "model_bayes_kfac2.pt")







    # statistics analysis

    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)

    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)

    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho", rho.item())


    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)


    stat1 = dict({"dt": dt, "bt":bt, "dv":dv, "bv":bv, "bd":bd ,"loss2":loss2.item(), "kl":kl.item(), "kl_1":kl_1.item(), "kl_2":kl_2.item(), "rho":rho.item()})


    print(stat1)


    kfac_list = FIM_kfac(model.mu_std[0], train_loader_approx,  device, mc=10, mode="kfra", empirical=True)
    es_list_new, ev_list_new = eigspace_FIM_kron(kfac_list)

    ev_list_new = ev_trans(ev_list_new)
    ev_new_vec = list_to_vec(ev_list_new).detach().cpu()
    idx = list(np.flip(ev_new_vec.numpy().argsort()))
    ev_vec = ev_new_vec[idx].detach()   # eigen value predicted by kfac in descent order
    xi_vec = list_to_vec(model.mu_std[1].parameters()).detach().cpu()
    xi_vec = xi_vec[idx]
    var_vec = torch.exp(2*xi_vec)
    inv_vec = 1 / var_vec   # inverse of variance predicted by kfac in the same order as eigen values

    stat2 = dict({"ev_new_vec":ev_vec, "inv_vec" : inv_vec, "var_vec": var_vec})
    print(stat2)

    torch.save((stat1, stat2), path + "stat_bayes_kfac2.pt")








main()


























