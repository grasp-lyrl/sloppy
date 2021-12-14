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
from models.lenet import lenet
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






parser = argparse.ArgumentParser()
parser.add_argument("--gpu_device", type=str,
                    default='cuda:0',
                    help="gpu device")
arg = parser.parse_args()



#######################
##### dataset #########
#######################


# prepare dataset
device = torch.device(arg.gpu_device if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}




## dataset
num_true = 55000  # number of samples with true labels used in training
num_prior = 5000   # num samples used for prior calculation
num_approx = 10000
num_classes = 10
num_random = 0
dataset = "mnist"
num_inplanes = 1 if dataset == "mnist" else 3

# lenet

model_name = "lenet"
args = ()
path = create_path(model_name, args, num_true, num_random, dataset)
print(path)
mkdir(path)
model = lenet(*args).to(device)





train_set, train_set_prior, train_set_approx, test_set = create_mnist(num_classes, num_true, num_prior, num_approx)

train_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=500,
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
                                          batch_size=len(train_set_prior.data),
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

    def __init__(self, c, u ,args, ns=100):
        super().__init__()
        self.w = c(*args).to(device)
        self.mu = c(*args).to(device)
        self.u = u.to(device)
        self.nd = len(self.u.T)
        self.xi = nn.Parameter(torch.empty(self.nd).normal_(0, 0.1).to(device))
        self.xi_c = nn.Parameter(torch.empty(1).normal_(0, 0.1).to(device))
        self.ns, self.args = ns, args
        orig_params_w, names_all_w = get_names_params(self.w)
        self.names_all_w = names_all_w

        self.num_params = sum(param.numel() for param in self.w.parameters())
        self.c = c

        
    def forward(self, x):
        ys = []
        for _ in range(self.ns):
            mean = list_to_vec(self.mu.parameters())
            r = torch.randn_like(mean)
            r1 = self.u.T@r
            r1 = (self.u*torch.sqrt(torch.exp(2*self.xi))) @ r1
            r2_com = self.u @ (self.u.T@r)
            r2 = (r-r2_com)*torch.sqrt(torch.exp(2*self.xi_c))


            param_new_list = vec_to_list(r1+r2+mean, self.mu)

            for name, p in zip(self.names_all_w, param_new_list):
                del_attr(self.w, name.split("."))
                set_attr(self.w, name.split("."), p)


            y = self.w(x)
            ys.append(y)

        self.w = self.c(*args).to(device)
        return torch.stack(ys)







































###########################
####### functions #########
###########################






def sec(model, model_init, rho, rho1, rho2, u, eig_FIM, num_samples, device, b1 = 10, c1 = 0.05, b2 = 10, c2 = 0.05, b = 10, c = 0.05, delta = 0.025):

    u, eig_FIM = u.to(device), eig_FIM.to(device)
    a, epsilon1, epsilon2 = torch.exp(-2*rho), torch.exp(-2*rho1), torch.exp(-2*rho2)
    tau, tau_c = torch.exp(2*model.xi), torch.exp(2*model.xi_c)

    z, z_c = a / (eig_FIM + epsilon1), a / epsilon2 


    pi = torch.tensor(np.pi)


    list_diff = diff_list(list(model.mu.parameters()),list(model_init.parameters()))
    d = list_to_vec(list_diff)
    dp = d@u

    
    kl_1 = (dp*z)@dp + z_c*(d@d-dp@dp)
    kl_2 = torch.sum(z*tau) + (model.num_params - model.nd) * (tau_c * z_c)
    kl_2 += torch.sum(torch.log(1e-12 + 1/(z*tau))) + (model.num_params - model.nd) * torch.log(1e-12 + 1/(z_c*tau_c))
    kl_2 = kl_2-model.num_params
    

    penalty = 6*torch.log(2*torch.abs(b*torch.log(c * a)) + 2*torch.abs(b1*torch.log(c1 / epsilon1)) + 2*torch.abs(b2*torch.log(c2 / epsilon2))) + torch.log(pi**2*num_samples / 6*delta)


    kl = (kl_1 + kl_2) / 2
    sec = torch.sqrt((kl + penalty) / (2*(num_samples - 1))) 

    


    return sec, kl, kl_1, kl_2, penalty














def train(model,model_init, num_samples, device, train_loader, criterion, optimizer, rho, rho1, rho2, u, eig_FIM, num_classes):

    model.train()
    for (data, targets) in train_loader:

        loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, rho1, rho2, u, eig_FIM, num_samples, device)
        

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



    for v, w in zip(model.mu_std[1].parameters(), model_trained.parameters()):
        v.data = 0.5*torch.log(torch.abs(w) / 10)




def initial2(model, model_trained, model_init, num_samples, rho):

    epsilon = torch.exp(-2*rho)

    model.mu_std[0] = model_trained

    for p, w, w0 in zip(model.mu_std[1].parameters(), model_trained.parameters(), model_init.parameters()):
        diff = w-w0
        p.data = 0.5*torch.log(1 / (num_samples*diff + epsilon))



def initial3(model, model_trained):
    state_dict = model_trained.state_dict()
    model.mu.load_state_dict(state_dict)
    model.w.load_state_dict(state_dict)
    model.xi.data = (0.5*torch.log(torch.abs(torch.ones(model.xi.shape)) / 5000)).to(device)
    model.xi_c.data = (0.5*torch.log(torch.abs(torch.ones(model.xi_c.shape)) / 5000)).to(device)









def initial4(model, model_trained, eig_FIM, rho1, rho2, device):
    eig_FIM = eig_FIM.to(device)
    a, epsilon = torch.exp(-2*rho1), torch.exp(-2*rho2)
    z, z_c = (a / eig_FIM)+epsilon, epsilon

    state_dict = model_trained.state_dict()
    model.mu.load_state_dict(state_dict)
    model.w.load_state_dict(state_dict)
    model.xi.data = (0.5*torch.log(torch.abs(1 / z) / 1000)).to(device)
    model.xi_c.data = (0.5*torch.log(torch.abs(1 / z_c) / 1000)).to(device)











































#########################
###### trainining #######
#########################






def main():

    c = lenet


    rho, rho1, rho2 = torch.tensor(-3).to(device).float(), torch.tensor(-0.5).to(device).float(), torch.tensor(-0.5).to(device).float()

    FIM, eig_FIM, u = torch.load(path + "FIM_true_init.pt")

    u, eig_FIM = u[:,:300], eig_FIM[:300]
    print(u.shape, eig_FIM.shape)

    # print("eig_FIM",eig_FIM)

    model = bayesian_nn(c,u,args)


    model_trained = lenet(*args)
    model_trained.load_state_dict(torch.load(path + "model.pt"))
    model_trained = model_trained.to(device)
    model_init = lenet(*args)
    model_init.load_state_dict(torch.load(path + "model_init.pt"))
    model_init = model_init.to(device)



    num_params = sum(p.numel() for p in model_trained.parameters())
    print(num_params)



    initial3(model, model_trained)




    model_state, rho, rho1, rho2 = torch.load(path + "model_bayes_proj_prior.pt", map_location='cpu')
    model.load_state_dict(model_state)
    model = model.to(device)
    rho, rho1, rho2 = torch.tensor(rho).to(device), torch.tensor(rho1).to(device), torch.tensor(rho2).to(device)



    epochs = 100
    rho.requires_grad, rho1.requires_grad, rho2.requires_grad = True, True, True
    param = list(model.parameters()) + [rho, rho1, rho2]
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(param, lr = 1e-3, weight_decay=0)







    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)

    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)



    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, rho1, rho2, u, eig_FIM, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho, rho1, rho2", rho.item(), rho1.item(), rho2.item())

    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)

















    # train loop

    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*0.985


        time_start = time.time()

        train(model,model_init, num_true, device, train_loader, criterion, optimizer, rho, rho1, rho2, u, eig_FIM, num_classes)
        time_end = time.time()

        if epoch%20 == 0:
            val_err, val_loss = val(model,device, test_loader, criterion, num_classes)
            train_err, train_loss = val(model,device, train_loader, criterion, num_classes)

            loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, rho1, rho2, u, eig_FIM, num_true, device)
            bd1 = train_err + loss2
            bd2 = train_loss * (1/np.log(2)) + loss2



            print('epoch', epoch)
            print('train_err, train_loss', train_err, train_loss)
            print('val_err, val_loss', val_err, val_loss)
            print('bd1, bd2, rho, rho1, rho2', bd1.item(), bd2.item(), rho.item(), rho1.item(), rho2.item())
            print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
            for g in optimizer.param_groups:
                    print(g['lr'])
            print('time', time_end - time_start)

            torch.save((model.state_dict(), rho.item(), rho1.item(), rho2.item()), path + "model_bayes_proj_prior.pt")



    # statistics analysis

    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)

    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)



    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, rho1, rho2, u, eig_FIM, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho, rho1, rho2", rho.item(), rho1.item(), rho2.item())

    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)

    stat1 = dict({"dt": dt, "bt":bt, "dv":dv, "bv":bv, "bd":bd ,"loss2":loss2.item(), "kl":kl.item(), "kl_1":kl_1.item(), "kl_2":kl_2.item(), "rho":rho.item(), "rho1":rho1.item(), "rho2":rho2.item()})

    print(stat1)

    torch.save(stat1, path + "stat_proj_prior.pt")




main()

































