import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets



def two_class(dataset):
    targets = dataset.targets
    m = torch.max(targets)
    targets = torch.where(targets<=torch.div(m, 2, rounding_mode='floor'), 0, 1)
    dataset.targets = targets





def sample_balance(dataset, n_balance):

    '''
    dataset: dataset to subsample from.
    n_balance: number of sample to use in total 
    return: balanced sampled data and targets

    '''

    data_list = []
    targets_list = []
    d = torch.max(dataset.targets)
    
    for i in range(d+1):
        idx = dataset.targets == i
        data_list.append(dataset.data[idx][:n_balance//(d+1)])
        targets_list.append(dataset.targets[idx][:n_balance//(d+1)])
    
    data = torch.cat(data_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    return data, targets





def split_balance(dataset, n1, n2):

    data1_list, targets1_list, data2_list, targets2_list = [], [], [], []
    d = torch.max(dataset.targets)

    for i in range(d+1):
        idx = dataset.targets == i
        data1_list.append(dataset.data[idx][:n1//(d+1)])
        targets1_list.append(dataset.targets[idx][:n1//(d+1)])
        data2_list.append(dataset.data[idx][-n2//(d+1):])
        targets2_list.append(dataset.targets[idx][-n2//(d+1):])

    data1 = torch.cat(data1_list, dim=0)
    targets1 = torch.cat(targets1_list, dim=0)
    data2 = torch.cat(data2_list, dim=0)
    targets2 = torch.cat(targets2_list, dim=0)

    return data1, targets1, data2, targets2





def split(dataset, n1, n2):

    data1 = dataset.data[:n1]
    targets1 = dataset.targets[:n1]
    data2 = dataset.data[-n2:]
    targets2 = dataset.targets[-n2:]

    return data1, targets1, data2, targets2












def sample_combined(dataset, n_true, n_random, balance = True):


    '''
    dataset: dataset to subsample from.
    n_true: number of samples with true labels
    n_ramdom: number of samples with random labels 
    return: data and targets with combined true and random data

    '''
    
    d = torch.max(dataset.targets)
    
    if balance == True:

        data_true_list = []
        targets_true_list = []
        data_random_list = []

        for i in range(d+1):
            idx = dataset.targets == i
            data_true_list.append(dataset.data[idx][:n_true//(d+1)])
            targets_true_list.append(dataset.targets[idx][:n_true//(d+1)])
            data_random_list.append(dataset.data[idx][-n_random//(d+1) :])

        data_true = torch.cat(data_true_list, dim=0)
        targets_true = torch.cat(targets_true_list, dim=0)
        data_random = torch.cat(data_random_list, dim=0)
        targets_random = torch.randint(0, d+1, (n_random,))


        data_combined = torch.cat((data_true, data_random), dim=0)
        targets_combined = torch.cat((targets_true, targets_random), dim=0)


    if balance == False:

        data_true = dataset.data[:n_true]
        targets_true = dataset.targets[:n_true]
        data_random = dataset.data[-n_random:]
        targets_random = torch.randint(0, d+1, (n_random,))

        data_combined = torch.cat((data_true, data_random), dim=0)
        targets_combined = torch.cat((targets_true, targets_random), dim=0)

        


    if n_random == 0:

        data_combined = data_true
        targets_combined = targets_true

    return data_true, targets_true, data_combined, targets_combined








def sample_combined2(dataset, n_true, n_random, n_approx):

    d = torch.max(dataset.targets)
    data_true = dataset.data[:n_true]
    targets_true = dataset.targets[:n_true]
    data_random = dataset.data[-n_random:]
    targets_random = torch.randint(0, d+1, (n_random,))

    data_combined = torch.cat((data_true, data_random), dim=0)
    targets_combined = torch.cat((targets_true, targets_random), dim=0)

    n_true_approx = int(n_true * (n_approx / (n_true + n_random)))
    n_random_approx = int(n_random * (n_approx / (n_true + n_random)))

    data_true_approx = dataset.data[:n_true_approx]
    targets_true_approx = dataset.targets[:n_true_approx]
    data_random_approx = dataset.data[-n_random_approx:]
    targets_random_approx = torch.randint(0, d+1, (n_random_approx,))

    data_combined_approx = torch.cat((data_true_approx, data_random_approx), dim=0)
    targets_combined_approx = torch.cat((targets_true_approx, targets_random_approx), dim=0)

    if n_random == 0:
        data_combined = data_true
        targets_combined = targets_true
        data_combined_approx = data_true_approx
        targets_combined_approx = targets_true_approx

    return data_combined, targets_combined, data_combined_approx, targets_combined_approx









def create_dataset(dataset, num_classes, num_true, num_random, balance =False):


    if dataset == 'mnist':

        mean = [0.131]
        std = [0.289]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.MNIST('./data',
                            train=True,
                            download=True,
                            transform=transform)

        
        train_set_combined = datasets.MNIST('./data',
                            train=True,
                            download=True,
                            transform=transform)
                            
        test_set = datasets.MNIST('./data',
                            train=False,
                            download=True,
                            transform=transform)

        test_balance = int(len(test_set.data) * 0.8)




    if dataset == 'cifar10':

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            normalize, 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4)
            ])

        transform2 = transforms.Compose([transforms.ToTensor(), normalize])
        

        train_set = datasets.CIFAR10('./data',
                            train=True,
                            download=True,
                            transform=transform1)

        train_set_combined = datasets.CIFAR10('./data',
                            train=True,
                            download=True,
                            transform=transform1)

        test_set = datasets.CIFAR10('./data',
                            train=False,
                            download=True,
                            transform=transform2)


        train_set.data, train_set.targets = torch.tensor(train_set.data), torch.tensor(train_set.targets)
        train_set_combined.data, train_set_combined.targets = torch.tensor(train_set_combined.data), torch.tensor(train_set_combined.targets)
        test_set.data, test_set.targets = torch.tensor(test_set.data), torch.tensor(test_set.targets)

        test_balance = len(test_set.data)

        



    if dataset == 'cifar100':

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_set = datasets.CIFAR100('./data',
                            train=True,
                            download=True,
                            transform=transform)

        train_set_combined = datasets.CIFAR100('./data',
                            train=True,
                            download=True,
                            transform=transform)

        test_set = datasets.CIFAR100('./data',
                            train=False,
                            download=True,
                            transform=transform)
    

        train_set.data, train_set.targets = torch.tensor(train_set.data), torch.tensor(train_set.targets)
        train_set_combined.data, train_set_combined.targets = torch.tensor(train_set_combined.data), torch.tensor(train_set_combined.targets)
        test_set.data, test_set.targets = torch.tensor(test_set.data), torch.tensor(test_set.targets)

        test_balance = len(test_set.data)


    
    data_true, targets_true, data_combined, targets_combined = sample_combined(train_set, num_true, num_random, balance)

    train_set.data = data_true
    train_set.targets = targets_true
    train_set_combined.data = data_combined
    train_set_combined.targets = targets_combined


    

    if balance == True:

        data_test, targets_test = sample_balance(test_set, test_balance)

        test_set.data = data_test
        test_set.targets = targets_test



    if num_classes == 2:

        two_class(train_set)
        two_class(train_set_combined)
        two_class(test_set)


    if dataset != "mnist":
        train_set.data = train_set.data.numpy()
        train_set_combined.data = train_set_combined.data.numpy()
        test_set.data = test_set.data.numpy()




    return train_set, train_set_combined, test_set














def create_cifar(num_classes, n1, n2, num_approx):

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean, std)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        normalize, 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4)
        ])

    transform2 = transforms.Compose([transforms.ToTensor(), normalize])
    

    train_set = datasets.CIFAR10('./data',
                        train=True,
                        download=True,
                        transform=transform1)

    train_set_approx = datasets.CIFAR10('./data',
                        train=True,
                        download=True,
                        transform=transform2)


    train_set_prior = datasets.CIFAR10('./data',
                        train=True,
                        download=True,
                        transform=transform2)

    test_set = datasets.CIFAR10('./data',
                        train=False,
                        download=True,
                        transform=transform2)


    train_set.data, train_set.targets = torch.tensor(train_set.data), torch.tensor(train_set.targets)
    train_set_prior.data, train_set_prior.targets = torch.tensor(train_set_prior.data), torch.tensor(train_set_prior.targets)
    test_set.data, test_set.targets = torch.tensor(test_set.data), torch.tensor(test_set.targets)

    data1, targets1, data2, targets2 = split_balance(train_set, n1, n2)

    data3, targets3, _, _ = split_balance(train_set, num_approx, 0)

    train_set.data = data1
    train_set.targets = targets1
    train_set_prior.data = data2
    train_set_prior.targets = targets2
    train_set_approx.data = data3
    train_set_approx.targets = targets3

    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_prior)
        two_class(test_set)
        two_class(train_set_approx)

    train_set.data = train_set.data.numpy()
    train_set_prior.data = train_set_prior.data.numpy()
    train_set_approx.data = train_set_approx.data.numpy()
    test_set.data = test_set.data.numpy()    

    return train_set, train_set_prior, train_set_approx, test_set









def create_mnist(num_classes, n1, n2, num_approx):
    mean = [0.131]
    std = [0.289]
    # normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)

    train_set_approx = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)


    train_set_prior = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)

    test_set = datasets.MNIST('./data',
                        train=False,
                        download=True,
                        transform=transform)


    data1, targets1, data2, targets2 = split(train_set, n1, n2)

    data3, targets3, _, _ = split(train_set, num_approx, 0)

    train_set.data = data1
    train_set.targets = targets1
    train_set_prior.data = data2
    train_set_prior.targets = targets2
    train_set_approx.data = data3
    train_set_approx.targets = targets3


    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_prior)
        two_class(test_set)
        two_class(train_set_approx)


    return train_set, train_set_prior, train_set_approx, test_set





def create_mnist_random(num_classes, num_true, num_random, num_approx):
    mean = [0.131]
    std = [0.289]
    # normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)

    train_set_combined = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)

    train_set_approx = datasets.MNIST('./data',
                        train=True,
                        download=True,
                        transform=transform)
    test_set = datasets.MNIST('./data',
                        train=False,
                        download=True,
                        transform=transform)

    data_combined, targets_combined, data_combined_approx, targets_combined_approx = sample_combined2(train_set, num_true, num_random, num_approx)  

    train_set_combined.data = data_combined
    train_set_combined.targets = targets_combined
    train_set_approx.data = data_combined_approx
    train_set_approx.targets = targets_combined_approx

    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_combined)
        two_class(train_set_approx)
        two_class(test_set)

    return train_set, train_set_combined, train_set_approx, test_set





































































































# # prepare dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
# normalize = transforms.Normalize(mean = [0.131], std = [0.289])
# transform = transforms.Compose([transforms.ToTensor(), normalize])


# n_random = 0  # number of samples with random labels used in training
# n_true = 40000  # number of samples with true labels used in training
# num_classes = 2
# num_neurons = 1000
# num_approx = 10000

# model_name = "fc2_" + str(num_neurons)
# path = create_path(model_name, n_random, n_true)
# mkdir(path)

# print(path)
# model = Network(nchannels = 1, nclasses = num_classes, num_neurons=num_neurons)
# model = model.to(device)


# model = wide_resnet_t(10, 2, 0.5, 2, 1)



# print(norm_2(model))



# num_params = sum(param.numel() for param in model.parameters())
# print('num parameters:', num_params)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr = 5e-3, momentum = 0.9, nesterov = True)


# train_set = datasets.MNIST('./data',
#                           train=True,
#                           download=True,
#                           transform=transform)

# train_set_combined = datasets.MNIST('./data',
#                           train=True,
#                           download=True,
#                           transform=transform)

# test_set = datasets.MNIST('./data',
#                          train=False,
#                          download=True,
#                          transform=transform)

# train_set_approx = datasets.MNIST('./data',
#                           train=True,
#                           download=True,
#                           transform=transform)







# def sample_balance(dataset, n_balance, num_classes = 2):

#     '''
#     dataset: dataset to subsample from.
#     n_balance: number of sample to use in total 
#     return: balanced sampled data and targets

#     '''

#     data_list = []
#     targets_list = []
    

#     for i in range(10):
#         idx = dataset.targets == i
#         data_list.append(dataset.data[idx][:n_balance//10])
#         targets_list.append(dataset.targets[idx][:n_balance//10])
    
#     data = torch.cat(data_list, dim=0)
#     targets = torch.cat(targets_list, dim=0)
    
#     if num_classes == 2:
#         targets = torch.where(targets<5, 0, 1)

#     return data, targets


# def sample_combined(dataset, n_true, n_random, num_classes = 2):


#     '''
#     dataset: dataset to subsample from.
#     n_true: number of samples with true labels
#     n_ramdom: number of samples with random labels 
#     return: data and targets with combined true and random data

#     '''

#     data_true_list = []
#     targets_true_list = []
#     data_random_list = []
#     for i in range(10):
#         idx = dataset.targets == i
#         data_true_list.append(dataset.data[idx][:n_true//10])
#         targets_true_list.append(dataset.targets[idx][:n_true//10])
#         data_random_list.append(dataset.data[idx][n_true//10: n_true//10 + n_random//10])

#     data_true = torch.cat(data_true_list, dim=0)
#     targets_true = torch.cat(targets_true_list, dim=0)
#     data_random = torch.cat(data_random_list, dim=0)
#     targets_random = torch.randint(0, 10, (n_random,))


#     data_combined = torch.cat((data_true, data_random), dim=0)
#     targets_combined = torch.cat((targets_true, targets_random), dim=0)

#     if num_classes == 2:
#         targets_true = torch.where(targets_true<5, 0, 1)
#         targets_random = torch.where(targets_random<5, 0, 1)
#         targets_combined = torch.where(targets_combined<5, 0, 1)

    


#     return data_true, targets_true, data_combined, targets_combined




# train_true_data, train_true_targets, train_combined_data, train_combined_targets = sample_combined(train_set, n_true, n_random, num_classes = 2)
# test_data , test_targets = sample_balance(test_set, 8000, num_classes = 2)
# train_data_approx, train_targets_approx = sample_balance(train_set, num_approx, num_classes = 2)



# train_set.data, train_set.targets = train_true_data, train_true_targets
# train_set_combined.data, train_set_combined.targets = train_combined_data, train_combined_targets
# test_set.data, test_set.targets = test_data, test_targets
# train_set_approx.data, train_set_approx.targets = train_data_approx, train_targets_approx



# train_loader = torch.utils.data.DataLoader(train_set,
#                                           batch_size=500,
#                                           shuffle = True,
#                                           **kwargs)


# train_loader_approx = torch.utils.data.DataLoader(train_set_approx,
#                                           batch_size=500,
#                                           shuffle = True,
#                                           **kwargs)

# train_loader_combined = torch.utils.data.DataLoader(train_set_combined,
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

    