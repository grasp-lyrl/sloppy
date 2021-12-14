import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

class Network(nn.Module):
    def __init__(self, num_classes, num_layers, num_neurons):
        super(Network, self).__init__()
        layers = []
        layers.append(nn.Linear(28*28, num_neurons))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(num_neurons, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class Network1(nn.Module):
    def __init__(self, num_classes, num_layers, num_neurons):
        super(Network1, self).__init__()
        layers = []
        layers.append(nn.Linear(200, num_neurons))
        layers.append(nn.Sigmoid())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(num_neurons, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x







# net = Network(2, 2, 300)
# print(net.classifier)

















