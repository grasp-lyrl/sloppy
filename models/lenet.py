import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary








#We are tesing our model on MNIST dataset which is 1x28x28 insize so we have to change C5 layer in FC layer becasue at this stage 
#pixel size would be 16x5x5 so we can not apply kernel of 5x5 

class lenet(nn.Module):
    def __init__(self):
        super(lenet,self).__init__()
        #Here, we are plementing those layers which are having learnable parameters.
        #Start implementation of Layer 1 (C1) which has 6 kernels of size 5x5 with padding 0 and stride 1
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5))
        
        #Start implementation of Layer 3 (C3) which has 16 kernels of size 5x5 with padding 0 and stride 1
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = (5,5))
        
        #Start implementation of Layer 5 (C5) which is basically flattening the data 
            
        self.L1 = nn.Linear(256, 120)
        
        #Start implementation of Layer 6 (F6) which has 85 Linear Neurons and input of 120
        
        self.L2 = nn.Linear(120,84)
        
        #Start implementation of Layer 7 (F7) which has 10 Linear Neurons and input of 84
        
        self.L3 = nn.Linear(84,10)
        
        #We have used pooling of size 2 and stride 2 in this architecture 
        
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        self.act = nn.Tanh()
        
    #Now we will implement forward function to produce entire flow of the architecture.
    
    def forward(self,x):
        x = self.conv1(x)
        #We have used tanh as an activation function in this architecture so we will use tanh at all layers excluding F7.
        x = self.act(x)
        #Now this will be passed from pooling 
        x = self.pool(x)
        #Next stage is convolution
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        #next we will pass from conv3, here we will not pass data from pooling as per Architecture 
        
        
        #Now the data should be flaten and it would be passed from FC layers. 
        x = x.view(x.size()[0], -1)
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        x = self.L3(x)
        
        return x
    

