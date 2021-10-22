import torch
import torch.nn as nn
import torch.nn.functional as F 


class AdverserialNet(nn.Module):
    def __init__(self,n_input,n_output):
        super().__init__()
        self.n_input = n_input 
        self.n_output = n_output 

        self.gen = nn.Sequential()

        self.dis = nn.Sequential()

    def forward(self,x):

        return x 

    def forward_generator(self,x):
        pass
 