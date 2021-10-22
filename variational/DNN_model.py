import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim,out_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, out_dim)
        self.fc22 = nn.Linear(hidden_dim, out_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # print(hidden)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_ = self.fc21(hidden)
        var_ = self.softplus(self.fc22(hidden))
        return loc_,var_

class Encoder(nn.Module):
    def __init__(self,data_dim, hidden_dim,out_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc2 = nn.Linear(data_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.act(self.fc2(x))
        out = torch.softmax(self.fc4(hidden),dim=1)
        return out


class Params(nn.Module):
    def __init__(self,out_dim):
        super().__init__()
        # setup the two linear transformations used
        self.weight = torch.nn.Parameter(data=torch.Tensor([1.] * out_dim), requires_grad=True)
        
    def forward(self):
        return torch.nn.functional.relu(self.weight)