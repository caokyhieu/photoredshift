import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm,trange
import numpy as np
import matplotlib.pyplot as plt
class SimModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size,1)

    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x

    def sample(self,x):
        # _trange = trange(28*28-1)
        result = [x[:,:,0:1]]
        with torch.no_grad():
            _x, hidden = self.lstm(x)
            _x = torch.sigmoid(self.fc(_x))
            p = torch.from_numpy(np.repeat(np.array([[[0,0]]]),x.size()[0],axis=0)).to(torch.float).to(_x.device)
            _x = torch.cat([_x,p],dim=2)
            

            for i in range(1,28*28):
                _x, hidden = self.lstm(_x,hidden)
                _x = torch.sigmoid(self.fc(_x))
                result.append(_x)
                p = torch.from_numpy(np.repeat(np.array([[[i//28,i%28]]]),x.size()[0],axis=0)).to(torch.float).to(_x.device)
                _x = torch.cat([_x,p],dim=2)
            result = torch.cat(result,dim=-2)
            # print(result.size())
            result = result.reshape(-1,28,28,1).squeeze()
            result = result.cpu().detach().numpy()*255
            # print(result.device)
            fig,ax = plt.subplots(4,4,figsize=(6,10))
            fig.subplots_adjust(wspace=0.01,hspace=0.01)
            for i in range(16):
                r = i//4
                c = i - r*4
                # ax[r,c].grid('on', linestyle='--')
                ax[r,c].axis('off')
                ax[r,c].imshow(result[i], cmap='gray')
            # fig.tight_layout()
            fig.savefig('random_digit.png')


transform = transforms.Compose(
    [
        transforms.RandomAffine(0,translate=(0.3,0.3)),\
        transforms.ToTensor(),
     ])


n_epoch = 20
batch_size = 32
input_size = 3
hidden_size = 300
num_layers = 1
learning_rate= 1e-3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = SimModel(input_size,hidden_size,num_layers)
model = model.to(device)
optim = torch.optim.Adam(model.parameters() , lr=learning_rate)

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

loss_func = nn.BCEWithLogitsLoss()
e_range = trange(n_epoch)
for e in e_range:
    for data in dataloader:
        x,_ = data 
        # plt.imshow(x[1,:,:,:].squeeze().cpu().detach().numpy()*255, cmap='gray')
        # plt.savefig('sample_digit.png')
        x = torch.reshape(x,(-1,1,28*28)).transpose(-1,1)
        x = x.to(device).to(torch.float)
        y =x[:,1:,:].to(torch.float)
        x = x[:,:-1,:]
        pos = torch.from_numpy(np.array([[[i,j] for i in range(28) for j in range(28)]])).repeat(batch_size,1,1).to(device).to(torch.float)
        x = torch.cat([x,pos[:,:-1,:]],dim=2)
        pred = model(x)
        loss = loss_func(pred,y)
        ### update 
        optim.zero_grad()
        loss.backward()
        optim.step()

    e_range.set_description(f"Epoch {e}, Loss: {loss.cpu().detach().numpy():.2f}")
    s = torch.rand(16,1,1).to(device).to(torch.float)
    p = torch.from_numpy(np.array([[[0.0,0.0]]])).repeat(16,1,1).to(device).to(torch.float)
    s = torch.cat([s,p],dim=2)
    model.sample(s)




# dataiter = iter(dataloader)
# image,label = dataiter.next() # ERROR
# print(image.size())