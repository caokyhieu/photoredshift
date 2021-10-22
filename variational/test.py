import torch
import torch.nn as nn
from torch.optim import Adam
import pdb
class S(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(10,20)
        self.ln2 = nn.Linear(20,30)
        self.ln3 = nn.Linear(30,1)


    def forward(self,x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        return x

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = S()
opt = Adam(model.parameters(),lr=0.01)
# pdb.set_trace()
model.ln2.register_forward_hook(get_activation('ln2'))
model.ln1.register_forward_hook(get_activation('ln1'))
model.ln3.register_forward_hook(get_activation('ln3'))

data = torch.rand(10,10)
label= torch.rand(10,)

loss = torch.sum(model(data) - label)
opt.zero_grad()
loss.backward()
opt.step()

for name,param in model.named_parameters():
    print(name, torch.norm(param.grad), param.requires_grad)
for k,v in  activation.items(): 
    print(f'layer {k}' + f": {torch.sum(v)}")
### free layer 2

model.ln2.weight.requires_grad = False 
model.ln2.bias.requires_grad = False
for i in range(3):
    # data = torch.rand(10,10)
    print('-'*10 + f'time {i}'+ '-'*10 )
    loss = torch.sum(model(data) - label)
    opt.zero_grad()
    loss.backward()
    opt.step()
    for name,param in model.named_parameters():
        print(name, torch.norm(param.grad),param.requires_grad)
    for k,v in  activation.items(): 
        print(f'layer {k}' + f": {torch.sum(v)}")
        print(f"Grad {k}: {torch.norm(v.grad)}")

