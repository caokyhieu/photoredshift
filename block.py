import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.optim import Adam,RMSprop
import matplotlib.pyplot as plt
import random
import pdb 
from tqdm import tqdm,trange
import numpy as np
from torch.autograd import Function
from torch.autograd.gradcheck import gradcheck


def check_gradient(grad):
    if grad is not None:
        print(f'Mean Abs grad : {torch.mean(torch.abs(grad)):.4f}, Grad Shape: {grad.size()}')
    else:
        print('Doesnt have grad')

def get_grad(model):
    result = []
    for param in model.parameters():
        if param.requires_grad:
            result.append(param.grad.data.norm(2).item())
    
    return np.sum(result)



class SimpleBlock(nn.Module):
    def __init__(self,inp_dim,out_dim,n_replicate,activation=None):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim 
        self.n_replicate = n_replicate
        if activation:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()
        self.model = []
        # self.drop = nn.Dropout(p=0.5)
        self.skip = nn.Linear(self.inp_dim,self.out_dim)
        for i in range(self.n_replicate - 1):
            self.model += [nn.Linear(self.inp_dim,self.inp_dim),\
                        self.activation,\
                           nn.BatchNorm1d(self.inp_dim)
                            ]
        self.model += [nn.Linear(self.inp_dim,self.out_dim),\
                            self.activation,\
                            nn.BatchNorm1d(self.out_dim),
                            ]
        
        self.model = nn.Sequential(*self.model)

    def forward(self,x):
        y = self.model(x)
        x = 1/2*(self.activation(self.skip(x)) + y)
        return x


class SimpleNet(nn.Module):
    def __init__(self,inp_dim,hidden_dim):
        super().__init__()
        self.block1 = SimpleBlock(inp_dim,hidden_dim,1,activation=True)

        self.block2 = SimpleBlock(hidden_dim,hidden_dim,5,activation=True)
        # self.block1 = SimpleBlock(inp_dim,inp_dim,1)
        # self.latent = LatentLayer(hidden_dim,hidden_dim,bias=False)
        self.fc = nn.Linear(hidden_dim,1)
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim

    
    def forward(self,x):
        x = self.block1(x)
        # x.register_hook(check_gradient)
        x = self.block2(x)
        # x.register_hook(check_gradient)
        # x = self.latent(x)
        t = x.dtype

        rand = torch.randn(x.size()[-1],x.size()[-1],dtype=t).to(x.device)
        rand = rand/torch.sqrt(torch.sum(rand**2,dim=0,keepdims=True))
        x = x.mm(rand)
        # x.register_hook(check_gradient)
        x = self.fc(x)
        return x

    def test_sanity(self):
        input =  torch.randn(10,self.inp_dim, dtype=torch.float64,requires_grad=True)
        
        test = gradcheck(self.double(), input, eps=1e-6, atol=1e-4)
        print(test)
        self.float()

    


class LinearRegress(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.fc = nn.Linear(input_size,1)

    def forward(self,x):
        x = self.fc(x)
        return x

class CustomLatent(Function):
     # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        t = input.dtype
        random_matr = torch.randn(weight.size()[0],weight.size()[0],dtype=t).to(input.device)
        random_matr = random_matr/torch.sqrt(torch.sum(random_matr**2,dim=0,keepdims=True))
        ctx.save_for_backward(input, weight, bias,random_matr)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.mm(random_matr)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, random_matr = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        grad_output = grad_output.mm(random_matr.t())
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(random_matr.T)
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.mm(random_matr.T)
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            # grad_bias = grad_output.mm(random_matr.T)
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
        

class LatentLayer(nn.Module):
    def __init__(self,input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomLatent.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )



n_epoch=10000

net = SimpleNet(100,30)
# net.test_sanity()
# pdb.set_trace()
list_loss = []
train_loss = []
loss_func = nn.MSELoss()
eval_metric = nn.MSELoss()
X =  torch.randn(10000,100)
y = torch.sum(X**3 + X,dim=-1,keepdims=True)
X_test = torch.randn(100,100)
y_test = torch.sum(X_test**3 + X_test,dim=-1,keepdims=True)
optimizer = Adam(net.parameters(),lr=1e-4)
# pdb.set_trace()

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
index = [i for i in range(len(X))]
r = trange(n_epoch)
# pdb.set_trace()
for i in r:
    net.train()
    for b in range(0,len(X),batch_size):
        optimizer.zero_grad()

        X_train = X[index[b:b+batch_size],:].to(device)
        y_train = y[index[b:b+batch_size]].to(device)
        
        out = net(X_train)
        loss = loss_func(out, y_train)
        loss.backward()
        # print(get_grad(net))
        optimizer.step()
       
    random.shuffle(index)
    train_loss.append(loss.cpu().detach().numpy())
    # for name,param in net.named_parameters():
    #     print(f'Name:{name}, Grad: {torch.mean(torch.abs(param.grad.data)):.2f}')
    # test
    
    with torch.no_grad():
        net.eval()
        out_pred = net(X_test)
        test_loss = eval_metric(out_pred , y_test)
        list_loss.append(test_loss.cpu().detach().numpy())
        r.set_description(f'Epoch {i}: , Train Loss: {loss:.5f}, Test Loss: {test_loss:.5f}')

r.close()
ax1 = plt.subplot(211)
ax1.plot(list_loss,'-r')
ax2 = plt.subplot(212)
plt.plot(train_loss,'-b')
plt.savefig('loss.png')





