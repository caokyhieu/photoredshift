import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import logging
import pyro
import pyro.distributions as dist
from pyro.optim import Adam,ClippedAdam
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO,TraceGraph_ELBO,TraceEnum_ELBO,config_enumerate, infer_discrete
from utils.util_photometric import read_template,read_flux,mean_flux,find_tzm_index,read_file_tzm,sample_tzm
from utils.visualize import Visualizer
from utils.utils import DataGenerator
import torch.distributions.constraints as constraints
import pandas as pd
import json
import pdb
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm,trange
import sys
import glob
from pyro import poutine
from pyro.distributions.util import broadcast_shape
import random
from torch.utils.tensorboard import SummaryWriter
from pyro.ops.indexing import Vindex

# torch.set_default_tensor_type(torch.cuda.FloatTensor)
pyro.set_rng_seed(235)
logging.basicConfig(filename='example.log', level=logging.DEBUG,format='%(levelname)s:%(message)s')
### global params
min_z = 0.
max_z = 1.3
min_m = -2.0
max_m = 17.
n_t = 2
zbin = 50
mbin = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f_size = n_t * zbin * mbin

bin_t = np.linspace(0,1,num=n_t+1)
bin_z = np.linspace(min_z,max_z,num=zbin+1)
bin_m = np.linspace(min_m,max_m,num=mbin+1)
bin_edge = (bin_t,bin_z,bin_m)
a = 1.

S_N_ratio = 5.0
lr = 1e-4
num_iters = 30000
num_g = 234098
batchsize =  40000
N_samples = 5000
wait_step= 100

path_obs ='/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv'    
list_temp_path = ['/data/phuocbui/MCMC/reimplement/data/Ell2_flux_z_no_header.dat',
                    '/data/phuocbui/MCMC/reimplement/data/Ell5_flux_z_no_header.dat']
template = read_template(list_temp_path)
data_or = DataGenerator(path_obs,cols=[2,3,4,5,6],header=None,cuda=False)
temp_z =  DataGenerator(path_obs,cols=[1],header=None,cuda=False)
temp_z.index = data_or.index
index = []
for i,j in enumerate(temp_z):
    if j < max_z:
        index.append(i)
index = index[:num_g]
data_or.index = [temp_z.index[i] for i in index]
data = [i for i in data_or]
print(len(data))

data = torch.tensor(data).to(torch.float)
min_data = torch.mean(data,dim=0,keepdim=True)
max_data = torch.std(data,dim=0,keepdim=True)

max_data = max_data.to(torch.float) + 1e-16
min_data = min_data.to(torch.float)


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim,out_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, out_dim)
        self.fc22 = nn.Linear(hidden_dim, out_dim)
        # setup the non-linearities
        self.softplus = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # print(hidden)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_ = self.fc21(hidden) 
        var_ = self.softplus(self.fc22(hidden)) + 1e-6
        return loc_,var_

class Encoder(nn.Module):
    def __init__(self,data_dim, hidden_dim,tdim,zdim,mdim,f_size=1000):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(f_dim, hidden_dim)
        self.tdim = tdim
        self.zdim = zdim
        self.mdim = mdim
        self.fc2 = nn.Linear(data_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc42 = nn.Linear(tdim, tdim*10)
        self.fc41 = nn.Linear(hidden_dim + tdim*10, tdim*150)
        self.fc4 = nn.Linear(tdim*150, tdim)
        self.fc52 = nn.Linear(zdim, hidden_dim)
        self.fc51 = nn.Linear(hidden_dim*2, zdim*3)
        self.fc5 = nn.Linear(zdim*3, zdim)
        self.fc62 = nn.Linear(mdim, mdim*3)
        self.fc61 = nn.Linear(hidden_dim + mdim*3, mdim*15)
        self.fc6 = nn.Linear(mdim*15, mdim)
        # self.fc7 = nn.Linear(f_size, hidden_dim)
        # self.fc22 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, hidden_dim)
        self.fc24 = nn.Linear(hidden_dim, hidden_dim)
        self.fc25 = nn.Linear(hidden_dim, hidden_dim)
        # setup the non-linearities
        # self.softplus = nn.Softplus()
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)

        # self.sigmoid = nn.Sigmoid()

    def forward(self,x,y):
        """
        y: ff (global)
        return t,z,m
        """
        # print(f"y input: {y.size()}")
        # define the forward computation on the latent z
        # first compute the hidden units
        # hidden1 = self.act(self.fc1(ff))

        batch = x.size()[0]
        y = y.reshape(self.tdim+1,self.zdim,self.mdim)
        # print(f"y shape: {y.shape}")
        y_t = y.sum(dim=(1,2))[[1]].unsqueeze(0)
        # print(f"y_t shape: {y_t.shape}")
        y_z = y.sum(dim=(0,2)).unsqueeze(0)
        # print(f"y_z shape: {y_z.shape}")
        y_m = y.sum(dim=(0,1)).unsqueeze(0)
        # print(f"y_m shape: {y_m.shape}")
        y_t = self.act(self.fc42(y_t.repeat(batch,1)))
        y_z =   self.act(self.fc52(y_z.repeat(batch,1)))
        y_m = self.act(self.fc62(y_m.repeat(batch,1)))

        # # print(f"batch shape: {batch}")
        # y = self.fc7(y).repeat(batch,1)
        # print(f"y shape: {y.size()}")
        hidden = self.act(self.fc2(x))
        # hidden = torch.cat((hidden,y),dim=1)
        # hidden = self.act(self.fc22(hidden))
        hidden = self.act(self.fc23(hidden))
        hidden = self.act(self.fc24(hidden))
        hidden = self.act(self.fc25(hidden))
        # hidden = torch.cat((hidden1,hidden2),dim=1)
        # hidden = self.act(self.fc3(hidden))
        hidden_t = torch.cat((hidden,y_t),dim=1)
        hidden_z = torch.cat((hidden,y_z),dim=1)
        hidden_m = torch.cat((hidden,y_m),dim=1)
        t = torch.sigmoid(self.fc4(self.act(self.fc41(hidden_t))))
        z = torch.softmax(self.fc5(self.act(self.fc51(hidden_z))),dim=1)
        m = torch.softmax(self.fc6(self.act(self.fc61(hidden_m))),dim=1)
        return t,z,m


class Params(nn.Module):
    def __init__(self,out_dim):
        super().__init__()
        # setup the two linear transformations used
        self.weight = torch.nn.Parameter(data=torch.Tensor([1.] * out_dim), requires_grad=True)
        
    def forward(self):
        return torch.nn.functional.relu(self.weight)


data_dim= 5
hidden_dim = 512
out_dim = 5

# decoder = Decoder(1, hidden_dim,out_dim)
encoder = Encoder(data_dim, hidden_dim,n_t-1,zbin,mbin,f_size)
# decoder.double()
p = Params(f_size)

def model(data):
    # define the hyperparameters that control the beta prior
    ff_prior = torch.tensor([1] * f_size)
    ff = pyro.sample("ff", dist.Dirichlet(ff_prior))
    with  pyro.plate("locals", len(data)):
        tzm = pyro.sample(f"tzm", dist.Categorical(ff))
        # print(f"Model tzm shape: {tzm.shape}")
        tzm = tzm.flatten()
        list_tzm = list(map(lambda x : (sample_tzm(x,bin_edge)), tzm))

        mean = list(map( lambda x: a * mean_flux(x,template), list_tzm))
        std = list(map( lambda x: [b/S_N_ratio  
                            if b > 0.0 else 1000.0
                            for b in x], mean))
        mean = torch.tensor(mean).to(torch.float).reshape(f_size,1,data.size()[1])
        # print(f"mean shape: {mean.shape}")
        std = torch.tensor(std).to(torch.float).reshape(f_size,1,data.size()[1])
        ## reshape data
        obs_data =  std.new_zeros(broadcast_shape(std.shape, data.shape))
        obs_data[..., :, :] = data                
        s = pyro.sample(f"obs", dist.Normal(mean,std).to_event(1), obs=obs_data)
        # print(f"sample batch shape: {dist.Normal(mean,std).to_event(1).batch_shape}")

@config_enumerate
def guide(data):
    # define the hyperparameters that control the beta prior
    ff_params = pyro.param("ff_param",torch.tensor([1] * f_size),constraint=constraints.positive)
    ff = pyro.sample("ff", dist.Dirichlet(ff_params))
    tzm_param = pyro.param(f"tzm_param",torch.ones(num_g,f_size),constraint=constraints.simplex)
  
    with  pyro.plate("locals", len(data)):
        # print(i)
        s = pyro.sample(f"tzm", dist.Categorical(tzm_param),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                     'baseline_beta': 0.95}))
        # print(f"Guide shape: {s.shape}")

def continous_model(data,tzm=(None,None,None),index=None):
    # define the hyperparameters that control the beta prior
    ff_prior = torch.tensor([1] * f_size).to(torch.float).to(device)
    ff = pyro.sample("ff", dist.Dirichlet(ff_prior))
    

    with pyro.plate("locals", len(data),device=device):
        t_,z_,m_ = tzm
        m = pyro.sample("m", dist.Categorical(ff.reshape(n_t,zbin,mbin).sum(dim=(0,1))),obs=m_) ### mdim
        z = pyro.sample("z", dist.Categorical(ff.reshape(n_t,zbin,mbin).sum(dim=(0,2))),obs=z_) ### zdim
        t =  pyro.sample("t", dist.Bernoulli(ff.reshape(n_t,zbin,mbin).sum(dim=(1,2))[1]),obs=t_)

        # print(f"model t shape {t.shape}")
        # print(f"model z shape {z.shape}")
        # print(f"model m shape {m.shape}")
        
        tzm = t*zbin*mbin + z*mbin + m ## broadcasting shape (n_t,zbin,mbin)
        shape = tzm.shape
        # print(f"tzm shape: {tzm.shape}")
        tzm = tzm.detach().cpu().numpy().astype(int).flatten()

        list_tzm = list(map(lambda x : (sample_tzm(x,bin_edge)), tzm))

        mean = list(map( lambda x: a * mean_flux(x,template), list_tzm))
        std = list(map( lambda x: [b/S_N_ratio  
                            if b > 0.0 else 1000.0
                            for b in x], mean))

        if len(shape) > 1:
            # print(shape)
            mean = torch.tensor(mean).to(torch.float).reshape(n_t,zbin,mbin,1,data.size()[1]).to(device)
            # print(f"mean shape: {mean.shape}")
            std = torch.tensor(std).to(torch.float).reshape(n_t,zbin,mbin,1,data.size()[1]).to(device)
        else:
            mean = torch.tensor(mean).to(torch.float).reshape(len(data),-1).to(device)
            # print(f"mean shape: {mean.shape}")
            std = torch.tensor(std).to(torch.float).reshape(len(data),-1).to(device)

        ## reshape data
        # print(f"Mean shape: {mean.shape}")
        # print(f"std shape: {std.shape}")
        obs_data =  std.new_zeros(broadcast_shape(std.shape, data.shape))
        obs_data[..., :, :] = data          
        s = pyro.sample(f"obs", dist.Normal(mean,std).to_event(1), obs=obs_data)
        # print(f"model sample shape: {s.shape}")
        # print(f"model Label shape: {obs_data.shape}")

@config_enumerate
def continous_guide(data,tzm=(None,None,None),index=None):
    # define the hyperparameters that control the beta prior
   
    ff_params = pyro.param("ff_param",torch.tensor([1] * f_size).to(device),constraint=constraints.positive)
    ff = pyro.sample("ff", dist.Dirichlet(ff_params))
    t_param = pyro.param("t_param",torch.ones(num_g).to(device),constraint=constraints.unit_interval)
    z_param = pyro.param("z_param",torch.ones(num_g,zbin).to(device),constraint=constraints.simplex)
    m_param = pyro.param("m_param",torch.ones(num_g,mbin).to(device),constraint=constraints.simplex)
   

    with  pyro.plate("locals", len(data),device=device):
        if index is not None:
            m_param_ = Vindex(m_param)[index,:]
            z_param_ = Vindex(z_param)[index, :]
            t_param_ = Vindex(t_param)[index]
        else: 
            m_param_ =  m_param
            z_param_ = z_param
            t_param_ = t_param
        m = pyro.sample("m", dist.Categorical(m_param_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                     'baseline_beta': 0.95}))
        z = pyro.sample("z", dist.Categorical(z_param_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                     'baseline_beta': 0.95}))
        t = pyro.sample("t", dist.Bernoulli(t_param_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                     'baseline_beta': 0.95}))
    pass

@config_enumerate
def continous_encoder_guide(data,tzm=(None,None,None),index=None):
    # define the hyperparameters that control the beta prior
   
    ff_params = pyro.param("ff_param",torch.tensor([1] * f_size),constraint=constraints.positive)
    ff = pyro.sample("ff", dist.Dirichlet(ff_params))
    pyro.module('encoderv4',encoder)
    

    with  pyro.plate("locals", len(data)):
        t_,z_,m_ = tzm
        if t_ is None:
            t_param,z_param,m_param = encoder((data-min_data)/max_data,ff_params.unsqueeze(0))
            t_param = t_param.squeeze(-1)
            m = pyro.sample("m", dist.Categorical(m_param),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
            
            z = pyro.sample("z", dist.Categorical(z_param),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
            t = pyro.sample("t", dist.Bernoulli(t_param),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
        else:
            m = pyro.sample("m", dist.Categorical(m_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
            
            z = pyro.sample("z", dist.Categorical(z_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
            t = pyro.sample("t", dist.Bernoulli(t_),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))

        # print(f"guide t shape {t.shape}")
        # print(f"guide z shape {z.shape}")
        # print(f"guide m shape {m.shape}")
        # print(f"guide t_param shape {t_param.shape}")
        # print(f"guide z_param shape {z_param.shape}")
        # print(f"guide m_param shape {m_param.shape}")
    pass

@config_enumerate
def new_model(data):
    # define the hyperparameters that control the beta prior
    ff_prior = torch.tensor([1] * f_size)
    ff = pyro.sample("ff", dist.Dirichlet(ff_prior))

    with pyro.plate("locals", len(data)):
        m = pyro.sample("m", dist.Categorical(ff.reshape(n_t,zbin,mbin).sum(dim=(0,1)))) ### mdim
        z = pyro.sample("z", dist.Categorical(ff.reshape(n_t,zbin,mbin).sum(dim=(0,2)))) ### zdim
        t =  pyro.sample("t", dist.Bernoulli(ff.reshape(n_t,zbin,mbin).sum(dim=(1,2))[1]))

        tzm = t*zbin*mbin + z*mbin + m ## broadcasting shape (n_t,zbin,mbin)
        shape = tzm.size()
        tzm = tzm.detach().numpy().astype(int).flatten()

        list_tzm = list(map(lambda x : (sample_tzm(x,bin_edge)), tzm))

        mean = list(map( lambda x: a * mean_flux(x,template), list_tzm))
        std = list(map( lambda x: [b/S_N_ratio  
                            if b > 0.0 else 1000.0
                            for b in x], mean))
        if len(shape) > 1:
            mean = torch.tensor(mean).to(torch.float).reshape(n_t,zbin,mbin,1,data.size()[1])
            std = torch.tensor(std).to(torch.float).reshape(n_t,zbin,mbin,1,data.size()[1])
        else:
            mean = torch.tensor(mean).to(torch.float).reshape(len(data),-1)
            # print(f"mean shape: {mean.shape}")
            std = torch.tensor(std).to(torch.float).reshape(len(data),-1)
      
        obs_data =  std.new_zeros(broadcast_shape(std.shape, data.shape))
        obs_data[..., :, :] = data            
        s = pyro.sample(f"obs", dist.Normal(mean,std).to_event(1), obs=obs_data)



def take_tzm_(data,encoder,batch=16,N_samples=100,num_g=500,bin_edge=None):
    """
    data: iterable
    encoder: NN with input is data sample and output is unnormalzie tzm prob
    """
    tzm = np.zeros((N_samples,num_g,3))
    with torch.no_grad():

        for i in tqdm(range(0,len(data),batch)):
            prob = encoder(data[i:i+batch]/mean_data).cpu().detach().numpy().astype(np.float64)
            # print(prob.shape)
            prob = prob/np.sum(prob,axis=1,keepdims=True)
            # print(i)
            # print(len(prob))
            for j in range(len(prob)):
                # print(np.sum(prob[j]))
                idx = np.argmax(np.random.multinomial(1,prob[j],size=N_samples),axis=1)
                for r,index in enumerate(idx):
                    t,z,m = sample_tzm(index,bin_edge)
                    tzm[r,i+j,:] = t,z,m

    return tzm

def take_sample(num_g,n_t,zbin,mbin,bin_edge,N_samples=1000,seperate='galaxy',encoder=None,data=None):

    from pyspark import SparkContext
    from pyspark import SparkConf
    # 
    conf=SparkConf()
    conf.set("spark.executor.memory", "512m")
    conf.set("spark.driver.memory", "80g")
    conf.set("spark.cores.max", "20")
    conf.set("spark.driver.maxResultSize","0")
    conf.set("spark.sql.shuffle.partitions","10")

    sc = SparkContext.getOrCreate(conf)
    
    samples = {'fraction':[],'tzm':np.zeros((N_samples,num_g,3))}
    dict_params = {name:pyro.param(name).data.cpu().numpy().tolist() for name in pyro.get_param_store().get_all_param_names()}
    ff_param = dict_params['ff_param']

    samples['fraction'] = np.random.dirichlet(ff_param,size=N_samples)

    if encoder is None:
        
        if seperate=='tzm':
            t_param = dict_params['t_param'] ## (shape N_g)
            z_param = dict_params['z_param']## shape N_g,zbin
            m_param = dict_params['m_param'] ## shape N_g,mbin
            
            t_list = list(map(lambda x:  np.random.binomial(1,x,size=N_samples), t_param))
            z_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), z_param))
            m_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), m_param))

            tzm_list = list(map(lambda x: x[0]* zbin* mbin + x[1] * mbin + x[2], zip(t_list,z_list,m_list))) ## shape (N_g,N_sample)

        elif seperate=='galaxy':
            tzm_param = [dict_params[f'tzm_param_{i}'] for i in range(num_g)]
            tzm_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), tzm_param))
        elif seperate=='enum':
            tzm_param = dict_params['tzm_param'] 
            assert len(tzm_param) == num_g
            print(np.sum(tzm_param[0]))
            tzm_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), tzm_param))

    else:
        encoder.eval()
        t_params = []
        z_params = []
        m_params = []
        with torch.no_grad():
            for k in range(0,len(data),batchsize):
                t_param,z_param,m_param = encoder((data[k:k+batchsize]-min_data)/max_data,torch.tensor(ff_param).unsqueeze(0))
                t_params.append(t_param.detach().cpu().numpy().squeeze(-1))
                z_params.append(z_param.detach().cpu().numpy())
                m_params.append(m_param.detach().cpu().numpy())

        t_param = np.concatenate(t_params,axis=0)
        z_param = np.concatenate(z_params,axis=0)
        m_param = np.concatenate(m_params,axis=0)

        t_list = list(map(lambda x:  np.random.binomial(1,x,size=N_samples), t_param))
        z_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), z_param))
        m_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), m_param))

        tzm_list = list(map(lambda x: x[0]* zbin* mbin + x[1] * mbin + x[2], zip(t_list,z_list,m_list))) ## shape (N_g,N_sample)



    tzm_list = np.array(tzm_list).flatten()
    tzm_list = sc.parallelize(tzm_list)  ##RDD
    lambda_func = lambda x: sample_tzm(x,bin_edge)
    tzm_list = tzm_list.map(lambda_func)
    samples['tzm'] = np.array(tzm_list.collect()).reshape(num_g,N_samples,3).swapaxes(0,1)

    return samples

def get_tzm_sample(t_param,z_param,m_param,N_samples=10):
    """
    t_param: shape N_g
    z_param: shape N_g,zbin
    m_param: hsape N_g,mbin
    """
    num_g = len(t_param)
    t_list = list(map(lambda x:  np.random.binomial(1,x,size=N_samples), t_param))
    z_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), z_param))
    m_list = list(map(lambda x:  np.argmax(np.random.multinomial(1,np.array(x).astype(np.float64)/np.sum(x,dtype=np.float64),size=N_samples),axis=1), m_param))
    tzm_list = list(map(lambda x: x[0]* zbin* mbin + x[1] * mbin + x[2], zip(t_list,z_list,m_list))) ## shape (N_g,N_sample)
    

    from pyspark import SparkContext
    from pyspark import SparkConf
    conf=SparkConf()
    conf.set("spark.executor.memory", "512m")
    conf.set("spark.driver.memory", "40g")
    conf.set("spark.cores.max", "12")
    conf.set("spark.driver.maxResultSize","0")
    conf.set("spark.sql.shuffle.partitions","10")

    sc = SparkContext.getOrCreate(conf)

    tzm_list = np.array(tzm_list).flatten()
    tzm_list = sc.parallelize(tzm_list)  ##RDD
    lambda_func = lambda x: sample_tzm(x,bin_edge)
    tzm_list = tzm_list.map(lambda_func)
    tzm_samples = np.array(tzm_list.collect()).reshape(num_g,N_samples,3).swapaxes(0,1)

    return tzm_samples


def init_loc_fn(site):
    if site["name"] == "ff":
        # Initialize weights to uniform.
        return torch.ones(f_size) / f_size
    # if site["name"] == "scale":
    #     return (data.var() / 2).sqrt()
    # if site["name"] == "locs":
    #     return data[torch.multinomial(torch.ones(len(data)) / len(data), K)]
    raise ValueError(site["name"])

def my_custom_L2_regularizer(my_parameters):
    reg_loss = 0.0
    for param in my_parameters:
        reg_loss = reg_loss + param.pow(2.0).sum()
    return reg_loss

def train(threshold=1e6,encode=True):
    writer = SummaryWriter('./logs/')
    idx = [i for i in range(len(data))]
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})##torch.optim.Adam
    
    # scheduler = pyro.optim.MultiplicativeLR({'optimizer': optimizer, 'optim_args': {'lr': lr}, 'lr_lambda': 0.95})
    ls =  TraceEnum_ELBO(max_plate_nesting=1,num_particles=1) #Trace_ELBO(num_particles=1) ##TraceGraph_ELBO()##Trace_ELBO(num_particles=1)##pyro.infer.JitTraceGraph_ELBO() #
    svi = SVI(continous_model, continous_encoder_guide, optimizer, loss=ls)
    elbo = svi.step(data[:batchsize].to(torch.float))
    list_params = []
    for name, value in pyro.get_param_store().named_parameters():
        list_params.append(value)
    optimizer = torch.optim.Adam(list_params,lr=lr)
    loss_fn = TraceEnum_ELBO(max_plate_nesting=1,num_particles=1).differentiable_loss
    
    min_val = 1e28
    count = 0
    # for name, value in pyro.get_param_store().named_parameters():
    #     value.register_hook(lambda g,name=name:  writer.add_scalar(f'{name}_grad', g.norm().item()))#gradient_norms[name].append(g.norm().item()))
    for i in trange(num_iters):
        eb = []
        for k in range(0,len(data),batchsize):
            tzm = (None,None,None)
            if k < (len(data) - batchsize):
                
                # elbo = svi.step(data[idx[k:k+batchsize]].to(torch.float),tzm)
                elbo = loss_fn(continous_model,continous_encoder_guide,data[idx[k:k+batchsize]].to(torch.float),tzm)
                (elbo + 1e-2 * my_custom_L2_regularizer(list_params) ).backward()
                # take a step and zero the parameter gradients
                optimizer.step()
                for tag, parm in pyro.get_param_store().named_parameters():
                    writer.add_histogram(f'{tag}_grad', parm.grad.clone().cpu().detach().numpy(), i)
                optimizer.zero_grad()
                eb.append(elbo)
            else:
                break
        eb = np.sum(eb) * len(data)/( len(data)//batchsize * batchsize)

        if eb <min_val:
            min_val = eb
            count=0 
            torch.save(encoder.state_dict(), 'encoder.pth')
            print(f"Save encoder")

        else:
            count +=1
        if count > wait_step:
            print(f"Loss not improve through {wait_step} steps. The min loss :{min_val}")
            break
        if eb < threshold:
            print(eb)
            print(f"Finish at {i} iter, ELBO: {eb}")
            
            break
        logging.info("Elbo loss: {}".format(eb))            
        random.shuffle(idx)
    
    samples = take_sample(num_g,n_t,zbin,mbin,bin_edge,N_samples=N_samples,seperate='tzm',encoder=encoder,data=data)
    if encode:
        samples['tzm'] = take_tzm_(data.to(torch.float),encoder,256,N_samples,num_g,bin_edge)
    real_tzm = DataGenerator(path_obs,cols=[0,1,2],cuda=False,header=None)
    real_tzm.index = data_or.index
    real_tzm = [i for i in real_tzm]
    vis = Visualizer(samples,real_tzm,bin_t,bin_z,bin_m,burn_in=0.2)
    vis.MCMC_converge('MCMC_converge')
    vis.scatter_z('z')
    vis.confusion_t('t')

def load_encoder(path):
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})##torch.optim.Adam
    # scheduler = pyro.optim.MultiplicativeLR({'optimizer': optimizer, 'optim_args': {'lr': lr}, 'lr_lambda': 0.95})
    ls =  TraceEnum_ELBO(max_plate_nesting=1,num_particles=1) #Trace_ELBO(num_particles=1) ##TraceGraph_ELBO()##Trace_ELBO(num_particles=1)##pyro.infer.JitTraceGraph_ELBO() #
    svi = SVI(continous_model, continous_encoder_guide, optimizer, loss=ls)
    svi.step(data[:50].to(torch.float))
    encoder.load_state_dict(torch.load(path))
    print('Loaded model')
    samples = take_sample(num_g,n_t,zbin,mbin,bin_edge,N_samples=N_samples,seperate='tzm',encoder=encoder,data=data)
    real_tzm = DataGenerator(path_obs,cols=[0,1,2],cuda=False,header=None)
    real_tzm.index = data_or.index
    real_tzm = [i for i in real_tzm]
    vis = Visualizer(samples,real_tzm,bin_t,bin_z,bin_m,burn_in=0.2)
    vis.MCMC_converge('MCMC_converge')
    vis.scatter_z('z')
    vis.confusion_t('t')


def train_(threshold=1e6,threshold2=170.):
   
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})
    
    global_guide = AutoDelta(poutine.block(new_model, expose=['ff']),init_loc_fn=init_loc_fn)
    ls =  TraceEnum_ELBO(max_plate_nesting=1) #Trace_ELBO(num_particles=1) ##TraceGraph_ELBO()##Trace_ELBO(num_particles=1)##pyro.infer.JitTraceGraph_ELBO() #
    svi = SVI(new_model, global_guide, optimizer, loss=ls)
   
    gradient_norms = defaultdict(list)
    ## init params
    elbo = svi.step(data.to(torch.float))
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g,name=name: gradient_norms[name].append(g.norm().item()))
    # pdb.set_trace()
    eb = []
    ff_data = data[:5000]
    for i in trange(num_iters):
        elbo = svi.step(ff_data.to(torch.float))
        # eb.append(elbo)
        if elbo < threshold:
            print(elbo)
            print(f"Finish at {i} iter, ELBO: {elbo}")
            # torch.save(encoder.state_dict(), 'encoder.pth')
            # print(f"Save encoder")
            break
        if i%2 ==0:
            logging.info("Elbo loss: {}".format(elbo))
            # eb = []
    ## infer
    samples = {'tzm':[],'fraction':[]}
    ff_param = global_guide(ff_data)['ff'].detach().numpy()
    samples['fraction'] = np.random.dirichlet(ff_param,size=N_samples)

    guide_trace = poutine.trace(global_guide).get_trace(data)  # record the globals
    trained_model = poutine.replay(new_model, trace=guide_trace)  # replay the globals

    @config_enumerate
    def full_guide(data):
        # Global variables.
        with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
            global_guide(data)

        # Local variables.
        with pyro.plate('locals', len(data)):
            m_probs = pyro.param('m_probs', torch.ones(len(data), mbin) / mbin,
                                        constraint=constraints.simplex)
            z_probs = pyro.param('z_probs', torch.ones(len(data), zbin) / zbin,
                                        constraint=constraints.simplex)
            t_probs = pyro.param('t_probs', torch.ones(len(data)),
                                        constraint=constraints.unit_interval)

            m = pyro.sample("m", dist.Categorical(m_probs)) ### mdim
            z = pyro.sample("z", dist.Categorical(z_probs)) ### zdim
            t =  pyro.sample("t", dist.Bernoulli(t_probs))
    
   
    t_param = []
    z_param = []
    m_param = []
    for i in range(0,len(data),batchsize):
        pyro.clear_param_store()

        optim = pyro.optim.Adam({'lr': 0.2, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        svi = SVI(new_model, full_guide, optim, loss=elbo)
        temp_data = data[i:i+batchsize]
        for j in trange(num_iters):
            elbo = svi.step(temp_data.to(torch.float))
            
            if elbo < threshold2:
                print(elbo)
                print(f"Finish at {j} iter, ELBO: {elbo}")
                # append param
                break
            if j %10==0:
                logging.info("Elbo {} loss: {}".format(i,elbo))
                
        t_param.append(pyro.param('t_probs').detach().numpy())
        z_param.append(pyro.param('z_probs').detach().numpy())
        m_param.append(pyro.param('m_probs').detach().numpy())
        print(f"Finish training batch {i}")
    
    print(f"Trained all data")
    t_param = np.concatenate(t_param)
    z_param = np.concatenate(z_param)
    m_param = np.concatenate(m_param)
    samples['tzm'] = get_tzm_sample(t_param,z_param,m_param,N_samples=N_samples)
    real_tzm = DataGenerator(path_obs,cols=[0,1,2],cuda=False,header=None)
    real_tzm.index = data_or.index
    real_tzm = [i for i in real_tzm]
    vis = Visualizer(samples,real_tzm,bin_t,bin_z,bin_m,burn_in=0.2)
    vis.MCMC_converge('MCMC_converge')
    vis.scatter_z('z')
    vis.confusion_t('t')

def train_v2(threshold=1e6,encode=True):
    pyro.clear_param_store()
    optimizer = Adam({"lr": 1e-1})
    ls =  TraceEnum_ELBO(max_plate_nesting=1,num_particles=1) #Trace_ELBO(num_particles=1) ##TraceGraph_ELBO()##Trace_ELBO(num_particles=1)##pyro.infer.JitTraceGraph_ELBO() #
    # ls.loss(continous_model,continous_encoder_guide, data)
    svi = SVI(continous_model, continous_guide, optimizer, loss=ls)
    max_ = 0.0
    anchor = 0
    idx = [i for i in range(len(data))]
    gradient_norms = defaultdict(list)
    ## init params
    elbo = svi.step(data[:batchsize].to(torch.float).to(device),(None,None,None),[i for i in range(batchsize)])
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g,name=name: gradient_norms[name].append(g.norm().item()))
    # pdb.set_trace()
    for i in trange(num_iters):
        ### loop samples
        list_elbo = []
        for k in range(0,len(data),batchsize):
            tzm = (None,None,None)
            if k < (len(data) - batchsize): 
                temp_data = data[idx[k:k+batchsize]].to(torch.float).to(device)
                elbo = svi.step(temp_data,tzm,idx[k:k+batchsize])
                list_elbo.append(elbo)
        if np.sum(list_elbo) < threshold:
            break
        logging.info("Elbo loss: {}".format(np.sum(list_elbo)))    

        random.shuffle(idx)
    print(f"Finished at ELBO: {np.sum(list_elbo)}")

    # samples = take_sample(num_g,n_t,zbin,mbin,bin_edge,N_samples=N_samples,seperate='tzm',encoder=encoder,data=data)
    samples = take_sample(num_g,n_t,zbin,mbin,bin_edge,N_samples=N_samples,seperate='tzm',encoder=None,data=None)
        # params[name] = pyro.param(name).data.numpy().tolist()
    if encode:
        samples['tzm'] = take_tzm_(data.to(torch.float),encoder,256,N_samples,num_g,bin_edge)
    

    real_tzm = DataGenerator(path_obs,cols=[0,1,2],cuda=False,header=None)
    real_tzm.index = data_or.index
    real_tzm = [i for i in real_tzm]
    vis = Visualizer(samples,real_tzm,bin_t,bin_z,bin_m,burn_in=0.2)
    vis.MCMC_converge('MCMC_converge')
    vis.scatter_z('z')
    vis.confusion_t('t')

    
