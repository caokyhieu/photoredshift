import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import logging
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO,TraceGraph_ELBO
from utils.util_photometric import read_template,read_flux,mean_flux,find_tzm_index,read_file_tzm,sample_tzm
from utils.visualize import Visualizer
from utils.utils import DataGenerator
import torch.distributions.constraints as constraints
import pandas as pd
import json
import pdb
from collections import defaultdict
import pandas as p
from tqdm import tqdm
from pyro.infer import MCMC,HMC, NUTS, config_enumerate, infer_discrete

# torch.set_default_tensor_type('torch.FloatTensor')
# torch.set_default_tensor_type(torch.DoubleTensor)
pyro.set_rng_seed(101)
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
lr = 1e-3
num_iters = 10000
num_g = 500
batchsize = 2048
N_samples = 500


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

data = torch.tensor(data)
mean_data,_ = torch.max(torch.abs(data),dim=0,keepdim=True)
mean_data = mean_data.to(torch.float)



def model(data):
   
    ff_prior = torch.tensor([1] * f_size)
    ff = pyro.sample("ff", dist.Dirichlet(ff_prior))
   
    for i in  pyro.plate("locals", len(data)):
       
        tzm = pyro.sample(f"tzm_{i}", dist.Categorical(ff))
        t,z,m = sample_tzm(tzm,bin_edge)
        mean = a * torch.tensor(mean_flux((t,z,m),template))
        std = torch.tensor([b/S_N_ratio  
                                if b > 0.0 else 1000.0
                                    for b in mean])
        s = pyro.sample(f"obs_{i}", dist.Normal(mean,std).to_event(1), obs=data[i])

def run():
    hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
    mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
    mcmc.run(data)
    mcmc.get_samples()