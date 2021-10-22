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



class VariationalTZM:

    def __init__(self):
        self.f_size = 1000
        self.batchsize = 128 
        self.threshold = 
        self.enum = None
        self.bin_edge = bin_edge
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass
    
    def model(self,data):
    ## define parameter for prior
        # define the hyperparameters that control the beta prior
        ff_prior = torch.tensor([1] * self.f_size)
    
        ff = pyro.sample("ff", dist.Dirichlet(ff_prior))
        with  pyro.plate("locals", len(data)):
            tzm = pyro.sample(f"tzm", dist.Categorical(ff))
            # print(f"Model tzm shape: {tzm.shape}")
            tzm = tzm.flatten()
            list_tzm = list(map(lambda x : (sample_tzm(x,self.bin_edge)), tzm))

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
        pass

    def guide(self):
           # define the hyperparameters that control the beta prior
        ff_params = pyro.param("ff_param",torch.tensor([1] * self.f_size),constraint=constraints.simplex)
        ff = pyro.sample("ff", dist.Dirichlet(ff_params))
        tzm_param = pyro.param(f"tzm_param",torch.ones(num_g,f_size),constraint=constraints.simplex)
    
        with  pyro.plate("locals", len(data)):
            # print(i)
            s = pyro.sample(f"tzm", dist.Categorical(tzm_param),infer=dict(baseline={'use_decaying_avg_baseline': True,
                                        'baseline_beta': 0.95}))
        pass

    def MAP_model(self,data):
        pass

    def MAP_guide(self,data):
        pass

    def MLE_model(self,data):
        pass
    
    def MLE_guide(self,data):
        pass
    
    def inference(self):
        pass
    
    def train(self,data):
        pass


