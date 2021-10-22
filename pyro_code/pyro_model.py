import logging
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,DistanceMetric
import numpy as np

from pyro.infer import MCMC, NUTS,HMC
pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.7.0')

from abc import ABC,abstractmethod
from scipy.stats import wasserstein_distance

class ToyExmaple:
    def __init__(self,num_m=5,num_samples=20000):
        
        self.M = num_m
        self.neigh = NearestNeighbors(metric='euclidean',n_neighbors=self.M)
        self.n_samples = num_samples
        
    def model_GMC(self,x_l, x_u, y,n_l,n_u,**kwargs):
        prior = kwargs.get('prior',None)
        assert prior is not None,'Need to set prior'
        ## need to discrete y and pi
        ### pi is an 1D array
        
#         max_y = np.max(y.detach().cpu().numpy())
#         min_y = np.min(y.detach().cpu().numpy())
#         num_c = 10
#         y_arr = np.linspace(min_y,max_y,num_c +1)
# #         y_arr_val = (y_arr[1:] + y[:-1])/2
        
#         pi = torch.linspace(start=0.1, end=1, steps=10)  #np.linspace(0.1,1,10)
        theta = pyro.sample('theta',dist.Dirichlet(self.theta_prior))
        
#         argmax_theta = theta.argmax(axis=1)
#         pi_ = pi[argmax_theta]
        weight_y_index = np.searchsorted(self.y_arr, y.detach().cpu().numpy())
        weight_y = theta[weight_y_index-1]
        argmax_weight_y = weight_y.argmax(axis=1)
        pi_ = self.pi[argmax_weight_y]
        
        typ = kwargs.pop('typ',False)
        weight = pyro.sample("weight", dist.Normal(torch.tensor(prior['weight']['mean']).float(),
                                                   torch.tensor(prior['weight']['std']).float()))
    
   
        m1 = pyro.sample("m1", dist.Normal(*prior['m1']))
        m2 = pyro.sample("m2", dist.Normal(*prior['m2']))
        sigma1 = pyro.sample("sigma1", dist.InverseGamma(*prior['sigma1']))
        sigma2 = pyro.sample("sigma2", dist.InverseGamma(*prior['sigma2']))
        
        with pyro.plate("x_L", n_l):
            x_label = pyro.sample("obs_x", dist.Normal(m1, sigma1), obs=x_l)
        with pyro.plate("x_U", n_u):
            x_unlabel = pyro.sample("unobs_x", dist.Normal(m2, sigma2), obs=x_u)
        
        p = pyro.sample("p", dist.Normal(torch.tensor(prior['p']['mean']).float(),
                                         torch.tensor(prior['p']['std']).float()))
        
        gamma = pyro.sample("gamma", dist.InverseGamma(torch.tensor(prior['gamma']['concentration']).float(),
                                                      torch.tensor(prior['gamma']['rate']).float()))
        
        covariate = self.covariate_export(x_label,p,gamma).float()
        mean =  torch.matmul(covariate,weight.float())
        if typ:

            sigma =  (x_label - x_unlabel.mean())**2/(x_label.mean() -x_unlabel.mean())**2
        else:
            sigma = 1.
        
        with pyro.plate("y", n_l),pyro.poutine.scale(scale=pi_):
            y_observer = pyro.sample("obs_y", dist.Normal(mean, np.sqrt(sigma)), obs=y)  
        
        return {'x_l':x_label.detach().numpy(),'x_u':x_unlabel.detach().numpy(),'y':y_observer.detach().numpy()}
    
    
    def model(self,x_l, x_u, y,n_l,n_u,**kwargs):
        prior = kwargs.get('prior',None)
        assert prior is not None,'Need to set prior'
        
#         a = pyro.sample("a", dist.Normal(*prior['a']))
#         b = pyro.sample("b", dist.Normal(*prior['b']))
        typ = kwargs.pop('typ',False)
        weight = pyro.sample("weight", dist.Normal(torch.tensor(prior['weight']['mean']).float(),
                                                   torch.tensor(prior['weight']['std']).float()))
   
        m1 = pyro.sample("m1", dist.Normal(*prior['m1']))
        m2 = pyro.sample("m2", dist.Normal(*prior['m2']))
        sigma1 = pyro.sample("sigma1", dist.InverseGamma(*prior['sigma1']))
        sigma2 = pyro.sample("sigma2", dist.InverseGamma(*prior['sigma2']))
        
        with pyro.plate("x_L", n_l):
            x_label = pyro.sample("obs_x", dist.Normal(m1, sigma1), obs=x_l)
        with pyro.plate("x_U", n_u):
            x_unlabel = pyro.sample("unobs_x", dist.Normal(m2, sigma2), obs=x_u)
        
        p = pyro.sample("p", dist.Normal(torch.tensor(prior['p']['mean']).float(),
                                         torch.tensor(prior['p']['std']).float()))
        
        gamma = pyro.sample("gamma", dist.InverseGamma(torch.tensor(prior['gamma']['concentration']).float(),
                                                      torch.tensor(prior['gamma']['rate']).float()))
        covariate = self.covariate_export(x_label,p,gamma).float()
        mean =  torch.matmul(covariate,weight.float())
        if typ:
#             sigma = (x_label  - x_unlabel.mean())**2
            sigma =  (x_label - x_unlabel.mean())**2/(x_label.mean() -x_unlabel.mean())**2
#             self.neigh.fit(x_unlabel.cpu().detach().numpy().reshape(-1,1))
#             distance ,idx = self.neigh.kneighbors(x_label.cpu().detach().numpy().reshape(-1,1)) ## shape n_x,M
#             self.neigh.fit(x_label.cpu().detach().numpy().reshape(-1,1))
#             self_distance,_ = self.neigh.kneighbors() ## shape n_x,M
#             sigma = np.sum(distance,axis=1)/np.sum(self_distance,axis=1)
#             sigma = torch.from_numpy(sigma)
        else:
            sigma = 1.
        
        with pyro.plate("y", n_l):
            y_observer = pyro.sample("obs_y", dist.Normal(mean, np.sqrt(sigma)), obs=y)  
        
        return {'x_l':x_label.detach().numpy(),'x_u':x_unlabel.detach().numpy(),'y':y_observer.detach().numpy()}
    
    
    def  covariate_export(self,x,p,gamma):
        """
        p: a list of param
        gamma : alsit of param
        """
        assert len(p) == len(gamma),'p and gamma need have the same length' 
        if p.ndim ==1:
        
            return torch.exp(-(x.view(-1,1)-p.view(1,-1))**2/gamma.view(1,-1)**2)
        else:
            
            return torch.exp(-(x.view(-1,1)-p)**2/gamma**2)
    
    def inference(self,model,prior,data,typ=False,prior_check=True):
        hmc_kernel =  HMC(model, step_size=0.0855, num_steps=4)
        mcmc = MCMC(hmc_kernel, num_samples=self.n_samples, warmup_steps=200)
        prior_samples = None
        self.max_y = np.max(data['y'].detach().cpu().numpy())
        self.min_y = np.min(data['y'].detach().cpu().numpy())
        self.num_c = 10
        self.y_arr = np.linspace(self.min_y,self.max_y,self.num_c +1)
                
        self.pi = torch.linspace(start=0.1, end=1, steps=10)  #np.linspace(0.1,1,10)
        self.theta_prior = torch.ones_like(self.pi.reshape(1,-1)).tile(self.num_c,1)
        
        if  prior_check:
            d = data.copy()
            d['y'] = None
            d['x_l'] = None
            d['x_u'] = None

            prior_samples = self.prior_checking(model,prior,d,typ=typ)
        
        mcmc.run(**data,typ=typ,prior=prior)
        samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
        return prior_samples,samples
    
    def prior_checking(self,model,prior,data,typ=False):
        """
        data: A dictionary data, which require for your model(both covariate and rensponse)
        prior: a prior you want to set for model
        both need in from of dictionary
        """
        assert isinstance(prior,dict),"prior must in dict form"
        assert isinstance(data,dict),"data must in dict form"
        assert set(data.keys()).intersection(set(model.__code__.co_varnames)) == set(data.keys()),\
           f'data need have key as the argument name in model {data.keys()}'
        assert None in data.values(),"you need to set None to obs sample"
        d = data.copy()
        d.update({'prior':prior})
        return model(**d,typ=typ)
                            
def plot_samples_prior(samples,data=None):
    assert set(samples.keys()).intersection(set(data.keys())) == set(samples.keys()),'samples do not have in data'
    sites = list(samples.keys())
    fig, axs = plt.subplots(nrows=1, ncols=len(sites), figsize=(12, 5))
    fig.suptitle("Prior Checking", fontsize=16)
    for i, ax in enumerate(axs.reshape(-1)):
        site = sites[i]
        sns.distplot(samples[site], ax=ax, label="Prior")
        sns.distplot(data[site].cpu().numpy(), ax=ax, label="True")
        ax.set_title(site)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right');
        
    
def plot_samples_post(samples,true_params=None):
    sites = list(samples.keys())
    fig, axs = plt.subplots(nrows=1, ncols=len(sites), figsize=(12, 5))
    fig.suptitle("Posterior Checking", fontsize=16)
    for i, ax in enumerate(axs.reshape(-1)):
        site = sites[i]
        if samples[site].ndim ==1:
            sns.distplot(samples[site], ax=ax, label="posterior")
            ax.axvline(true_params[site],ymin=0,ymax=1,color='red',label='True')
            ax.set_title(site)
        else:
            for i in range(samples[site].shape[-1]):
                sns.distplot(samples[site][:,i], ax=ax, label= f"posterior {site}_{i}")
                ax.axvline(true_params[site][i],ymin=0,ymax=1,color='red',label=f'True {site}_{i}')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right');
    
def plot_predictive(x_l,x_u,y,y_u,sample1,sample2,true_params,name='test_GMC.png'):
    fig,axes = plt.subplots(figsize=(5,5))
    predict_df = pd.DataFrame(columns=['x','y_pred','sigma'])
    model_weight1 = sample1['weight']
    model_weight2 = sample2['weight']
  
    
    covariate1 = radial_basic_function(x_u,sample1['p'].mean(axis=0),sample1['gamma'].mean(axis=0))
    covariate2 = radial_basic_function(x_u,sample2['p'].mean(axis=0),sample2['gamma'].mean(axis=0))
    
    our_model_error = np.mean((y_u - np.matmul(covariate1 ,np.mean(model_weight1,axis=0,keepdims=True).T))**2)#0.0
    sig_1_model_error = np.mean((y_u - np.matmul(covariate2 ,np.mean(model_weight2,axis=0,keepdims=True).T))**2)#0.0

    print(f"model 1 error: {our_model_error}")
    print(f"model 2 error: {sig_1_model_error}")
    
    covariate_full1 = list(map(lambda x:radial_basic_function(x_u,x[0],x[1]), zip(sample1['p'],sample1['gamma'])))
    covariate_full2 = list(map(lambda x:radial_basic_function(x_u,x[0],x[1]), zip(sample2['p'],sample2['gamma'])))
    y_pred1 = list(map(lambda x: np.matmul(x[0],x[1].T).flatten(),zip(covariate_full1,model_weight1)))
    y_pred2 = list(map(lambda x: np.matmul(x[0],x[1].T).flatten(),zip(covariate_full2,model_weight2)))

    predict_df_1 = pd.DataFrame({'x':np.tile(x_u,len(model_weight1)),'y_pred':np.concatenate(y_pred1),\
                                         'sigma':['our model']*len(model_weight1) * len(x_u) })
    predict_df_2 = pd.DataFrame({'x':np.tile(x_u,len(model_weight2)),'y_pred':np.concatenate(y_pred2),\
                                         'sigma':['1']*len(model_weight2) * len(x_u) })

    predict_df = pd.concat([predict_df_1,predict_df_2])
    axes.set_title("Our model Error {:.3f}, {} model error {:.3f}".format(our_model_error\
                    ,r'$\sigma=1$',sig_1_model_error))
    axes.plot(x_u,y_u,'ro',label="unlabel data")
    axes.plot(x_l,y,'go',label="label data" )

    pseu_do = np.linspace(min(min(x_u),min(x_l)),max(max(x_l),max(x_u)),100)
    axes.plot(pseu_do,np.matmul(radial_basic_function(pseu_do,true_params['p'],true_params['gamma']),\
                                                        true_params['weight']),'-.',label="true func" )

    sns.lineplot(data=predict_df, x="x", y="y_pred", hue="sigma",ax=axes,ci='sd')
    fig.savefig(name)
    print(f"figure save at {name}")
    return predict_df
    
       
## prepare data
if __name__ == '__main__':
    m1=-1
    m2=3.
    sig1=3.
    sig2=4.

    weight = np.array([1.,2.,3.])
    p = np.array([1,2,5])
    gamma = np.array([5,6.,1.])
    n_l =100
    n_u = 100
    def radial_basic_function(x,p,gamma):
        """
        This function for scalar inputs
        """
        if p.ndim ==1:
            return np.exp((-(x.reshape(-1,1)-p.reshape(1,-1))**2/gamma.reshape(1,-1)**2) )
        else:
            
            return np.exp((-(x.reshape(-1,1)-p)**2/gamma**2) )

    x_l = np.random.normal(m1,np.sqrt(sig1),size=n_l)
    x_u = np.random.normal(m2,np.sqrt(sig2),size=n_u)

    y = np.random.normal(0,0.5,size=n_l) + np.matmul(radial_basic_function(x_l,p,gamma),weight) 
    y_u = np.random.normal(0,0.5,size=n_u) +  np.matmul(radial_basic_function(x_u,p,gamma),weight) 

    ## change to tensor
    true_params = {"weight":weight,
                "m1":m1,
                "m2":m2,
                    "p":p,
                    "gamma":gamma,
                "sigma1":sig1,
                "sigma2":sig2,
                    }
    data = {"x_l": torch.from_numpy(x_l),
            "x_u":torch.from_numpy(x_u),
            "y":torch.from_numpy(y),
        
            "n_l":n_l,
            "n_u":n_u}

    prior = {"weight":{"mean":[-1,-4.,-3],
                    "std":[1.,2.,5]},
            "p":{"mean":[2,-6,6],
                "std":[5,4,7]},
            "gamma":{"concentration":[1,2,3],
                    "rate":[1,1,2]},
            
            "m1":[-5,5],
            "m2":[-5,5],
            "sigma1":[1,3],
            "sigma2":[4,4]}

    toy = ToyExmaple(num_samples=4000)
    prior_samples,samples = toy.inference(toy.model_GMC,prior,data,typ=True,prior_check=False)
    prior_samples_2,samples_2 = toy.inference(toy.model,prior,data,typ=False,prior_check=False)
    df = plot_predictive(x_l,x_u,y,y_u,samples,samples_2,true_params)