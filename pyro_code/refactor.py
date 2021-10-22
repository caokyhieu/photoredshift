import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import ones_like
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import time 
import pdb
import seaborn as sns
import os
from itertools import product
import pdb
import matplotlib.gridspec as gridspec

def radial_basic_function(x,p,gamma):
    """
    This function for scalar inputs
    """
    if p.ndim==1:
        return np.exp(-(x.reshape(-1,1)-p.reshape(1,-1))**2/gamma.reshape(1,-1)**2)\
                    # + np.random.normal(0,1,size=x.shape)
    else:
        return np.exp(-(x-p)**2/gamma**2) #+ np.random.normal(0,1,size=x.shape)


from models.gibbs_sampling import ConditionalDistribution,GibbsSampling
from models.metropolis_hasting import MHSampling
from scipy.stats import invgamma,gamma
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,DistanceMetric

M = 5
GLOBAL_NEIGH = NearestNeighbors(metric='euclidean',n_neighbors=M)

def log_likelihood(y,x,x_u,theta,phi1,phi2,p,gamma,return_params=False,var_origin=True,neigh=GLOBAL_NEIGH):
     
    """Calculate likelihood of labeled data.
    Use parameter of unlabel distribution to calculate
    with assumption data in region of high prob of unlabel data, the variance will be small ^_^
    
    Keyword arguments:
    argument -- y: label
                x : labeled data
                theta: parameter of predictive model [a,c]
                phi2: parameter of unlabel covariate generative mdoel [m2,sig2]
        
    Return: likelihood of labeled covariate with our assumption
    """
    mean_label = np.mean(x)
    mean_unlabel = np.mean(x_u)
    _mean =  np.matmul(radial_basic_function(x,p,gamma),theta).flatten()
    dist_un = (x-phi2[0])**2
    dist_label = (x - mean_label)**2
    if var_origin:
        _var = 1.
    else:
        # p_l_u = np.exp(log_prob_normal(x,phi2,return_total=False))
        # p_l_l = np.exp(log_prob_normal(x,phi1,return_total=False))
        # _var = p_l_l/p_l_u
        _var = (x - mean_unlabel)**2/(x-mean_label)**2

    llh = - (y-_mean)**2/(2*_var) - 1/2 *np.log(2*np.pi * _var)

    if return_params:
        return np.sum(llh),_mean,_var
    return np.sum(llh)

def log_prob_normal(x,params,return_total=True):
    """calculate log prob of normal distribution
    
    Keyword arguments:
    argument -- params: mu and sigma
    Return: log prob
    """
    result = -(x-params[0])**2/(2*params[1]) - 1/2 * np.log(2*np.pi*params[1])
    # print(f'log_prob_normal shape {result.shape}')
    if return_total:
        return np.sum(result)
    return result


def log_prob_phi(x):
    """calculate log prob of phi distribution Normal-Inverse-Gamma dist
    
    Keyword arguments:
    argument -- params: mu and sigma
    Return: log prob
    """
    score = log_prob_normal(x[0],np.array([0,100])) ## params[0] is mean and var of Normal
    score += invgamma.logpdf(x[1], 1/2, loc=0, scale=1/2)
    # print(f"log_prob_phi shape {score.shape}")
    return score

import scipy.stats as stats

class MetropolisHasting(MHSampling):
    
    def __init__(self,target,step_size=10.0):
        super().__init__(step_size)

        """
        target: target log pdf function. with parameter x, return pdf of x value.
                  It has func logpdf(val) to return pdf of value and rvs to random value 
        n_sample : num_sample
        """
        self.target = target

    def set_target(self,target):
        self.target = target

    def score(self,x,y):
        s_ = self.target(x) \
            - np.sum(stats.norm.logpdf(x,loc=y,scale=self.step_size))
        return s_

    def jump(self,loc):
        return np.random.normal(loc=loc ,scale=self.step_size)

class PhiMetropolisHasting(MHSampling):
    
    def __init__(self,target,step_size=10.0):
        super().__init__(step_size)

        """
        target: target log pdf function. with parameter x, return pdf of x value.
                  It has func logpdf(val) to return pdf of value and rvs to random value 
        n_sample : num_sample
        """

        self.target = target

    def set_target(self,target):
        self.target = target

    def score(self,x,y):
        s_ = self.target(x) \
                - stats.norm.logpdf(x[0],loc=y[0],scale=self.step_size)\
                -   stats.uniform.logpdf(x[1],loc=y[1]/(1 + self.step_size),scale =y[1] *(1 +self.step_size)-y[1]/(1 + self.step_size))
        
        return s_
    def jump(self,loc):
        mean = loc[0]
        var = loc[1]
        ran_mean = np.random.normal(loc=mean,scale=self.step_size)
        ran_var = np.random.uniform(low=var/(1 + self.step_size),high=var *(1 +self.step_size))

        return np.array([ran_mean,ran_var])


class GammaMetropolisHasting(MHSampling):
    
    def __init__(self,target,step_size=10.0):
        super().__init__(step_size)

        """
        target: target log pdf function. with parameter x, return pdf of x value.
                  It has func logpdf(val) to return pdf of value and rvs to random value 
        n_sample : num_sample
        """

        self.target = target

    def set_target(self,target):
        self.target = target

    def score(self,x,y):
        s_ = self.target(x) \
                -   (stats.uniform.logpdf(x,loc=y/(1 + self.step_size),scale =y *(1 +self.step_size)-y/(1 + self.step_size))).sum()

        return s_
    def jump(self,loc):
        return np.random.uniform(low=loc/(1 + self.step_size),high=loc *(1 +self.step_size))

class ConditionalDistributionTheta(ConditionalDistribution):
    def __init__(self,y,x_l,x_u,scale=1.0,n_samples=100,var_origin=True,time=0):
        self.y = y
        self.x_l = x_l
        self.x_u = x_u
        self.scale = scale
        self.n_samples = n_samples
        self.var_origin= var_origin
        self.generator = MetropolisHasting(None,step_size=self.scale)
        self.time=time
        
        pass
    def prob_func(self,phi1,phi2,p,gamma):

        def score_func(x):
            log_prob = log_prob_normal(x,np.array([0,100]))
            llh = log_likelihood(self.y,self.x_l,self.x_u,x,phi1,phi2,p,gamma,var_origin=self.var_origin)
            if self.time >0:
                print(f"prior: {log_prob:.3f}, Liklihood: {llh:.3f} ")
                time.sleep(self.time)
            return log_prob + llh

        return  score_func

    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(phi1=self.phi1,phi2=self.phi2,p=self.p,gamma=self.gamma))

        pass 
    def sample(self):
        sample_ =  self.generator(self.theta,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from theta {sample_}")
        return sample_

class ConditionalDistributionPhi1(ConditionalDistribution):
    def __init__(self,y,x_l,x_u,scale=1.0,n_samples=100,var_origin=True,time=0):
        self.x_l = x_l
        self.scale = scale
        self.n_samples = n_samples
        self.generator = PhiMetropolisHasting(None,step_size=self.scale)
        self.y = y
        self.time = time
        self.x_u = x_u
        self.var_origin = var_origin

    def prob_func(self,theta,phi2,p,gamma):
        def score_func(x):
            log_prob = log_prob_normal(self.x_l,x)
            prior = log_prob_phi(x)
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,x,phi2,p,gamma,var_origin=self.var_origin)
            if self.time > 0:
                print(f"Marginal x_u: {log_prob:.3f}")
                time.sleep(self.time)
            return log_prob + prior + llh

        return score_func
                        
    def set_params(self,**kwargs):
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi2=self.phi2,p=self.p,gamma=self.gamma))
        pass 
    def sample(self):
        sample_ = self.generator(self.phi1,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from phi1 {sample_}")
        return sample_
        

class ConditionalDistributionPhi2(ConditionalDistribution):
    def __init__(self,y,x_l,x_u,scale=1.0,n_samples=100,var_origin=True,time=0):
        self.y = y
        self.x_l = x_l
        self.x_u=x_u
        self.scale = scale
        self.var_origin=var_origin
        self.n_samples = n_samples
        self.generator = PhiMetropolisHasting(None,step_size=self.scale)
        self.time = time
        pass

    def prob_func(self,theta,phi1,p,gamma):
        def score_func(x):
            log_prob = log_prob_normal(self.x_u,x)
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,phi1,x,p,gamma,var_origin=self.var_origin)
            prior = log_prob_phi(x)
            if self.time > 0:
                print(f"Liklihood: {llh:.3f}, marginal x_u: {log_prob:.3f}")
                time.sleep(self.time)
            return log_prob + llh + prior

        return score_func
                        
    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi1=self.phi1,p=self.p,gamma=self.gamma))
        pass 
    def sample(self):
        sample_ = self.generator(self.phi2,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from phi2 {sample_}")
        return sample_

class ConditionalDistributionP(ConditionalDistribution):
    def __init__(self,y,x_l,x_u,scale=1.0,n_samples=100,var_origin=True,time=0):
        self.y = y
        self.x_l = x_l
        self.x_u = x_u
        self.scale = scale
        self.n_samples = n_samples
        self.var_origin= var_origin
        self.generator = MetropolisHasting(None,step_size=self.scale)
        self.time=time
        
        pass
    def prob_func(self,theta,phi1,phi2,gamma):

        def score_func(x):
            log_prob = log_prob_normal(x,np.array([0,100]))
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,phi1,phi2,x,gamma,var_origin=self.var_origin)
            if self.time >0:
                print(f"prior: {log_prob:.3f}, Liklihood: {llh:.3f} ")
                time.sleep(self.time)
            return log_prob + llh

        return  score_func

    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi1=self.phi1,phi2=self.phi2,gamma=self.gamma))

        pass 
    def sample(self):
        sample_ =  self.generator(self.p,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from theta {sample_}")
        return sample_

class ConditionalDistributionGamma(ConditionalDistribution):
    def __init__(self,y,x_l,x_u,scale=1.0,n_samples=100,var_origin=True,time=0):
        self.y = y
        self.x_l = x_l
        self.x_u = x_u
        self.scale = scale
        self.n_samples = n_samples
        self.var_origin= var_origin
        self.generator = GammaMetropolisHasting(None,step_size=self.scale)
        self.time=time
        
        pass
    def prob_func(self,theta,phi1,phi2,p):

        def score_func(x):
            log_prob = (invgamma.logpdf(x,1/2, loc=0, scale=1/2)).sum()
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,phi1,phi2,p,x,var_origin=self.var_origin)
            if self.time >0:
                print(f"prior: {log_prob:.3f}, Liklihood: {llh:.3f} ")
                time.sleep(self.time)
            return log_prob + llh

        return  score_func

    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi1=self.phi1,phi2=self.phi2,p=self.p))

        pass 
    def sample(self):
        sample_ =  self.generator(self.gamma,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from theta {sample_}")
        return sample_

class TestGibbsSampling(GibbsSampling):
    
    def __init__(self,y,x_l,x_u,scale=1.,n_samples=1,var_origin=True,time=0):
        self.y = y
        self.x_l = x_l 
        self.x_u = x_u
        self.scale = scale
        self.var_origin = var_origin
        self.n_samples = n_samples
        self.time = time
        # self.flux = read_flux(path_obs)
        
        # self.n_samples=n_samples
        self.set_conditional()
        pass

    def set_conditional(self):
        self.conditional = OrderedDict({'theta':ConditionalDistributionTheta( self.y,self.x_l,self.x_u,scale=self.scale,n_samples=self.n_samples,var_origin=self.var_origin,time=self.time),
                                        'phi1':ConditionalDistributionPhi1(self.y,self.x_l,self.x_u,scale=self.scale,n_samples=self.n_samples,var_origin=self.var_origin,time=self.time),
                                        'phi2':ConditionalDistributionPhi2(self.y,self.x_l,self.x_u,scale=self.scale,n_samples=self.n_samples,var_origin=self.var_origin,time=self.time),
                                        'p':ConditionalDistributionP(self.y,self.x_l,self.x_u,scale=self.scale,n_samples=self.n_samples,var_origin=self.var_origin,time=self.time),
                                        'gamma':ConditionalDistributionGamma(self.y,self.x_l,self.x_u,scale=self.scale,n_samples=self.n_samples,var_origin=self.var_origin,time=self.time),

                            })

def plot_predictive(x_l,x_u,y,y_u,sample1,sample2,true_params):
    fig,axes = plt.subplots(figsize=(5,5))
    predict_df = pd.DataFrame(columns=['x','y_pred','sigma','true_y'])
    model_weight1 = sample1['theta']
    model_weight2 = sample2['theta']
  
    
    covariate_full1 = list(map(lambda x:radial_basic_function(x_u,x[0],x[1]), zip(sample1['p'],sample1['gamma'])))
    covariate_full2 = list(map(lambda x:radial_basic_function(x_u,x[0],x[1]), zip(sample2['p'],sample2['gamma'])))
    y_pred1 = list(map(lambda x: np.matmul(x[0],x[1].T).flatten(),zip(covariate_full1,model_weight1)))
    y_pred2 = list(map(lambda x: np.matmul(x[0],x[1].T).flatten(),zip(covariate_full2,model_weight2)))

    predict_df_1 = pd.DataFrame({'x':np.tile(x_u,len(model_weight1)),'y_pred':np.concatenate(y_pred1),\
                                         'sigma':['our model']*len(model_weight1) * len(x_u), \
                                        'true_y':np.tile(y_u,len(model_weight1))})
    predict_df_2 = pd.DataFrame({'x':np.tile(x_u,len(model_weight2)),'y_pred':np.concatenate(y_pred2),\
                                         'sigma':['1']*len(model_weight2) * len(x_u),\
                                        'true_y':np.tile(y_u,len(model_weight2)) })
    
    predict_df = pd.concat([predict_df_1,predict_df_2])
    our_model_error = (predict_df[predict_df['sigma']=='our model']['y_pred'] - predict_df[predict_df['sigma']=='our model']['true_y'])**2
    sig_1_model_error = (predict_df[predict_df['sigma']=='1']['y_pred'] - predict_df[predict_df['sigma']=='1']['true_y'])**2
    axes.set_title("Our model Error {:.3f}, {} model error {:.3f}".format(our_model_error.mean()\
                    ,r'$\sigma=1$',sig_1_model_error.mean()))
    axes.plot(x_u,y_u,'ro',label="unlabel data")
    axes.plot(x_l,y,'go',label="label data" )

    pseu_do = np.linspace(min(min(x_u),min(x_l)),max(max(x_l),max(x_u)),100)
    axes.plot(pseu_do,np.matmul(radial_basic_function(pseu_do,true_params['p'],true_params['gamma']),\
                                                        true_params['theta']),'-.',label="true func" )

    sns.lineplot(data=predict_df, x="x", y="y_pred", hue="sigma",ax=axes,ci='sd')
    fig.savefig("result.png")
    return predict_df

def plot_samples_post(samples,true_params=None):
    sites = list(true_params.keys())
    # fig, axs = plt.subplots(nrows=1, ncols=len(sites), figsize=(12, 5))
    fig = plt.figure(figsize=(18, 4))
    outer = gridspec.GridSpec(1,len(sites) , wspace=0.2, hspace=0.2)
    fig.suptitle("Posterior Checking", fontsize=12)
    
    for i, out in enumerate(outer):
        site = sites[i]
        dim_suplot = true_params[site].shape[-1]
        inner = gridspec.GridSpecFromSubplotSpec(1,dim_suplot ,
                    subplot_spec=out, wspace=0.5, hspace=0.1)
        if  dim_suplot==1:
            for j in range(dim_suplot):
                ax = plt.Subplot(fig, inner[j])
                sns.distplot(samples[site], ax=ax, label="posterior")
                ax.axvline(true_params[site],ymin=0,ymax=1,color='red',label='True')
                ax.set_title(site)
                fig.add_subplot(ax)
        else:
            for j in range(dim_suplot):
                print(inner[j])
                ax = plt.Subplot(fig, inner[j])
                sns.distplot(samples[site][:,j], ax=ax, label= f"posterior {site}_{j}")
                ax.axvline(true_params[site][j],ymin=0,ymax=1,color='red',label=f'True {site}_{j}')
                ax.set_title(f"{site}_{j}")
                fig.add_subplot(ax)

    
    fig.savefig("post.png")

def run_experiment_basic_function(m1=-1.,m2=2.,sig1=1.,sig2=2.,
                                a=1,b=2,c=3,n_l=100,n_u=100,
                                p=np.array([1,5,9]),
                                gamma=np.array([1,2,4]),*args,**kwargs):
    
    theta = np.array([a,b,c])
    
    print(theta)
    x_l = np.random.normal(m1,np.sqrt(sig1),size=n_l)
    x_u = np.random.normal(m2,np.sqrt(sig2),size=n_u)
    y = np.random.normal(0,0.3,size=n_l) + np.matmul(radial_basic_function(x_l,p,gamma) ,theta).flatten()  ##a*x_l**2 - c * np.exp(x_l) 
    y_u = np.random.normal(0,0.3,size=n_u) + np.matmul(radial_basic_function(x_u,p,gamma) ,theta).flatten()##a*x_u**2 - c  * np.exp(x_u) 


    init_values = OrderedDict({'theta':[np.random.normal(loc=0,scale=10,size=3)],
                                'phi1':[np.array([np.random.normal(loc=0,scale=10),invgamma.rvs(1/2, loc=0, scale=1/2)])],
                                'phi2': [np.array([np.random.normal(loc=0,scale=10),invgamma.rvs(1/2, loc=0, scale=1/2)])],
                                'p': [np.random.normal(loc=0,scale=10,size=3)],
                                'gamma': [invgamma.rvs(1/2, loc=0, scale=2, size=3)],
    })

    n_samples=5000
    burn_in = 200

    true_params = {'theta':theta,'p':p,'gamma':gamma}
    photometric = TestGibbsSampling(y,x_l,x_u,scale=1.,n_samples=1,var_origin=False,time=0)
    samples = photometric(init_values,n_samples=n_samples,progress_bar=True)
    samples = {k:v[burn_in:] for k,v in samples.items()}

    photometric2 = TestGibbsSampling(y,x_l,x_u,scale=1.,n_samples=1,var_origin=True,time=0)
    samples2 = photometric(init_values,n_samples=n_samples,progress_bar=True)
    samples2 = {k:v[burn_in:] for k,v in samples2.items()}

    plot_predictive(x_l,x_u,y,y_u,samples,samples2,true_params)
    plot_samples_post(samples,true_params)
