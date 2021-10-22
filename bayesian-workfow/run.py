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
# np.random.seed(seed=123)
## load data


def covariate_export(x):
    return np.concatenate([x.reshape(-1,1)**3,-x.reshape(-1,1),np.zeros_like(x).reshape(-1,1)],axis=1)

def covariate_misclassify_export(x):
    return np.concatenate([x.reshape(-1,1),np.ones_like(x).reshape(-1,1),np.zeros_like(x).reshape(-1,1)],axis=1)

from models.gibbs_sampling import ConditionalDistribution,GibbsSampling
from models.metropolis_hasting import MHSampling
from scipy.stats import invgamma,gamma
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,DistanceMetric


### theta is parameter of condiotnal model (prediction model) (a,c and sig)
## phi1 and phi2 are paramter of geneartive model of covariate x  label and unlabel (m1,m2,sig1,sig2)
M = 5
GLOBAL_NEIGH = NearestNeighbors(metric='euclidean',n_neighbors=M)

def log_likelihood(y,x,x_u,theta,phi1,phi2,return_params=False,var_origin=True,neigh=GLOBAL_NEIGH):
     
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
    # _mean = theta[0] *x**2  - theta[1] * np.exp(x) 
    # _mean = np.sum(covariate_export(x) * theta ,axis=1)
    _mean = np.sum(covariate_misclassify_export(x) * theta,axis=1)
    dist_un = (x-phi2[0])**2
    dist_label = (x - mean_label)**2
    if var_origin:
        _var = 1.
    else:
        # neigh.fit(x_u.reshape(-1,1))
        # distance ,idx = neigh.kneighbors(x.reshape(-1,1)) ## shape n_x,M
        # neigh.fit(x.reshape(-1,1))
        # self_distance,_ = neigh.kneighbors() ## shape n_x,M

        # _var = np.sum(distance,axis=1)/np.sum(self_distance,axis=1)
      

        # _var =   1 + np.max(dist)/(dist+np.max(dist))
        # _var = 1.5


        p_l_u = np.exp(log_prob_normal(x,phi2,return_total=False))
        p_l_l = np.exp(log_prob_normal(x,phi1,return_total=False))
        # _var = np.max(p_l_u)/p_l_u * np.sum(distance,axis=1)/np.sum(self_distance,axis=1)
        # _var = 1.
        # _var =  np.max(dist)/dist
        # _var = (dist_un/dist_label)**2
        _var = p_l_l/p_l_u
        # print(f"min var: {min(_var):.3f}, max var: {max(_var):.3f}")

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
            # - np.sum(stats.norm.logpdf(x,loc=y,scale=self.step_size))
        print()
        # print(f"Theta : {s_}")
        return s_

    def jump(self,loc):
        # return stats.uniform(loc=loc/(1 + self.step_size),scale= loc *(1 +self.step_size)-loc/(1 + self.step_size) ).rvs()
        # return np.random.uniform(low=loc/(1 + self.step_size),high=loc *(1 +self.step_size))
        return np.random.normal(loc=loc ,scale=self.step_size)
        # return np.random.normal(loc=loc,scale=self.step_size)

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
                -   stats.uniform.logpdf(x[1],loc=y[1]/(1 + self.step_size),scale =y[1] *(1 +self.step_size)-y[1]/(1 + self.step_size))\
                                #   - stats.gamma.logpdf(x[1],y[1])\

                # -invgamma.logpdf(x[1],y[0])
                
                                  
                                                    #  - stats.norm.logpdf(x[0],loc=y[0],scale=self.step_size)\

        # print(f"Phi {s_}")
        return s_
    def jump(self,loc):
        # return stats.uniform(loc=loc/(1 + self.step_size),scale= loc *(1 +self.step_size)-loc/(1 + self.step_size) ).rvs()
        mean = loc[0]
        var = loc[1]
        # ran_mean = np.random.normal(loc=mean,scale=self.step_size)
        ran_mean = np.random.normal(loc=mean,scale=self.step_size)
        ran_var = np.random.uniform(low=var/(1 + self.step_size),high=var *(1 +self.step_size))
        # ran_var =  invgamma.rvs(var)
        # ran_var = np.random.gamma(var)

        return np.array([ran_mean,ran_var])
        # return np.random.uniform(low=loc/(1 + self.step_size),high=loc *(1 +self.step_size))
        # return np.random.normal(loc=loc,scale=1)

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
    def prob_func(self,phi1,phi2):

        def score_func(x):
            log_prob = log_prob_normal(x,np.array([0,100]))
            llh = log_likelihood(self.y,self.x_l,self.x_u,x,phi1,phi2,var_origin=self.var_origin)
            if self.time >0:
                print(f"prior: {log_prob:.3f}, Liklihood: {llh:.3f} ")
                time.sleep(self.time)
            return log_prob + llh

        return  score_func

    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(phi1=self.phi1,phi2=self.phi2))

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

    def prob_func(self,theta,phi2):
        def score_func(x):
            log_prob = log_prob_normal(self.x_l,x)
            prior = log_prob_phi(x)
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,x,phi2,var_origin=self.var_origin)
            if self.time > 0:
                print(f"Marginal x_u: {log_prob:.3f}")
                time.sleep(self.time)
            return log_prob + prior + llh

        return score_func
                        
    def set_params(self,**kwargs):
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi2=self.phi2))
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

    def prob_func(self,theta,phi1):
        def score_func(x):
            log_prob = log_prob_normal(self.x_u,x)
            llh = log_likelihood(self.y,self.x_l,self.x_u,theta,phi1,x,var_origin=self.var_origin)
            prior = log_prob_phi(x)
            if self.time > 0:
                print(f"Liklihood: {llh:.3f}, marginal x_u: {log_prob:.3f}")
                time.sleep(self.time)
            return log_prob + llh + prior

        return score_func
                        
    def set_params(self,**kwargs):
        ## update sampling params in gibbs (theta,phi1,phi2)
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(theta=self.theta,phi1=self.phi1))
        pass 
    def sample(self):
        sample_ = self.generator(self.phi2,n_samples=self.n_samples,progress_bar=False)[-1]
        # print(f"sample from phi2 {sample_}")

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
                           
                            })

def plot_dist(samples,x_l,x_u,name=''):
        fig,axes = plt.subplots(ncols=4,figsize=(15,3))

        for idx in range(len(axes)):
            if idx==0:
                sns.histplot([x_l,x_u],ax=axes[idx],kde=True,stat="density")
                axes[idx].set_title(r'x')
                legend = axes[idx].get_legend()
                handles = legend.legendHandles
                legend.remove()
                axes[idx].legend(handles, [r'$x_{L}$', r'$x_{U}$'], title=r'data')

            if idx == 1:
                theta = samples['theta']

                sns.histplot([[t[0] for t in theta],[t[1] for t in theta],[t[2] for t in theta]],ax=axes[idx],kde=True,stat="density")
                axes[idx].set_title(r"$\theta$ ( a={:.3f},b={:.3f}, c={:.3f})".format(*np.mean(samples['theta'],axis=0)))
                legend = axes[idx].get_legend()
                handles = legend.legendHandles
                legend.remove()
                axes[idx].legend(handles, ['a','b', 'c'], title=r'$\theta$ params')
            

            elif idx == 2:
                phi1 = samples['phi1']
                
                sns.histplot([[t[0] for t in phi1],[t[1] for t in phi1]],ax=axes[idx],kde=True,stat="density")
                axes[idx].set_title("{} mean {}={:.1f},mean {}={:.1f})".format(r'$\psi_{L}$',r'$\mu_{L}$',np.mean(samples['phi1'],axis=0)[0],r'$\sigma^{2}_{L}$',np.mean(samples['phi1'],axis=0)[1]))
                legend = axes[idx].get_legend()
                handles = legend.legendHandles
                legend.remove()
                axes[idx].legend(handles, [r'$\mu_{L}$', r'$\sigma^{2}_{L}$'], title=r'$\psi_{L}$ params')
                
            
            elif idx == 3:
                phi2 = samples['phi2']
                sns.histplot([[t[0] for t in phi2],[t[1] for t in phi2]],ax=axes[idx],kde=True,stat="density")
                axes[idx].set_title("{} (mean {}={:.1f},mean {}={:.1f})".format(r'$\psi_{U}$',r'$\mu_{U}$',np.mean(samples['phi2'],axis=0)[0],r'$\sigma^{2}_{U}$',np.mean(samples['phi2'],axis=0)[1]))

                legend = axes[idx].get_legend()
                handles = legend.legendHandles
                legend.remove()
                axes[idx].legend(handles, [r'$\mu_{U}$', r'$\sigma^{2}_{U}$'], title=r'$\psi_{U}$ params')
            
        fig.savefig(f"{name}.png")

def run_exp(m1=-2,m2=0.5,sig1=1.,sig2=1.,a=1,b=2,c=3,n_l=100,n_u=100,path_fig=''):

    # m1,m2 = -2,0.5
    # sig1,sig2 = 1.,1.
   
    # a,b,c =1,2,3
    theta = np.array([a,b,c])
    print(theta)
    x_l = np.random.normal(m1,np.sqrt(sig1),size=n_l)
    x_u = np.random.normal(m2,np.sqrt(sig2),size=n_u)
    print(covariate_export(x_l).shape)
    y = np.random.normal(0,0.3,size=n_l) + np.sum(covariate_export(x_l) * theta ,axis=1)  ##a*x_l**2 - c * np.exp(x_l) 
    y_u = np.random.normal(0,0.3,size=n_u) + np.sum(covariate_export(x_u) * theta ,axis=1)##a*x_u**2 - c  * np.exp(x_u) 

    init_values = lambda x: OrderedDict({'theta':[np.random.normal(loc=0,scale=10,size=3)],
                                'phi1':[np.array([np.random.normal(loc=0,scale=10),invgamma.rvs(1/2, loc=0, scale=1/2)])],
                                'phi2': [np.array([np.random.normal(loc=0,scale=10),invgamma.rvs(1/2, loc=0, scale=1/2)])],
    })

    n_samples=30000
    photometric = TestGibbsSampling(y,x_l,x_u,scale=1.,n_samples=1,var_origin=False,time=0)
    samples = photometric(init_values,n_samples=n_samples,progress_bar=True)
    burn_in = 1
    samples = {k:v[burn_in:] for k,v in samples.items()}
    print(f"Mean of theta: {np.mean(samples['theta'],axis=0)}")
    print(f"Mean of phi1: {np.mean(samples['phi1'],axis=0)}")
    print(f"Mean of phi2: {np.mean(samples['phi2'],axis=0)}")

    # prediction
    mean_a,mean_b,mean_c  = np.mean(samples['theta'],axis=0)

    

    plot_dist(samples,x_l,x_u,name=path_fig+ '/' +str(n_samples)+'_v2')
    ### test with var_origin
    photometric = TestGibbsSampling(y,x_l,x_u,scale=1.,n_samples=1,var_origin=True)
    samples_origin = photometric(init_values,n_samples=n_samples,progress_bar=True)
    samples_origin = {k:v[burn_in:] for k,v in samples_origin.items()}

    print(f"Mean of theta: {np.mean(samples_origin['theta'],axis=0)}")
    print(f"Mean of phi1: {np.mean(samples_origin['phi1'],axis=0)}")
    print(f"Mean of phi2: {np.mean(samples_origin['phi2'],axis=0)}")
    mean_a_or,mean_b_or,mean_c_or  = np.mean(samples_origin['theta'],axis=0)

    plot_dist(samples_origin,x_l,x_u,name=path_fig+ '/'+ str(n_samples)+'_sigma1')

    fig,axes = plt.subplots(figsize=(5,5))
    predict_df = pd.DataFrame(columns=['x','y_pred','sigma'])
    model_params = np.array(samples['theta'])
    model_1_params = np.array(samples_origin['theta'])
    covariate =  covariate_misclassify_export(x_u) ###np.concatenate([(x_u**2).reshape(-1,1),-np.exp(x_u).reshape(-1,1)],axis=1)
    our_model_error = np.mean((y_u - np.sum(covariate * np.mean(model_params,axis=0,keepdims=True),axis=1))**2)#0.0
    sig_1_model_error = np.mean((y_u - np.sum(covariate * np.mean(model_1_params,axis=0,keepdims=True),axis=1))**2)#0.0

    print(f"model 1 error: {our_model_error}")
    print(f"model 2 error: {sig_1_model_error}")

    for i in range(len(x_u)):
        predict_df = pd.concat([predict_df,pd.DataFrame({'x':np.tile(x_u[i],len(model_params)),'y_pred':np.sum(model_params * covariate[i],axis=1),
                                        'sigma':['our model']*len(model_params)})])
        # our_model_error += np.sum((y_u[i] - np.sum(model_params * covariate[i],axis=1))**2)
        predict_df = pd.concat([predict_df,pd.DataFrame({'x':np.tile(x_u[i],len(model_1_params)),'y_pred':np.sum(model_1_params * covariate[i],axis=1),
                                        'sigma':['1']*len(model_1_params)})])
        # sig_1_model_error += np.sum((y_u[i] - np.sum(model_1_params * covariate[i],axis=1))**2)

    axes.set_title("Our model Error {:.3f}, {} model error {:.3f}".format(our_model_error\
                    ,r'$\sigma=1$',sig_1_model_error))
    axes.plot(x_u,y_u,'ro',label="unlabel data")
    axes.plot(x_l,y,'go',label="label data" )
    pseu_do = np.linspace(-0.5,1.5,100)
    axes.plot(pseu_do,np.sum(covariate_export(pseu_do) * theta ,axis=1),'-.',label="true func" )
    sns.lineplot(data=predict_df, x="x", y="y_pred", hue="sigma",ax=axes,estimator = np.median,ci='sd')
    

    fig.savefig(path_fig+ '/'+f'prediction_{n_samples}.png')

m1=0.5
# m2= 4.
sig1=0.5**2
# sig2=4.
a=1
b=1.
c=1.
list_m2 = [ 0.0]
list_sig2 = [0.3**2]
for m2,sig2 in product(list_m2,list_sig2):
    name_folder = f"fig/miss_pl_cubic_linear_m1={m1}_m2={m2}_sig1={sig1}_sig2={sig2}_a={a}_b={b}_c={c}"
    os.makedirs(name_folder,exist_ok=True)
    run_exp(m1,m2,sig1,sig2,a,b,c,path_fig=name_folder)




