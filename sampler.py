import numpy as np
from  scipy.special import gamma as gam_f
from  scipy.special import loggamma as log_gam_f


from scipy.stats import uniform,gamma
import matplotlib.pyplot as plt
from utils import plot_hist
import scipy.stats as stats
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import pdb
from metropolis_hasting import MetropolisHasting
### use for hierrachical model
## how to sampling

def gibbs_sampler(conditional,initial_val,n_sample):
    """
    Gibbs sampling use for multidimension problem
    Gibbs sampling require to have all conditional distribution to sample eternatively
    conditional: list full conditional distribution, it have ability to sample by the parameter
    initial_val: initial value for gibbs samp. We will sample by the order in conditional 
    """

    
    pass



def important_sampling(data,params,weight_func,mean=False):
    """
    params: array of alpha,beta
    data: data for each hospital
    weight_func: function to compute weight w_i, need proposal and target params for this func
    """

    weights = np.array(list((map(weight_func(data),params))))
    weights = weights/np.sum(weights)
    if not mean:
        return weights
    else:
        return np.sum(np.array(weights) * params)
    
def weight_func_generator(data):
    """
    data: array of data hostpital (n,k)
    """
    def alpha_beta(par):
        alpha = par[0]
        beta = par[1]
        "func to calculate prob w"
        gamma_func = lambda x: gam_f(alpha+beta)/(gam_f(alpha) * gam_f(beta)) * gam_f(alpha+x[1]) * gam_f(beta + x[0] - x[1])/ gam_f(alpha + beta + x[0])
        return np.prod(np.array(list(map(gamma_func,data))))
    return alpha_beta

def sampling_alpha_beta(a1,a2,b1,b2,num_samples=100):
    """
    a1,a2,b1,b2: hyperprior for alpha beta
    """
    prior_0 =gamma.rvs(a=a1,loc=0,scale=1/a2,size=num_samples)
    prior_1 = gamma.rvs(a=b1,loc=0,scale=1/b2,size=num_samples)
    return np.array([prior_0,prior_1]).T

def sampling_post_alpha_beta(data,a1,a2,b1,b2,num_samples=1000):
    prior_alpha_beta = sampling_alpha_beta(a1,a2,b1,b2,num_samples)

    pos_weight = important_sampling(data,prior_alpha_beta,weight_func_generator)
    # result = []
    # print(pos_weight)
    ### random pick  one tuple (alpha,beta)
    pos_weight = np.cumsum(pos_weight)
    u = uniform.rvs(loc=0, scale=1, size=1)
    idx = 0
    for val in pos_weight:
        if u > val:
            idx+=1
        else:
            break
            # result.append(prior_alpha_beta[i])
    # print(f'Acceptance rate: {len(result)/num_samples:0.3f}')

    return prior_alpha_beta[idx:idx+1],idx

def sampling_post_theta(data,alpha_beta):
    result = []
    for alpha,beta in alpha_beta:
        theta = []
        for d in data:
            theta.append(np.random.beta(alpha + d[1],beta + d[0] - d[1]))
        result.append(theta)
    
    return result

def rejected_sampling(proposal,target,M,n_sample=1000):
    """
    require proposal and target have same support
    target > 0 every x belong proposal
    We can easily sample from proposal and M * proposal > target 
    we can evaluate target/ (proposal*M)
    we can use proposal as uniform [samp/(1+c),samp(1+c)]
    """
    result = []
    i = 0
    t = 0
    # samp = 0.5
    while i<n_sample:
        # proposal = uniform(samp/(1+c),samp*(1+c))
        samp = proposal.rvs(size=1)
        t+=1
        v =  target.pdf(samp)/(M * proposal.pdf(samp))
        if uniform.rvs(size=1) < v:
            result.append(samp)
            i+=1
    print(f'Accepted rate: {i/t:0.3f}')

    return result

def alpha_logprob(data,beta,theta,a1,a2):
    n = len(theta)
    n_arr = np.array([i[0] for i in data])
    k_arr = np.array([i[1] for i in data])

    def _logprob(x):
        return  n * log_gam_f(x + beta)\
                - n * log_gam_f(x)\
                - a2*x\
                + (a1-1) * np.log(x)\
                + x * np.sum(np.log(theta))\
                # + np.sum(log_gam_f(x + k_arr) - log_gam_f(x + beta + n_arr))\


    return _logprob

def beta_logprob(data,alpha,theta,b1,b2):
    n = len(theta)
    n_arr = np.array([i[0] for i in data])
    k_arr = np.array([i[1] for i in data])
    def _logprob(x):
        return  n * log_gam_f(alpha + x) \
                - n * log_gam_f(x)\
                - b2*x\
                + (b1-1) * np.log(x)\
                + x * np.sum(np.log(1 - np.array(theta)))\
                # + np.sum(log_gam_f(x + n_arr - k_arr) - log_gam_f(alpha + x + n_arr))\


    return _logprob



def gibbs_sampling_hospital(initial_val,hyper_prior,data,n_sample=10000):
    """
    initial_val: initial_val for alpha and beta
    hyper_prior: hyper prior for alpha and beta
    data: data from hospital
    """
    result = []
    inter_alpha,inter_beta = initial_val
    a1,a2,b1,b2 = hyper_prior
    t_a = 0
    t_b = 0
    n = len(data)
    # start_point = 20.0
    ## sampling for theta|(alpha,beta,data)
    for i in tqdm(range(n_sample)):
    # while len(result) < n_sample:
        pos_theta = sampling_post_theta(data,[(inter_alpha,inter_beta)])
        
        ## define jumps distribution for metropolis
        # c = 2.0
        # # jumps = lambda x: uniform(loc= x/(1 + c),scale=x *(1+c) - x/(1 + c))
        # jumps = lambda x: stats.uniform(loc=0,scale= c) if x<c/2 else stats.uniform(loc=x-c/2,scale= c)
        # jumps = lambda x: stats.gamma(x,1)

        ### sampling for new alpha,beta|(theta,data)
        ### using metropolis sampling
        ### first define p(alpha|beta,theta,data) by consider the join p(alpha,beta,theta|data) and fix beta and theta
     
        alpha_func = alpha_logprob(data,inter_beta,pos_theta[0],a1,a2)
        mh = MetropolisHasting(alpha_func,scale=1.)
        inter_alpha= mh(inter_alpha,n_samples=100,progress_bar=False)
        inter_alpha = inter_alpha[-1]
        t_a+=0

        # alpha_,ta = metropolis_sampling(jumps,alpha_func,inter_alpha,n_sample=1)
        # inter_alpha = alpha_[-1]
        # t_a+=ta

        ### define p(beta|alpha,theta,data)
   
        beta_func = beta_logprob(data,inter_alpha,pos_theta[0],b1,b2)
        mh.set_target(beta_func)
        inter_beta= mh(inter_beta,n_samples=100,progress_bar=False)
        inter_beta = inter_beta[-1]
        t_b+=0
        
        # beta_,tb = metropolis_sampling(jumps,beta_func,inter_beta,n_sample=1)
        # t_b+=tb
        # inter_beta = beta_[-1]


        result.append([[inter_alpha,inter_beta],pos_theta[0]])
    
    return result,t_a,t_b



if __name__ == "__main__":
     # data
    data = [(15,7),(2,2),(20,5),(35,23),(13,10),(1,0),(19,6),(27,18),(21,10),(43,31)]
    ## sampling post alpha,beta
    ## hyper prior
    a1 = 1.0
    a2 = 1/10.0
    b1 = 1.0
    b2 = 1/10.0
    beta = 1.0
    alpha = 0.57/0.43
    # beta = 15.0
    # alpha = 15.0
    n_bins = 10
    n_samples=5000
    # # ## test gibbs sampling in data
    result,t_a,t_b = gibbs_sampling_hospital((alpha,beta),(a1,a2,b1,b2),data,n_sample=n_samples)
    theta = np.array([i[1] for i in result])
    # print(theta.shape)
    result = np.array([i[0] for i in result])
   
    print(f'Accept rate a: {t_a/(n_samples):.3f}, Accept rate b: {t_b/(n_samples):.3f}')
    print(f'Mean of theta: {np.mean(theta,axis=0)}')
    print(f'Mean of alpha: {result[:,0].mean():.3f}, Mean of beta: {result[:,1].mean():.3f}')
    
    # ### test gibbs sampling
    
    # # plot_hist(np.array([i[0][1] for i in result]),stats.gamma,'beta_sample.png',nbins=10)
    
    # # t = 5000
    # # result = []
    # # rate= []
    # # for ti in range(t):
    # #     sample,idx = sampling_post_alpha_beta(data,a1,a2,b1,b2,num_samples=100)
    # #     if len(sample)>0:
    # #         result.append(sample)
    # #         rate.append((idx+1)/100)

    # # result = np.vstack(result)
    # # print(result.shape)
    # # theta = np.array(sampling_post_theta(data,result))
    # # print(f'Acceptance rate alpha,beta: {np.mean(rate)}')
    # # print(f'Mean of theta: {np.mean(theta,axis=0)}')
    # # print(f'Mean of alpha: {result[:,0].mean()}, Mean of beta: {result[:,1].mean()}')


    # ### plot sampling

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(result[::2,0], bins=n_bins)
    axs[0].axvline(x=np.mean(result[:,0]),color="red")
    axs[0].set_xlabel('Alpha')
    axs[0].set_ylabel('Freq')
    axs[1].hist(result[::2,1], bins=n_bins)
    axs[1].axvline(x=np.mean(result[:,1]),color="red")
    axs[1].set_xlabel('Beta')
    axs[1].set_ylabel('Freq')
    fig.savefig('gibbs_sample_1.png')
    plt.close('all')

    ## plot alpha,beta
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True,figsize=(10,5))
    axs[0].plot(result[:,0], 'b-',label='alpha') 
    axs[1].plot(result[:,1], 'r-',label='beta')
    axs[0].set_xlabel('Iter')
    axs[0].set_ylabel('Val')
    axs[0].legend()
    axs[1].set_xlabel('Iter')
    axs[1].set_ylabel('Val')
    axs[1].legend()
    fig.savefig('gibbs_alpha_beta_1.png')
    plt.close('all')
    

   

    




   



        










