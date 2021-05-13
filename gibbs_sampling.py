import numpy as np
import scipy.stats as stats
import pandas as pd
from utils import plot_hist
from abc import ABC,abstractmethod
from tqdm import tqdm,trange
from metropolis_hasting import MetropolisHasting,MHSampling
from  scipy.special import gamma as gam_f
from  scipy.special import loggamma as log_gam_f
import matplotlib.pyplot as plt
class ConditionalDistribution(ABC):
    
    @abstractmethod
    def set_params(self,**kwargs):
        """
        Each conditional distribution has this function 
        """
    
    @abstractmethod    
    def sample(self):
        """
        This function to random 
        """

class ConditionalTheta(ConditionalDistribution):
    def __init__(self,data):
        self.data = data
        pass 

    def set_params(self,**kwargs):
        self.generator = np.random.beta
        self.__dict__.update(kwargs)
     
    def sample(self):
        return list(map(lambda x: self.generator(self.alpha + x[1],self.beta + x[0] - x[1]), self.data))


class ConditionalAlpha(ConditionalDistribution):
    def __init__(self,len_data,a1,a2,scale=1.0,n_samples=100):
        self.n = len_data
        self.scale = scale
        self.a1 = a1
        self.a2 = a2
        self.n_samples = n_samples
        self.generator = MetropolisHasting(None,scale=self.scale)
        pass 

    def prob_func(self,beta,theta):
        return  lambda x: self.n * log_gam_f(x + beta)\
                            - self.n * log_gam_f(x)\
                            - self.a2*x\
                            + (self.a1-1) * np.log(x)\
                            + x * np.sum(np.log(theta))\

    
    def set_params(self,**kwargs):
        # self.generator = stats.beta
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(beta=self.beta,theta=self.theta))
        # self.generator =  MetropolisHasting(self.prob_func(beta=self.beta,theta=self.theta),scale=1.)
     
    def sample(self):
        return self.generator(self.alpha,n_samples=self.n_samples,progress_bar=False)[-1]


class ConditionalBeta(ConditionalDistribution):
    def __init__(self,len_data,b1,b2,scale=1.0,n_samples=100):
        self.n = len_data
        self.scale = scale
        self.b1 = b1
        self.b2 = b2
        self.n_samples = n_samples
        self.generator =  MetropolisHasting(None,scale=1.)
        # self.generator =  MetropolisHasting(alpha_func,scale=1.)
        pass 

    def prob_func(self,alpha,theta):
        return  lambda x: self.n * log_gam_f(alpha + x) \
                        - self.n * log_gam_f(x)\
                        - self.b2*x\
                        + (self.b1-1) * np.log(x)\
                        + x * np.sum(np.log(1 - np.array(theta)))

    
    def set_params(self,**kwargs):
        # self.generator = stats.beta
        self.__dict__.update(kwargs)
        self.generator.set_target(self.prob_func(self.alpha,self.theta)) 
        # =  MetropolisHasting(self.prob_func(self.alpha,self.theta),scale=1.)
     
    def sample(self):
        return self.generator(self.beta,n_samples=self.n_samples,progress_bar=False)[-1]

class GibbsSampling(ABC):

    def __init__(self):
        """
        initial_params: init val for parameters

        """
        self.conditional = None
        pass
    
    @abstractmethod
    def set_conditional(self):
        """
        This method set dict of conditional distribution
        each of them can sample current value given all previous 
        """
    
    def __call__(self,init_value,n_samples=10000,progress_bar=False):
        """
        init_value: dict of block params

        """
        assert len(set(self.conditional.keys()).intersection(set(init_value.keys()))) == len(self.conditional)
        _range = trange(n_samples) if progress_bar else range(n_samples)
        samples = init_value
        # n = len(init_value)
        for i in _range:
            for key,cond in self.conditional.items():
                cond.set_params(**{k:samples[k][-1] for k in samples})

                new_val = cond.sample()
                samples[key].append(new_val)

                if issubclass(cond.generator.__class__,MHSampling) and progress_bar:
                    _range.set_description(f'Rate {key}: {cond.generator.accept/(i+1):.2f}')
                    

        if progress_bar:
            _range.close()

        return {key:samples[key][1:] for key in self.conditional}

class HospitalGibbsSampling(GibbsSampling):

    def __init__(self,data,a1,a2,b1,b2,scale=0.1):
        self.data = data
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        # self.n_samples=n_samples
        self.set_conditional()
        pass

    def set_conditional(self):
        self.conditional = {'theta':ConditionalTheta(self.data),
                            'alpha':ConditionalAlpha(len(self.data),self.a1,self.a2,scale=1.0,n_samples=1),
                            'beta':ConditionalBeta(len(self.data),self.b1,self.b2,scale=1.0,n_samples=1)
                            }
        # print(self.conditional)

if __name__ =='__main__':

    ## test
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
    n_samples=10000

    hospital = HospitalGibbsSampling(data,a1,a2,b1,b2)
    init_values = {'alpha':[alpha],'beta':[beta],'theta': [[0.5 for i in range(len(data))]]}
    samples = hospital(init_values,n_samples=n_samples,progress_bar=True)
    # print(samples)
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(samples['alpha'], bins=n_bins)
    axs[0].axvline(x=np.mean(samples['alpha']),color="red")
    axs[0].set_xlabel('Alpha')
    axs[0].set_ylabel('Freq')
    axs[1].hist(samples['beta'], bins=n_bins)
    axs[1].axvline(x=np.mean(samples['beta']),color="red")
    axs[1].set_xlabel('Beta')
    axs[1].set_ylabel('Freq')
    fig.savefig('gibbs_sample_1.png')
    plt.close('all')

    ## plot alpha,beta
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True,figsize=(10,5))
    axs[0].plot(samples['alpha'], 'b-',label='alpha') 
    axs[1].plot(samples['beta'], 'r-',label='beta')
    axs[0].set_xlabel('Iter')
    axs[0].set_ylabel('Val')
    axs[0].legend()
    axs[1].set_xlabel('Iter')
    axs[1].set_ylabel('Val')
    axs[1].legend()
    fig.savefig('gibbs_alpha_beta_1.png')
    plt.close('all')
    