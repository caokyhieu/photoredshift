import numpy as np
import scipy.stats as stats
import pandas as pd
from abc import ABC,abstractmethod
from tqdm import tqdm,trange
from models.metropolis_hasting import MetropolisHasting,MHSampling
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

        # return {key:samples[key][1:] for key in self.conditional}
        ## return numpy array
        return {key:np.array(samples[key][1:]) for key in self.conditional}
