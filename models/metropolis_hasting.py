import numpy as np
import scipy.stats as stats
from  scipy.special import gamma as gam_f
from  scipy.special import loggamma as log_gam_f
import matplotlib.pyplot as plt
# from utils import plot_hist
from tqdm import tqdm
import pdb
from abc import ABC,abstractmethod
from tqdm import tqdm,trange

class MHSampling(ABC):
    def __init__(self,step_size=10.0):
       
        self.accept = 0
        self.step_size = step_size
        pass 
    
    @abstractmethod
    def score(self,x,y):
        """
        this method will return score for each value in each steps
        """

    @abstractmethod
    def jump(self,x):
        """
        This method will give proposal for each step
        """
   
    def __call__(self,init_value,n_samples=10000,progress_bar=False):

        _range = tqdm(range(n_samples)) if progress_bar else range(n_samples)
        sample = []
        old_val = init_value
        new_val = self.jump(old_val)
    
        for i in _range:
            old_score = self.score(old_val,new_val)
            new_score = self.score(new_val,old_val)
            if np.log(np.random.uniform()) < (new_score - old_score):
                old_val = new_val
                # old_score = new_score 
                self.accept+=1
                if progress_bar:
                    _range.set_description(f"Accepted rate: {self.iter/i * 100:.2f}")
                # print('Accpeted')
            
            # print(new_score - old_score)
            sample.append(old_val)
            new_val =  self.jump(old_val)
        if progress_bar:
            _range.close()

        return sample


class MetropolisHasting(MHSampling):
    
    def __init__(self,target,scale=10.0):
        super().__init__(scale)

        """
        target: target log pdf function. with parameter x, return pdf of x value.
                  It has func logpdf(val) to return pdf of value and rvs to random value 
        n_sample : num_sample
        """

        self.target = target

    def set_target(self,target):
        self.target = target

    def score(self,x,y):
        return self.target(x) - stats.uniform.logpdf(x,loc=y/(1 + self.step_size),scale =y *(1 +self.step_size)-y/(1 + self.step_size))

    def jump(self,loc):
        # return stats.uniform(loc=loc/(1 + self.step_size),scale= loc *(1 +self.step_size)-loc/(1 + self.step_size) ).rvs()
        return np.random.uniform(low=loc/(1 + self.step_size),high=loc *(1 +self.step_size))


    # def uniform_proposal(self, loc, scale=30.0):
    #     """
    #     This proposal use for non negative distribution
    #     """
    #     if loc <scale/2:
    #         return stats.uniform(loc=0,scale= scale)
    #     else:
    #         return stats.uniform(loc=loc-scale/2,scale= scale) 
    
    # def normal_proposal(self, loc, scale=30.0):
    #     """
    #     This proposal use for R distribution
    #     """
    #     return stats.norm(loc=loc,scale=scale)

    # def truncate_proposal(self,loc=0,scale=1):
    #     """
    #     This proposal used for interval support distribution .ex Beta in (0,1)
    #     """
    #     return stats.uniform(loc=loc,scale=scale)
    
    # def general_uniform(self,loc,scale):

    #     return stats.uniform(loc=loc/(1 + scale),scale= loc *(1 +scale)-loc/(1 + scale) ) 

    # def score(self,new_x,old_x):

    #     r = self.target(new_x) -  self.target(old_x)
    #     return r

    

# def mixture_gaussian(x):
#     if x > 3:
#         return stats.norm(loc=10,scale=2).logpdf(x)
#     else:
#         return stats.norm(loc=0,scale=2).logpdf(x)

# if __name__ == '__main__':

#     # new_target = lambda x: stats.gamma(100,scale=2).logpdf(x)
#     # mh_sampling = MetropolisHasting(mixture_gaussian,-10.,100000,scale=1.,progress_bar=True,typ='R')
#     # mh_sampling.process()
#     # plot_hist(np.array(mh_sampling.sample),stats.beta,'beta_new.png',nbins=50)

#     # test gibbs sampling by multivariate normal

#     ## p(x1,x2) ~ MVN([1,1],[[1,1/4],[1/4,1]])
#     data = stats.multivariate_normal(mean=[1,1],cov=[[1,1/4],[1/4,1]]).rvs(100)
#     ## we know that conditional x1|x2 ~ N(m1 + sigma1/sigma2 * ro*(x2-m2),(1-ro^2)sigma1^2)
#     print(data.shape)

#     init_x1 = 4.0
#     init_x2 = 4.0
#     n_samples=10000
#     scale = 4.0
#     # start_point = .0
#     result = []
#     alpha_accept = 0
#     beta_accept = 0
#     with trange(n_samples) as t:
#         for i in t:
#             ### sampling conditional x2|x1 ~ N(m2 + sigma2/sigma1 * ro *(x1-m2),(1-ro^2)*sigma2^2)
#             # c =10.0
#             # jumps = lambda x: stats.norm(loc=x,scale=c)
#             target_x2 = lambda x:  - 1/(2 * (1-(1/4)**2)) * (x- 1 - 1/4 *(init_x1 - 1) )**2
#             mh = MetropolisHasting(target_x2,scale=scale)
#             init_x2_ = mh(init_x2,n_samples=1,progress_bar=False)
#             if init_x2 !=  init_x2_[-1]:
#                 beta_accept+=1
#             init_x2 = init_x2_[-1]

#             target_x1 = lambda x:  - 1/(2 * (1-(1/4)**2)) * (x- 1 - 1/4 *(init_x2 - 1) )**2
#             mh.set_target(target_x1)
#             init_x1_= mh(init_x1,n_samples=1,progress_bar=False)
#             if init_x1 !=  init_x1_[-1]:
#                 alpha_accept+=1
#             init_x1 = init_x1_[-1]
#             t.set_description(f'Rate a:{alpha_accept/(i+1):.2f}, Rate b: {beta_accept/(i+1):.2f} ')
#             result.append([init_x1,init_x2])
    
#     plot_hist(np.array(result)[:,0],stats.norm,'x1_multi_variate.png',nbins=20)
#     plot_hist(np.array(result)[:,1],stats.norm,'x2_multi_variate.png',nbins=20)
#     plt.scatter(np.array(result)[:,0],np.array(result)[:,1])
#     plt.savefig('MVN.png')
#     # dat = pd.DataFrame( np.array(result),index=[i for i in range(len(result))])
#     # sns.kdeplot(dat,x=0,y=1)
#     # plt.savefig('multi_variate.png')
                
                


