import numpy as np
import scipy.stats as stats
import pandas as pd
from abc import ABC,abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt

class ABCSampling(ABC):
    def __init__(self,data,threshold):
        
        self.threshold = threshold
        self.data = data
        self.accept = 0
        pass
    
    @abstractmethod
    def set_prior(self):
        """
        This method helps to set prior for object.
        This prior can sample proposal theta by sample method
        """

    @abstractmethod
    def generate_replica(self):
        """
        This method helps to generate replica data from proposal p(theta) from prior
        This method return replicated data
        parameters: proposal theta
        """ 

    @abstractmethod
    def distance(self,data,replica):
        """
        This method helps to calculate statistics and compare the distance between replica data and original data.
        input of this method is data and replica data
        output is the distance
        First try with Wassenstein Distance
        """
    
    def sample(self,n_samples,progress_bar=False):
        _range = tqdm(range(n_samples)) if progress_bar else range(n_samples)

        samples = []
        for i in _range:
            proposal = self.prior.sample(n_samples=None)
            replica = self.generate_replica(proposal)
            ### check proposal valid or not
            if self.distance(self.data,replica) < self.threshold:
                samples.append(proposal)
                self.accept +=1
    
        return samples        

class WassersteinABC(ABCSampling):

    def __init__(self,data,threshold):
        super().__init__(data,threshold)
        
        self.set_prior()
    
    def set_prior(self):
        self.prior = type('Prior', (object,), {'sample' : lambda n_samples: np.random.normal(loc=0,scale=1,size=n_samples)})

    def generate_replica(self,theta):
        # thetas = self.prior.sample(n_samples=len(self.data))
        return np.random.uniform(low=theta - 2.0,high=theta + 2.0,size=len(self.data))

    def distance(self,data,replica):
        """
        Just use for one dimension data
        Wasserstein Distance very effective for capture distribution
        Improve by slice wasserstein distance: very impressive
        * Next move for multivariate data (2 dims)
        """
        weight = np.random.multivariate_normal(mean=np.random.normal(size=4),cov = np.eye(4))
        data = np.matmul(np.expand_dims(data,-1),np.expand_dims(weight,0))
        replica = np.matmul(np.expand_dims(replica,-1),np.expand_dims(weight,0))
      
        result = [stats.wasserstein_distance(data[:,i],replica[:,i]) for i in range(len(weight))]

        return np.mean(result)
        # return np.abs(np.mean(data) - np.mean(replica)) + np.abs(np.std(data) - np.std(replica))

# if __name__ == '__main__':
#     ### prior ~ N(0,1)
#     data = [np.random.normal(loc=theta,scale= 2.0) for theta in np.random.normal(loc=0,scale=1,size=1000)]
#     ##-> posterior has mean = mean(data) * n /(sigma^2 + n)
#     threshold = 1.2
#     posterior = WassersteinABC(data,threshold)
#     samples = posterior.sample(n_samples=100000,progress_bar=True)
#     print(np.array(samples).shape)
#     # print(samples)
#     print(f'Accepted rate: {len(samples)/100000 * 100:.2f}%')
#     plt.hist(np.array(samples).squeeze(),bins=20)
#     plt.axvline(x=np.mean(samples),color="red")
#     plt.axvline(x=np.mean(data)*len(data)/(4+len(data)),color="purple")
#     plt.savefig('ABC.png')




            

            


        
    
