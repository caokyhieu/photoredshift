import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LocalCoverage:

    def __init__(self,estimate,num_alpha=10):
        """
        Require:
        (X,Y) in that we want to check p(Y|X)
        Model to estimate f^(y|x)
        test point x
        estimate: estimate pdf(y|x), with parameter is X, have pit func (y,x) return conditional cdf
        ######
        This method is quite suitable for gibbs sampling,
        where we have to model the conditional distribution to sample to join distribution (x,y)
        """
        self.alpha = np.linspace(0.01,1. ,num_alpha)
        self.estimate = estimate
        pass

    
    def fit(self,data):
        """
        Data: (X,Y)
        """
