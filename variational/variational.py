import numpy as np
import pandas as pd
from scipy.special import digamma,beta

class CAVIPhotoz:
    """
    Coordinate Ascent variational inference 
    """

    def __init__(self,data,kdim,bin_edge,template,a,S_N_ratio):
        """
        log P(ff,tzm,F) = log P(ff) + \sigma [P(tzm|ff) * P(F|tzm,ff)]
        q(ff|phi) ~ Dirichlet
        q(tzm|ff) ~  

        data: Fluxes
        """
        # self.max_z = max_z
        self.bin_edge = bin_edge
        self.template = template
        self.a = a
        self.S_N_ratio = S_N_ratio
        self.data = data ### Shape (N_galaxies,b_bands)
        self.kdim = kdim

        self.p_f = None ### params for ff dirichlet (1,K dim)
        self.p_tzm = None ## params for Cat  tzm (N_galaxies , K dim) 
        self.mean = np.load('mean.npz')
        self.std = np.load('std.npz')    
        ### construct for onehot tzm to z
        
        self.len_t = len(self.bin_edge[0]) -1
        self.len_z = len(self.bin_edge[1]) -1
        self.len_m = len(self.bin_edge[2]) -1
        self.vec_tzm = np.ones((len_t,len_z,len_m))
        self.vec_tzm *=  (self.bin_edge[1] - self.bin_edge[1][1]/2)[1:][np.newaxis,:,np.newaxis]
        self.vec_tzm = np.flatten(self.vec_tzm)
       

        pass
    


    def get_elbo(self):
        """
        Model:
            p(ff) ~ Dirichlet (1,....) ->  p_ff =  constant
            p(tzm_g|ff) ~ Cat(ff)  -> 
            p(F_g|tzm_g) ~ N(m(tzm),s(tzm))
        
        Variational:
            q(ff|prams) ~ Dirichlet(prams) 
            q(tzm_g|params) ~ Cat(params)

        ELBO = Eq(log p(tzm,ff,F)) - Eq(log q(tzm,ff))
        logp(tzm,ff,F) = log p(ff) + \sum logp(F_g|tzm_g) + logp(tzm_g|ff)
        
        Eq(logp(ff_i)) = constant (alpha =1)
        ####
        Eq(logp(tzm_g|ff)) = Eq(logp())

        Eq(logp(F_g|tzm_g)) = Eq(\sum_b -(F_i - mu_i)**2/sigma_i2 - log sigma_i)
        
        With meanfield assumption: q(tzm,ff) = q(ff)*\prod q(tzm_g)
        logq(tzm,ff) = logq(ff) + \sum logq(t_g) + \sum logq(z_g)
        ff ~ dirichlet
        """
        
        # logpff = np.sum([digamma(alpha) - digamma(np.sum(self.alpha)) for alpha in self.alpha] ) ## logpff = const with prior ~ (1,...)

        ### tzm
        logp_tzm = self.p_tzm *  (digamma(self.p_f) - digamma(np.sum(self.p_f,keepdims=True,axis=1))) ### p_tzm_g = Cat(ff) 
        
        logpF =  list(map(lambda x:  self.likelihood_F(x[0],flux[1]), zip(self.p_tzm,self.data)  ## loglikelihood  data shape (N,b)
        
        ### entropy
        entropy_ff =    - np.log(np.sum(self.p_f))  + np.sum(np.log(gamma(self.p_f)))   \
                        - np.sum ((self.p_f -1.) * (digamma(self.p_f) \
                            - digamma(np.sum(self.p_f,keepdims=True,axis=1))))

        entropy_tzm = - np.sum(np.log(self.p_tzm) * self.p_tzm) 
        
        return np.sum(logp_tzm) + np.sum(logpF) + entropy_ff + entropy_tzm
    
    def likelihood_F(self,tzm,flux):
        """
        tzm: one hot vector dim (k,)
        flux: (5,) shape
        """
        # t,z,m = sample_tzm(index,self.bin_edge)
        # if z > self.max_z:
        #     # print('out range')
        #     return -np.inf
        # frac_index = index 
        # z = np.sum(tzm * self.vec_tzm)
        # obs_mean = (self.mean['coef'] * z).squeeze() + self.mean['intercept']
        # obs_std = (self.std['coef'] * z).squeeze() + self.std['intercept']
        
        # # print(obs_mean)
        # # obs_std = obs_mean/self.S_N_ratio
        obs_mean = self.a * mean_flux(tzm,self.template) ## shape (bands,1)
        obs_std = np.asarray([b/self.S_N_ratio  
                                if b > 0.0 else 1000.0
                                    for b in obs_mean])
        
        ### likeihood
        liklhood = - 1/2 *np.sum(((flux - obs_mean)/obs_std)**2) - np.sum(np.log(obs_std))

        return liklhood
    # def likelihood_F(self,tzm,flux):
    #     """
    #     We have mean and 1/std from paramater of linear fitting (linear function of z)
    #     because of tzm_g ~ Multinomial (1) -> z ~ Multinomial(1)
    #     This likelihood is sum over b_bands
    #     tzm: one hot vector dim (k,)
    #     flux: (5,) shape
    #     here we just consider z
    #     """
    #     p_z = np.reshape(tzm,self.len_t,self.len_z,self.len_m)
    #     p_z = np.sum(p_z,axis=(0,2)) ## shape (len_z,)
    #     a = self.mean['coef'] ### b_band shape
    #     b = self.mean['intercept']### b_band shape
    #     c = self.std['coef']
    #     d = self.std['intercept']
        
    #     return liklhood

    def update_ff_params(self):
        self.p_f = 
        pass

    def update_tzm_params(self):
        self.p_tzm = digamma(self.p_f) - digamma(np.sum(self.p_f,keepdims=True,axis=1))

        pass

    def coordinate_ascent(self,threshold):
        """
        first update tzm params
        second update f params
        """
        initial_ff = 0.0

        pass






    








