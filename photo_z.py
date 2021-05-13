from gibbs_sampling import ConditionalDistribution,GibbsSampling
import numpy as np
import scipy.stats as stats
from metropolis_hasting import MetropolisHasting,MHSampling
from util_photometric import read_template,read_flux,mean_flux,find_tzm_index
from visualize import Visualizer
import pdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

## parallel
# from concurrent.futures import ThreadPoolExecutor
# e = ThreadPoolExecutor(6)
## utils func 

         
class TZMMetropolis(MHSampling):
    
    def __init__(self,target,template,scale=10.0):
        super().__init__(scale)

        """
        target: target log pdf function. with parameter x, return pdf of x value.
                  It has func logpdf(val) to return pdf of value and rvs to random value 
        n_sample : num_sample
        """

        self.target = target
        self.template = template

    def set_target(self,target):
        self.target = target

    def score(self,x,y):
        # print(self.target(x))
        return self.target(x) + np.log(y[1])
        # stats.uniform.logpdf(x,loc=y/(1 + self.step_size),scale =y *(1 +self.step_size)-y/(1 + self.step_size))

    def jump(self,tzm):
        """
        tzm : list t,z,m value
        return tzm_proposed
        """
        t_new = np.random.uniform()
        ### WHAT!!!!! should not use random function of scipy
        # z_new = stats.uniform(loc=tzm[1]/(1 + self.step_size),scale= tzm[1] *(1 +self.step_size)-tzm[1]/(1 + self.step_size)).rvs()
        z_new = np.random.uniform(low =tzm[1]/(1 + self.step_size), 
                                       high=tzm[1] *(1 +self.step_size))
        m_new = 0.0
        m_new = np.log10(mean_flux((t_new,z_new,m_new),self.template)[-1])

        return [t_new,z_new,m_new]

class ConditionalTZM(ConditionalDistribution):
    def __init__(self,a,bin_edge,S_N_ratio,template,flux,max_z,scale):
        """
        template: table for look up mean flux, sigma flux
                : {'z':[],'F_bands':[]} z shape (t,z_dim) F_bands : (t,z_dim,N_b)
        """
        self.a = a
        self.bin_edge = bin_edge
        self.template = template
        self.S_N_ratio = S_N_ratio
        self.n_samples = 1
        self.flux = flux
        self.max_z = max_z
        self.scale = scale
        self.generator = TZMMetropolis(None,self.template,scale=self.scale)
        pass 

    def prob_func(self,flux,fraction):
        # print(len(fraction[0]))
        """
        flux: {F_b}g for specific galaxy
        fraction: {f_ijk}
        bin_edge,a,S_N_ratio,template: other params
        return log prob
        """
        
        def prob_post(tzm):
            "Here we have to fix m value because the proposal m is not true"
            t,z,m = tzm
            if z > self.max_z:
                # print('out range')
                return -np.inf
            frac_index = find_tzm_index(tzm, self.bin_edge)
            obs_mean = self.a * mean_flux(tzm,self.template)
            # print(obs_mean)
            # obs_std = obs_mean/self.S_N_ratio
            obs_std = np.array([b/self.S_N_ratio  
                                   if b > 0.0 else 1000.0
                                       for b in obs_mean])
            # print(frac_index)
            # print(np.sum(np.log(obs_std)))
            return np.log(fraction[frac_index]) \
                    - np.sum(np.log(obs_std)) \
                    - 1/2 *np.sum(((np.array(flux) - obs_mean)/obs_std)**2) 
        
        return prob_post 

    def set_params(self,**kwargs):
        # self.__dict__.update({'flux':kwargs['flux']})
        self.__dict__.update(kwargs)
        # print(f"Inside ConditionalTZM {len(self.fraction)}")
        ### loop for all galaxies
        ## flux has shape (N_g,N_b), this is the observed data
        self.samples = []
        ### need to parallel
        for i in range(len(self.flux)):
            func = self.prob_func(flux=self.flux[i],fraction=self.fraction)
            self.generator.set_target(func)
            self.samples.append(self.generator(self.tzm[i],n_samples=self.n_samples,progress_bar=False)[-1])
        
        
        # futures = [e.submit(self.return_sample, flux, tzm) for flux,tzm in list(zip(self.flux,self.tzm))]
        # self.samples = [f.result() for f in futures]

   
            
        
    def sample(self):
        ### normalize the iter in generator
        # self.generator.iter /= len(self.flux)
        return  self.samples

class ConditionalFraction(ConditionalDistribution):
    def __init__(self,bin_t,bin_z,bin_m):
       
        self.bin_z = bin_z
        self.bin_m = bin_m
        self.bin_t = bin_t
        ##prior for f (Dirichlet distribution)
        self.f_prior = np.array([1.0 for i in range(0,(len(bin_z) -1 ) * (len(bin_m)- 1) * (len(bin_t) -1))])


    def set_params(self,**kwargs):
        """
        need provide tzm argument: ({tg,zg,mg} data have shape (n_sample,3)
        """
        self.__dict__.update(kwargs)

        ### calculate concurrence 
        tzm_raw_bin_counts, edges  = np.histogramdd(np.asanyarray(self.tzm), 
                                        bins=(self.bin_t,self.bin_z,self.bin_m), 
                                        normed=False, weights=None)
        ## flatten concurrence
        tzm_bin_counts = np.array([item for sublist_t  in tzm_raw_bin_counts 
                       for sublist_tz in sublist_t 
                       for item       in sublist_tz]) 

        assert len(self.f_prior) == len(tzm_bin_counts),"Wrong shape between prior and concurrence"
        self.generator = stats.dirichlet(self.f_prior + tzm_bin_counts)
        self.posterior = self.f_prior + tzm_bin_counts
        

    def sample(self):
        " return vector shape (1,K)"
        return self.generator.rvs()[0].tolist()
        # return np.random.dirichlet(self.posterior).tolist()

class PhotometricGibbsSampling(GibbsSampling):

    def __init__(self,path_obs,list_temp_path,a,S_N_ratio,n_t,min_m,max_m,min_z,max_z,zbin,mbin,scale=1.0):
        self.flux = read_flux(path_obs)
        self.template = read_template(list_temp_path)
        self.a = a
        self.S_N_ratio = S_N_ratio

        self.n_t = n_t 
        self.min_m = min_m 
        self.max_m = max_m
        self.min_z = min_z 
        self.max_z = max_z
        self.zbin = zbin
        self.mbin = mbin
        self.scale = scale
        # self.size = size

        ## find range of z and m
        # self.list_t,self.list_z,self.list_m = list(zip(*tzm_list))
        self.bin_t = np.linspace(0,1,num=self.n_t+1)
        self.bin_z = np.linspace(self.min_z,self.max_z,num=self.zbin+1)
        self.bin_m = np.linspace(self.min_m,self.max_m,num=self.mbin+1)
       
        # self.n_samples=n_samples
        self.set_conditional()
        pass

    def set_conditional(self):
        self.conditional = {'fraction':ConditionalFraction(self.bin_t,self.bin_z,self.bin_m),
                            'tzm':ConditionalTZM(self.a,(self.bin_t,self.bin_z,self.bin_m),self.S_N_ratio,self.template,self.flux,self.max_z,self.scale),
                            }

if __name__ == '__main__':

    path_obs = 'obs_F.csv'
    list_temp_path = ['Ell2_flux_z_no_header.dat','Ell5_flux_z_no_header.dat']
    real_data_path = 'true_tzm.csv'
    a = 1.0
    S_N_ratio = 5.0
    n_t = 2
    min_m = -2.0
    max_m = 17.0
    min_z = 0.0
    max_z = 1.30
    zbin = 50
    mbin = 10
    num_g = 500
    N_MCMC = 50000
    init_values = {'fraction':[[0.1]* (n_t *zbin * mbin) ],'tzm':[[[0.1,0.3,.3] for i in range(num_g)]]}
    # print(f"length of fraction: {len(init_values['fraction'][0])}")
    # print(f"length of tzm: {len(init_values['tzm'][0])}")
    photometric = PhotometricGibbsSampling(path_obs,list_temp_path,a,S_N_ratio,n_t,min_m,max_m,min_z,max_z,zbin,mbin,scale=1.)
    # pdb.set_trace()

    # e = ThreadPoolExecutor(6)
    # futures = [e.submit(photometric,init_values,n_samples=N_MCMC,progress_bar=True) for i in range(10)]
    # samples_ = [f.result() for f in futures]
    samples = photometric(init_values,n_samples=N_MCMC,progress_bar=True)
    new_samples = {}
    new_samples['fraction'] = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/MCMC_samples_f.csv',header=None).values
    new_t = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_t.csv',header=None).values
    new_z= pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_z.csv',header=None).values
    new_m = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_m.csv',header=None).values
    new_samples['tzm'] = [list(zip(new_t[i],new_z[i],new_m[i])) for i in range(len(new_t))]
    vis = Visualizer(samples,real_data_path,photometric.bin_t,photometric.bin_z,photometric.bin_m)

    # vis.plot_tzm()
    vis.plot_ff(burn_in=0.2)
    # print(samples['fraction'])