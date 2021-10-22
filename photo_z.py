from models.gibbs_sampling import ConditionalDistribution,GibbsSampling
import numpy as np
import scipy.stats as stats
from models.metropolis_hasting import MetropolisHasting,MHSampling
from utils.util_photometric import read_template,read_flux,mean_flux,find_tzm_index,read_file_tzm,sample_tzm
from utils.visualize import Visualizer
import pdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor   
from utils.utils import DataGenerator
# from numba import jit,jit_module
import cupy as cp
import ray
import psutil
from models.ABC import ABCSampling
from tqdm import tqdm
import scipy.stats as stats
import os
import sys
import glob
import copy
import random
from dtaidistance import dtw_ndim
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression


# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus)
## parallel
# from concurrent.futures import ThreadPoolExecutor
# e = ThreadPoolExecutor(6)
## utils func 

## config for spark
# spark_home = '/opt/spark'
# if 'SPARK_HOME' not in os.environ:
#     os.environ['SPARK_HOME'] = spark_home

# SPARK_HOME = os.environ['SPARK_HOME']

# sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
# for lib in glob.glob(os.path.join(SPARK_HOME, "python", "lib", "*.zip")):
#     sys.path.insert(0,lib)

from pyspark import SparkContext
from pyspark import SparkConf
# 
conf=SparkConf()
conf.set("spark.executor.memory", "512m")
conf.set("spark.driver.memory", "40g")
conf.set("spark.cores.max", "10")
conf.set("spark.driver.maxResultSize","0")
conf.set("spark.sql.shuffle.partitions","5")

sc = SparkContext.getOrCreate(conf)



def prior_sample(n_samples,ff,__bin_edge):

    return sc.parallelize(np.random.multinomial(1,ff,size=n_samples)).map(lambda x: tzm_transform(x,__bin_edge))
    # return list(map(lambda x: tzm_transform(x,__bin_edge),np.random.multinomial(1,ff,size=n_samples)))

def distance_instance(instance,data):
    instance = np.array(instance)[np.newaxis,:]
    dist = np.mean((instance - data)**2,axis=1)
    return dist 
    ### change to just for one tzm tuple
def tzm_transform(multinomial,__bin_edge):
    """
    Multinomial is one hot vector (f_shape)
    return (tzm)
    """
    frac_index = np.argmax(multinomial)
    tzm_arr = sample_tzm(frac_index,__bin_edge)

    return np.array(tzm_arr)

     ### change to just for one tzm tuple
def generate_replica_plus(tzm_arr,__a,__template,__S_N_ratio):
    """
    This func helps to generate one flux of one galaxy
    """
    obs_mean = __a * mean_flux(tzm_arr,__template)
    obs_std = [i/__S_N_ratio if i > 0.0 else 1000.0 for i in obs_mean]

    result = np.array([np.random.normal(mean,std) for mean,std in zip(obs_mean,obs_std)])
    return result

class TZMABC(ABCSampling):
    __template = None
    __bin_edge = None
    __S_N_ratio = None
    __a = None
    # __data = None

    def __init__(self,data,template,bin_edge,a,S_N_ratio,threshold,true_tzm):
        super().__init__(data,threshold)
        if TZMABC.__template is None:
            TZMABC.__template = template 
        if TZMABC.__bin_edge is None:
            TZMABC.__bin_edge = bin_edge 
        if TZMABC.__S_N_ratio is None:
            TZMABC.__S_N_ratio = S_N_ratio 
        if TZMABC.__a is None:
            TZMABC.__a = a
        
        self.threshold = threshold
        self.true_tzm = true_tzm
        # self.regressor = 

        pass
    def set_data(self,data):
        self.data = data

    def set_prior(self,ff):

        self.prior = type('Prior', (object,), {'sample' : lambda n_samples: prior_sample(n_samples,ff,TZMABC.__bin_edge)})

    # @staticmethod
    # def prior_sample(n_samples,ff):
    #     return sc.parallelize(np.random.multinomial(1,ff,size=n_samples)).map(TZMABC.tzm_transform)


    # ### change to just for one tzm tuple
    # @staticmethod
    # def tzm_transform(multinomial):
    #     """
    #     Multinomial is one hot vector (f_shape)
    #     return (tzm)
    #     """
    #     frac_index = np.argmax(multinomial)
    #     # tzm_arr = list(map(lambda index : sample_tzm(index,self.bin_edge),frac_index))
    #     tzm_arr = sample_tzm(frac_index,TZMABC.__bin_edge)

    #     # print(f"Length tzm:{len(self.true_tzm)}, Length samples: {len(tzm_arr)}")
    #     # print(f"Shape tzm:{len(self.true_tzm[1])}, Shape samples: {len(tzm_arr[1])}")
    #     # tzm_arr = [[i[0][0],i[1][1],i[0][2]] for i in zip(self.true_tzm,tzm_arr)] ## t,m is real value
    #     return np.array(tzm_arr)

    #  ### change to just for one tzm tuple
    @staticmethod
    def generate_replica(tzm_arr):
        """
        This func helps to generate one flux of one galaxy
        """

        # frac_index = np.argmax(theta,axis=1)
        # tzm_arr = list(map(lambda index : sample_tzm(index,self.bin_edge),frac_index))
        # obs_mean = list(map(lambda x: self.a * mean_flux(x,self.template),tzm_arr))
        obs_mean = TZMABC.__a * mean_flux(tzm_arr,TZMABC.__template)
        # print(f'Sample: {tzm_arr[0]}')
        # print(f"Mean: {obs_mean[0]}")
        # obs_std = list(map(lambda x: [i/self.S_N_ratio if i > 0.0 else 1000.0 for i in x],obs_mean))
        obs_std = [i/TZMABC.__S_N_ratio if i > 0.0 else 1000.0 for i in obs_mean]

        # print(f"std: {obs_std[0]}")
        # result = np.array(list(map(lambda x: [np.random.normal(mean,std) for mean,std in zip(*x)], zip(obs_mean,obs_std))))
        result = np.array([np.random.normal(mean,std) for mean,std in zip(obs_mean,obs_std)])

        # print(f"Result{result[0]}")
        return result

    def statistics(self,x):
        """
        this function calculate statstistic for data
        """
        stats = np.mean(x)
        return stats

    def distance(self,data,replica):
        """
        Try with L2 distance
        """
        assert data.shape == replica.shape, print(f'Real data and replica data do not have the same shape: {data.shape} != {replica.shape}')
        dim = data.shape[-1]
        # weight = np.random.multivariate_normal(mean=np.random.normal(size=dim),cov = np.eye(dim),size=2)
        # weight = weight/np.sqrt(np.sum(weight**2,axis=1,keepdims=True))

        ## to preserve orders, we can try with different subsets
        # random_index = random.sample([i for i in range(len(data))],k=500)
        # data = np.matmul(data,weight.T)
        # replica = np.matmul(replica,weight.T)

        # result = [np.mean((data[:,i] - replica[:,i])**2) for i in range(2)]
        # result = dtw_ndim.distance(data, replica)
        result = (self.statistics(data) - self.statistics(replica))**2
        # return np.mean((data - replica)**2)
        return result

        
    
    def sample(self,n_samples):
        iter_ = 0
        accept = 0
        critic = 1
        # list_proposal = []
        a = TZMABC.__a
        template = TZMABC.__template
        SN_ratio = TZMABC.__S_N_ratio
        data = self.data
        while accept<critic:
            iter_+=1
            proposal = self.prior.sample(n_samples=n_samples*100) ## at the moment is list ## RDD obj
            # proposal = sc.parallelize(proposal)

            # print(proposal[0])
            # proposal =  sc.parallelize(proposal) ##list(zip(flux,tzm))
        # print(f'Data count: {data.count()}')
            replica = proposal.map(lambda x: generate_replica_plus(x,a,template,SN_ratio))
            # replica = np.array(replica.collect())
            # replica = self.generate_replica(proposal)
            # print(f"Shape of proposal : {proposal.shape}")
            # replica = np.array(list(map(TZMABC.generate_replica,proposal)))
            # print(f"Shape of replica : {replica.shape}")
            dist = replica.map(lambda x: distance_instance(x,data))
            # temp_replica = replica[:,np.newaxis,:]
            # temp_data = self.data[np.newaxis,:,:]
            # dist = np.mean((temp_replica - temp_data)**2,axis=2)
            # print(np.array(dist.collect()).shape)
            min_index = np.argmin(np.array(dist.collect()),axis=0)
            # print(proposal)
            # print(replica)
            ### check proposal valid or not
            replica  = np.array(replica.collect())[min_index]
            print(self.distance(self.data,replica))
            if self.distance(self.data,replica) < self.threshold:
                # print(f"Range z: {np.min(proposal[:,1])} - {np.max(proposal[:,1])}")
                accept +=1
                # list_proposal.append(proposal[min_index])

        print(f"Accepted rate: {accept/iter_:.2f}")

        # return np.mean(list_proposal,axis=0)
        return np.array(proposal.collect())[min_index]
    
              
    
         
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
        # t_new = np.random.uniform()
        t_new = np.random.binomial(1, p=0.5)
        ### WHAT!!!!! should not use random function of scipy
        # z_new = stats.uniform(loc=tzm[1]/(1 + self.step_size),scale= tzm[1] *(1 +self.step_size)-tzm[1]/(1 + self.step_size)).rvs()
        z_new = np.random.uniform(low =tzm[1]/(1 + self.step_size), 
                                       high=tzm[1] *(1 +self.step_size))
        m_new = 0.0
        m_new = np.log10(mean_flux((t_new,z_new,m_new),self.template)[-1])

        return [t_new,z_new,m_new]

def prob_func(flux,fraction,max_z,bin_edge,a,template,S_N_ratio):
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
        if z > max_z:
            # print('out range')
            return -np.inf
        frac_index = find_tzm_index(tzm, bin_edge)
        obs_mean = a * mean_flux(tzm,template)
        # print(obs_mean)
        # obs_std = obs_mean/self.S_N_ratio
        obs_std = np.asarray([b/S_N_ratio  
                                if b > 0.0 else 1000.0
                                    for b in obs_mean])
        # print(frac_index)
        # print(np.sum(np.log(obs_std)))
        return np.log(fraction[frac_index]) \
                - np.sum(np.log(obs_std)) \
                - 1/2 *np.sum(((flux - obs_mean)/obs_std)**2) 
    
    return prob_post 

    
def vectorize_func(prob_func,fraction,template,scale):
    generator = TZMMetropolis(None,template,scale=scale)
    def vectorize_func_(var):
        ## change var just tzm
        flux,tzm = var
        func = prob_func(flux)
        generator.set_target(func)
        return generator(tzm,n_samples=1,progress_bar=False)[-1]
    return vectorize_func_

def apply_func(fraction,max_z,bin_edge,a,template,S_N_ratio,scale):
    prob_func_ = lambda x:prob_func(x,fraction,max_z,bin_edge,a,template,S_N_ratio)
    return vectorize_func(prob_func_,fraction,template,scale)

class ConditionalTZM(ConditionalDistribution):
    def __init__(self,a,bin_edge,S_N_ratio,template,flux,max_z,scale,true_tzm):
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
        # self.generator = TZMMetropolis(None,self.template,scale=self.scale)
        # self.sc = sc
        self.generator = TZMABC(np.array([i for i in self.flux]),self.template,self.bin_edge,self.a,self.S_N_ratio,2e8,true_tzm)
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
            obs_std = np.asarray([b/self.S_N_ratio  
                                   if b > 0.0 else 1000.0
                                       for b in obs_mean])
            # print(frac_index)
            # print(np.sum(np.log(obs_std)))
            return np.log(fraction[frac_index]) \
                    - np.sum(np.log(obs_std)) \
                    - 1/2 *np.sum(((flux - obs_mean)/obs_std)**2) 
        
        return prob_post 

    
    # @staticmethod
    # def vectorize_func(prob_func,fraction,template,scale):
    #     generator = TZMMetropolis(None,template,scale=scale)
    #     def vectorize_func_(var):
    #         flux,tzm = var
    #         func = prob_func(flux)
    #         generator.set_target(func)
    #         return generator(tzm,n_samples=1,progress_bar=False)[-1]
    #     return vectorize_func_

    # @staticmethod
    # def apply_func(fraction,max_z,bin_edge,a,template,S_N_ratio,scale):
    #     prob_func_ = lambda x: ConditionalTZM.prob_func(x,fraction,max_z,bin_edge,a,template,S_N_ratio)
    #     return vectorize_func(prob_func_,fraction,template,scale)

        
    def set_params(self,**kwargs):
        # self.__dict__.update({'flux':kwargs['flux']})
        self.__dict__.update(kwargs)
        # print(f"Inside ConditionalTZM {len(self.fraction)}")
        ### loop for all galaxies
        ## flux has shape (N_g,N_b), this is the observed data
        # self.samples = []
        # ### need to parallel
        # # self.samples = self.process_multi(list(range(len(self.flux))))
        
        # listoflist = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        
        # lists = listoflist(list(range(len(self.flux))),sz=len(self.flux)//num_cpus)
        # fluxes_ref = ray.put(self.flux)
        # template_ref = ray.put(self.template)
        # tzm_ref = ray.put(self.tzm)
        # fraction_ref = ray.put(self.fraction)
        # generator_ref = ray.put(self.generator)
        # samples_ = ray.get([multi_process.remote(fluxes_ref,template_ref,li,self.scale,self.prob_func,fraction_ref,tzm_ref,generator_ref) for li in lists])

        ## for i in samples_:
        ##     self.samples+= i

        ### code for pyspark
        
        # data =  sc.parallelize(list(zip(self.flux,self.tzm))) ##list(zip(flux,tzm))
        # print(f'Data count: {data.count()}')
        # data = data.map(apply_func(self.fraction,self.max_z,self.bin_edge,a,self.template,self.S_N_ratio,self.scale))
        # samples = data.collect()

        # samples=[]
        # for i in range(len(self.flux)):
            # func = self.prob_func(flux=self.flux[i],fraction=self.fraction)
            # self.generator =  TZMMetropolis(None,self.template,scale=self.scale)
            # self.generator.set_target(func)
            # samples.append(self.generator(self.tzm[i],n_samples=self.n_samples,progress_bar=False)[-1])
        
        ## for ABC
        self.generator.set_prior(self.fraction)
        samples = self.generator.sample(len(self.flux))
        print(samples.shape)

        ## check fraction
        # ff = np.array(self.fraction)
        # ff = ff.reshape(len(self.bin_edge[0])-1,len(self.bin_edge[1])-1,len(self.bin_edge[2])-1)

        # print(f"Z: {self.bin_edge[1][np.argmax(np.sum(ff,axis=(0,2)))]}")
        # self


        # ##vectorize
        # use_func = vectorize_func(self.flux,self.prob_func,self.fraction,self.tzm,self.generator)
        # ## 
        # self.samples = list(map(use_func,[i for i in range(len(self.flux))]))


        # # # # self.samples = generate_tzm(self.flux,self.fraction,self.template,self.tzm,self.scale,self.n_samples,self.prob_func)
        self.samples = np.array(samples)
        # self.generator.set_prior(self.fraction)
        # for i in range(len(self.flux)):
        #     self.generator.set_data(self.flux[i:i+1])
        
        # self.samples = self.generator.sample(n_samples=len(self.flux))
        # print(self.samples.shape)
        # futures = [e.submit(self.return_sample, flux, tzm) for flux,tzm in list(zip(self.flux,self.tzm))]
        # self.samples = [f.result() for f in futures]

   
    def sample(self):
        ### normalize the iter in generator
        # self.generator.iter /= len(self.flux)
        return  self.samples



    

# @ray.remote
# def multi_process(flux,template,indexes,scale,prob_func,fraction,tzm,generator):
#     samples = []
#     # generator = TZMMetropolis(None,template,scale=scale)
#     for i in indexes:
#         func = prob_func(flux=flux[i],fraction=fraction)
#         generator.set_target(func)
#         samples.append(generator(tzm[i],n_samples=1,progress_bar=False)[-1])
#     return samples
           



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
        # print(type(self.bin_t))
        # print(self.tzm.shape)
        tzm_raw_bin_counts, edges  = np.histogramdd(self.tzm,bins=(self.bin_t,self.bin_z,self.bin_m))
        ## flatten concurrence
        # tzm_bin_counts = cp.array([item for sublist_t  in tzm_raw_bin_counts 
        #                for sublist_tz in sublist_t 
        #                for item       in sublist_tz]) 
        tzm_bin_counts = tzm_raw_bin_counts.flatten()

        assert len(self.f_prior) == len(tzm_bin_counts),"Wrong shape between prior and concurrence"
        # self.generator = stats.dirichlet(self.f_prior + tzm_bin_counts)
        self.posterior = self.f_prior + tzm_bin_counts
        self.generator = np.random.dirichlet(self.posterior)
        

    def sample(self):
        " return vector shape (1,K)"
        return self.generator
        # return self.generator.rvs()[0].tolist()
        # return np.random.dirichlet(self.posterior).tolist()

class PhotometricGibbsSampling(GibbsSampling):

    def __init__(self,path_obs,list_temp_path,a,S_N_ratio,n_t,min_m,max_m,min_z,max_z,zbin,mbin,scale=1.0,num_g=500,true_tzm=None):
        # self.flux = read_flux(path_obs)
        self.template = read_template(list_temp_path)

        ## change flux to new data generator
        # pdb.set_trace()
        # self.flux = DataGenerator(path_obs,cols=[0,1,2,3,4],header=None,cuda=False)
        # self.flux =  DataGenerator(path_obs,cols=[14,19,24,29,34],cuda=False,num=500)
        self.flux =  DataGenerator(path_obs,cols=[2,3,4,5,6],header=None,cuda=False)
        
        ### Check max z
        temp_z =  DataGenerator(path_obs,cols=[1],header=None,cuda=False)
        temp_z.index = self.flux.index
        index = []
        for i,j in enumerate(temp_z):
            if j < max_z:
                index.append(i)
        index = index[:num_g]
        self.flux.index = [temp_z.index[i] for i in index]

        print(len(self.flux))
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

        true_tzm.index = self.flux.index
        true_tzm = [i for i in true_tzm]
        self.true_tzm = true_tzm
       
        # self.n_samples=n_samples
        self.set_conditional()
        pass

    def set_conditional(self):
        self.conditional = {'fraction':ConditionalFraction(self.bin_t,self.bin_z,self.bin_m),
                            'tzm':ConditionalTZM(self.a,(self.bin_t,self.bin_z,self.bin_m),self.S_N_ratio,self.template,self.flux,self.max_z,self.scale,self.true_tzm),
                            }


if __name__ == '__main__':

    # path_obs = 'obs_F.csv'
    # path_obs = '/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat'
    path_obs ='/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv'
    # path_obs ='/data/phuocbui/MCMC/reimplement/data/combine_data.csv'
    
    list_temp_path = ['/data/phuocbui/MCMC/reimplement/data/Ell2_flux_z_no_header.dat',
                        '/data/phuocbui/MCMC/reimplement/data/Ell5_flux_z_no_header.dat']
    real_data_path = '/data/phuocbui/MCMC/reimplement/data/true_tzm.csv'
    a = 1.0
    S_N_ratio = 5.0
    n_t = 2
    min_m = -2.0
    max_m = 17.0
    min_z = 0.0
    max_z = 1.3
    zbin = 50
    mbin = 10
    num_g = 5000#553019
    N_MCMC = 500
    # print(f"length of fraction: {len(init_values['fraction'][0])}")
    # print(f"length of tzm: {len(init_values['tzm'][0])}")
    # pdb.set_trace()

    # e = ThreadPoolExecutor(2)
    
    def generate_chain(i):
        init_values = {'fraction':[np.random.uniform(size=n_t *zbin * mbin)],'tzm':[np.array([[np.random.binomial(1,0.5),np.random.uniform(low=min_z,high=max_z),np.random.uniform(low=min_m,high=max_m)] for i in range(num_g)])]}

        photometric = PhotometricGibbsSampling(path_obs,list_temp_path,a,S_N_ratio,n_t,min_m,max_m,min_z,max_z,zbin,mbin,scale=0.7)

        samples = photometric(init_values,n_samples=N_MCMC,progress_bar=True)
        return samples
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     samples_ = executor.map(generate_chain, range(4))
    # futures = [e.submit(generate_chain) for i in range(5)]
    # samples_ = [f.result() for f in futures]
    # new_samples = {}
    # new_samples['fraction'] = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/MCMC_samples_f.csv',header=None).values
    # new_t = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_t.csv',header=None).values
    # new_z= pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_z.csv',header=None).values
    # new_m = pd.read_csv('/home/phuocbui/MCMC/hierarchical model/Cl_data_included_MCMC_samples_m.csv',header=None).values
    # new_samples['tzm'] = [list(zip(new_t[i],new_z[i],new_m[i])) for i in range(len(new_t))]
    # samples = {'fraction':[],'tzm':[]}
    # for sam in samples_:
    #     for k,v in sam.items():
    #         samples[k]+=v
     # real_tzm =  read_file_tzm(real_data_path)
    real_tzm = DataGenerator(path_obs,cols=[0,1,2],cuda=False,header=None)
    
    init_values = {'fraction':[np.random.uniform(size=n_t *zbin * mbin)],'tzm':[np.array([[np.random.binomial(1,0.5),np.random.uniform(low=min_z,high=max_z),np.random.uniform(low=min_m,high=max_m)] for i in range(num_g)])]}
    photometric = PhotometricGibbsSampling(path_obs,list_temp_path,a,S_N_ratio,n_t,min_m,max_m,min_z,max_z,zbin,mbin,scale=1.,num_g=num_g,true_tzm=real_tzm)
    samples = photometric(init_values,n_samples=N_MCMC,progress_bar=True)

    real_tzm.index = photometric.flux.index
    real_tzm = [i for i in real_tzm]
   
    vis = Visualizer(samples,real_tzm,photometric.bin_t,photometric.bin_z,photometric.bin_m,burn_in=0.2)
    vis.MCMC_converge('MCMC_converge')
    vis.scatter_z('z')
    vis.confusion_t('t')
    # for i,samp in enumerate(samples_):
    #     vis = Visualizer(samp,real_tzm,photometric.bin_t,photometric.bin_z,photometric.bin_m,burn_in=0.2)

    #     # vis.plot_tzm()
    #     vis.plot_ff(name=f'{i}_ff.png')
    #     ### plot 

    #     vis.MCMC_converge(f'MCMC_converge_{i}')
    #     vis.scatter_z(f'z_{i}')
    #     vis.confusion_t(f't_{i}')
