import numpy as np
import utils
import pdb
#### implement model for noiseless cases

def posterior_dirichlet(data_path,zbin=20,mbin=20,size=800):
    """
    data_path: file path of true tzm,
    zbin: number of equal bin of redshift range
    mbin : number of equal bin of magnitude 
    size: number of random samples of posterior distribution

    """
    tzm_list = utils.read_file_tzm(data_path)
    
    ## find range of z and m
    list_t,list_z,list_m = list(zip(*tzm_list))
    range_z = (np.min(list_z),np.max(list_z))
    range_m = (np.min(list_m),np.max(list_m))
    
    n_t = len(set(list_t))
    bin_z = np.linspace(*range_z,num=zbin+1)
    bin_m = np.linspace(*range_m,num=mbin+1)
    bin_t = np.linspace(0,1,num=n_t+1)

    ##prior for f (Dirichlet distribution)
    f_prior = np.array([1.0 for i in range(0,n_t*zbin*mbin)])
    ## count n_ijk
    # print(tzm_list.shape)
    tzm_raw_bin_counts, edges  = np.histogramdd(np.array(tzm_list), 
                                        bins=(bin_t,bin_z,bin_m), 
                                        normed=False, weights=None)
    # tzm_bin_counts is a depth 3 list of lists, with indexing over m as the 
    # innermost, then z, then t. When this is unflattened it should index in the
    # same order that I've done ff_prior_parameters, making for easy use of the 
    # Dirichlet sampling function (which requires a flat list of parameters)
    tzm_bin_counts = np.array([item for sublist_t  in tzm_raw_bin_counts 
                            for sublist_tz in sublist_t 
                            for item       in sublist_tz]) 
    ### params of f_posterior
    assert len(tzm_bin_counts) == len(f_prior),'Length of prior and concurrences is different '

    post_params = f_prior +  tzm_bin_counts
    # return post_params
    ## return random sample
    return np.random.dirichlet(post_params, size=size),post_params


