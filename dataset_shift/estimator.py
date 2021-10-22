import numpy as np
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors,DistanceMetric
import pdb
from sklearn.neighbors import KernelDensity

def estimate_beta(x:np.array,x_l:np.array,x_u:np.array,M:int=1):
    """
    x: query data
    x_l: numpy array label data
    x_u: numpy arr unlabel data
    return: beta_hat for x
    """
    n_u = len(x_u)
    n_l = len(x_l)
    n_x = len(x)
    ## need to define distance
    dist = DistanceMetric.get_metric('euclidean')
    neigh = NearestNeighbors(metric='euclidean',n_neighbors=M) 
    neigh.fit(x_l)
    distance ,idx = neigh.kneighbors(x)## return distance and indexs shape (n_x,M)
    ## M_anchor distnace
    M_distance = distance[:,M-1:] ## shape (n_x,1)
    
    udist_arr = dist.pairwise(x,x_u) ##shape (n_x,n_u)

    count = np.sum(udist_arr < M_distance,axis=1,keepdims=True)

    return n_l/(M * n_u) * count

def loss_beta(x_l_til,x_u_til,x_l_train,x_u_train,M):
    """
    This function used to evaluate beta estimator

    """
    n_l_til = len(x_l_til)
    n_u_til = len(x_u_til)
    beta_l = estimate_beta(x_l_til,x_l_train,x_u_train,M)
    beta_u = estimate_beta(x_u_til,x_l_train,x_u_train,M)

    return 1/n_l_til * np.sum(beta_l**2) - 2/n_u_til * np.sum(beta_u)

def reduce_to_onehot(z_val,bin_z,beta):
    """
    z_val: (N,)
    bin_z: (K,)
    beta: (N,1)
    """
    ## get index of z in bin_z
    idx = np.searchsorted(bin_z,z_val) - 1 ## return (N,)
    one_hot = np.eye(len(bin_z) -1 )[idx] ## shape (N,len(bin_z) - 1)

    return one_hot * beta


## estimate conditional dist
def NN_histogram(x,x_l,x_u,z_l,bin_z,N,M):
    """
    beta: estimate beta for x
    x: shape(n_x,5)
    z
    bin_z: interval of z
    N: number of NN
    return : estimate f(z|x) (size = n_x,bin_z -1)
    """
    # beta = estimate_beta(x,x_l,x_u,M) ## shape (n_x,1)
    beta = estimate_beta(x_l,x_l,x_u,M) ## shape (n_xl,1)

    n_x = len(x)

    dist = DistanceMetric.get_metric('euclidean')
    neigh = NearestNeighbors(metric='euclidean',n_neighbors=N) 
    neigh.fit(x_l)
    distance ,idx = neigh.kneighbors(x)
    samples = z_l[idx].squeeze(-1) ##(shape n_x, N)
    beta_samples = beta[idx] ##(shape n_x, N, 1)
    

    # count,edge = np.histogram(z_l, bins=bin_z)
    # count = list(map(lambda y: np.histogram(y,bins=bin_z)[0],samples))
    count = list(map(lambda y: reduce_to_onehot(y[0],bin_z,y[1]),zip(samples,beta_samples)))
    # print(samples)
    # pdb.set_trace()

    count = np.array(count) ## shape (n_x,N,len(bin_z) -1)
    prob_f = np.sum(count,axis=1) + 1e-12 ##shape (n_x,len(bin_z) -1)
    prob_f =  prob_f/np.sum(prob_f,axis=1,keepdims=True)
    return prob_f

def kernel_NN(x,x_l,x_u,z_l,bin_z,N,M,eps=1.):
    """
    beta: estimate beta for x
    x: shape(n_x,5)
    z
    z_l: (n_l,1)
    bin_z: interval of z
    N: number of NN
    return : estimate f(z|x) (size = n_x,bin_z -1)
    """
    beta = estimate_beta(x_l,x_l,x_u,M) ## shape n_x
    n_x = len(x)
    # kernel = 1.0 * RBF(eps)

    dist = DistanceMetric.get_metric('euclidean')
    neigh = NearestNeighbors(metric='euclidean',n_neighbors=N) 
    neigh.fit(x_l)
    distance ,idx = neigh.kneighbors(x)
    samples = z_l[idx] ##(shape n_x , N, 1)
    beta_samples = beta[idx].squeeze(-1)##(shape n_x , N)
    result = list(map(lambda x: KernelDensity(kernel='gaussian', bandwidth=eps).fit(x[0],sample_weight=x[1] + 1e-12), zip(samples,beta_samples)))

    # sample_mean = np.mean(samples * beta_samples,axis=1,keepdims=True)##(shape n_x ,1)

    return  result

def loss_f(x_l_til,x_u_til,x_l_train,x_u_train,z_train,bin_z,M,N):

    beta_l = estimate_beta(x_l_til,x_l_train,x_u_train,M)
    n_u = len(x_u_til)
    n_l = len(x_l_til)
    f_u = NN_histogram(x_u_til,x_l_train,x_u_train,z_train,bin_z,N,M)
    f_l = NN_histogram(x_l_til,x_l_train,x_u_train,z_train,bin_z,N,M)
    idx = np.searchsorted(bin_z,z_train) - 1 ## shape n_l

    f_l = list(map(lambda x: x[1][x[0]],zip(idx,f_l)))
    f_l = np.array(f_l)
    return 1/n_u * np.sum(f_u**2)  - 2/n_l * np.sum(f_l * beta_l)


def spectral_estimate(x,x_l,x_u,z_l,bin_z,N,M):
    
    pass

