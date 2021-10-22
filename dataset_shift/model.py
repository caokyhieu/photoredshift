from sklearn.neighbors import KNeighborsClassifier
from utils.util_photometric import read_template,read_flux,mean_flux,find_tzm_index,read_file_tzm,sample_tzm
from utils.visualize import Visualizer
from utils.utils import DataGenerator
from tqdm import tqdm,trange
import numpy as np

from estimator import estimate_beta,loss_beta,NN_histogram,loss_f,kernel_NN
import pdb

min_z = 0.
max_z = 1.3
min_m = -2.0
max_m = 17.
n_t = 2
zbin = 50
mbin = 10

f_size = n_t * zbin * mbin

bin_t = np.linspace(0,1,num=n_t+1)
bin_z = np.linspace(min_z,max_z,num=zbin+1)
bin_m = np.linspace(min_m,max_m,num=mbin+1)
bin_edge = (bin_t,bin_z,bin_m)
a = 1.

S_N_ratio = 5.0
lr = 1e-4
num_iters = 30000
num_g = 234098
batchsize =  40000
N_samples = 5000
wait_step= 100
N = 5
M = 5

path_obs        = '/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv'    
list_temp_path  =   ['/data/phuocbui/MCMC/reimplement/data/Ell2_flux_z_no_header.dat',
                    '/data/phuocbui/MCMC/reimplement/data/Ell5_flux_z_no_header.dat']
template = read_template(list_temp_path)
data = DataGenerator(path_obs,cols=[2,3,4,5,6],header=None,cuda=False)
temp_z =  DataGenerator(path_obs,cols=[1],header=None,cuda=False)


### code for sampling bias
label_data_index = [i for i,j in enumerate(temp_z.data) if j[0] < 0.6][:5000] + \
                    [i for i,j in enumerate(temp_z.data) if (j[0] > 0.6 and j[0] < 1.)][50000:50500]

unlabel_data_index = [i for i,j in enumerate(temp_z.data) if (j[0] > 0.6 and j[0] < 1.)][:50000] +\
                     [i for i,j in enumerate(temp_z.data) if j[0] < 0.6][5000:10000]
label_z = temp_z.data[label_data_index]
label_data = data.data[label_data_index]
unlabel_data = data.data[unlabel_data_index]
test_index = [i for i,j in enumerate(temp_z.data) if (i>100000 and j[0]<1)][:6000]
beta_hat = estimate_beta(unlabel_data,label_data,unlabel_data,M)

## NN_hist
prob_f = NN_histogram(data.data[test_index],label_data,unlabel_data,label_z,bin_z,N,M) ## shape(N,50)
true_idx = np.array(list(map(lambda x: np.argmax(np.random.multinomial(1, x,size=200),axis=1),prob_f))) ### (N,200)
z_pred = np.array(list(map(lambda x: bin_z[x],true_idx.flatten())))
z_pred = z_pred.reshape(len(prob_f),200,1)
# z_pred = np.expand_dims(z_pred,axis=1)

## NN_kernel 

# sample_func = kernel_NN(data.data[test_index],label_data,unlabel_data,label_z,bin_z,N,M,eps=0.5)
# pdb.set_trace()

# z_pred = np.array([func.sample(n_samples=400) for func in sample_func])

z_pred = np.swapaxes(z_pred,0,1)
t = np.ones_like(z_pred)
m = np.zeros_like(z_pred)
samples = np.concatenate((t,z_pred,m),axis=2)
samples = np.tile(samples,(2,1,1))
real_tzm = np.concatenate((t[0,:,:],temp_z.data[test_index],m[0,:,:]),axis=1)
samples = {'fraction':[np.random.uniform(size=n_t *zbin * mbin)]*100,'tzm':samples}
vis = Visualizer(samples,real_tzm,bin_t,bin_z,bin_m,burn_in=0.0)

vis.scatter_z('new_z_v2')



### code for 






