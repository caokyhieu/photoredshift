import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import pdb
from itertools import tee
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import random
import seaborn as sns
from utils.util_photometric import read_template,read_flux,read_file_tzm


def generate_plot_dirichlet(x,params,index=1,len_i=20,len_j=2,len_k=20,file_name='plot.png', num_bins=50):
    # fig, ax = plt.subplots()
    fig = plt.figure()
    alpha = np.sum(params)
    print(f'Alpha:{alpha}')
    
    
    if isinstance(index,list):
        list_index = gen_index(*index)
        data = x.reshape(-1,len_i,len_j,len_k)
        # list_index = gen_index(index)
        if index[0] is not None:
            data = np.sum(data[:,index[0],:,:],axis=(1,2))
        elif index[1] is not None:
            data = np.sum(data[:,:,index[1],:],axis=(1,2))
        else:
            data = np.sum(data[:,:,:,index[2]],axis=(1,2))

        # data = x[:,list_index]
        # data = [data[i,j] for i in range(data.shape[0]) for j in range(data.shape[1])]
        alpha_i = np.sum(params[list_index])
        

    else:
        alpha_i = params[index]
        data = x[:,index]
    # pdb.set_trace()
    ##beta distribution
    # rv = beta(alpha_i, alpha-alpha_i)
    print(f'Alpha_i:{alpha_i}')
    plt.hist(data, num_bins,density=True,histtype = 'bar', facecolor = 'blue')
    new_data = np.sort(data)
    plt.plot(new_data, beta.pdf(new_data, alpha_i, alpha-alpha_i),'k-', lw=2, alpha=0.6, label='beta pdf')
    # # add a 'best fit' line
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.ylabel("Density")
    plt.xlabel("Value")    
    # ax.plot(bins)
    # fig.tight_layout()
    fig.savefig(file_name)
    plt.close(fig)



def gen_index(i,j,k,len_i=20,len_j=2,len_k=20):

    """
    gen a list with index according i,j,k
    """
    l = len_i*len_j*len_k
    if i is not None:
        return [t for t in range(i*len_j*len_k,(i+1)*len_j*len_k,1)]
    elif j is not None:
        return [h + p for h in range(j*len_k,len_i*len_j*len_k,len_j*len_k) for p in range(len_k)]
    else:
        return [t for t in range(k,l,len_k)]

def plot_hist(data,dist,name,nbins=50):
    params = dist.fit(data)
    ci = dist(*params).interval(0.95)
    print(f'Params of this distribution: {params}')
    height, bins, patches = plt.hist(np.array(data),bins=nbins,alpha=0.3,density=True)
    plt.fill_betweenx([0, height.max()], ci[0], ci[1], color='g', alpha=0.1)
    plt.vlines(np.array(data).mean(),ymin=0, ymax=height.max(),colors='red',linestyles='dashed',label='mean')
    plt.savefig(name)
    plt.close('all')

def read_chunk_file(path,cols=[3],num=3):
    with open(path,'rb') as f:
        lines = f.readlines()
    for j in range(1,len(lines),num):
        print(j)
        yield [[i[c] for c in cols] for i in  lines[j:j+num]]


class DataGenerator:
    """
    Data Generator for learning batchsize
    """
    def __init__(self,path,cols=[3],header= True,cuda=False,num=None):
        self.path = path
        self.cols = cols
        self.token = ',' if path.endswith('.csv') else None
        with open(path,'r') as f:
            if header:
                self.features = f.readline().split(sep=self.token)
                self.columns = [self.features[1:][c] for c in self.cols]
            else:
                self.features = None 
                self.columns = None
            self.data = [g.split(sep=self.token) for g in f.readlines()]
            self.data = [[float(l[c]) for c in self.cols] for l in self.data]
            if cuda:
                self.data = cp.array(self.data)
            else:
                self.data = np.array(self.data)

        
        
        self.index = [i for i in range(0,len(self.data))]
        if num:
            self.index = self.index[:num]

        self.current = -1
        print(f" Length of data loader {len(self)}")

    def __len__(self):
        "Get length of data"
        return len(self.index)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self.index)-1:
            self.current = -1
            raise StopIteration
            
        self.current = self.current +1
        # print(self.__getitem__(self.current))
        return self.__getitem__(self.current)

    def __getitem__(self,idx):
        # d = self.data[self.index[idx]].split(sep=self.token)
        # return [float(d[c]) for c in self.cols]
        return self.data[self.index[idx]]



    
def plot_images(save_path='samples.png',size=(10,6)):

    dic = read_template(paths=['Ell2_flux_z_no_header.dat','Ell5_flux_z_no_header.dat'])
    z_ = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv',header=False,cols=[1],num=10000)
    # c = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/clean_data.csv',header=False,cols=[14,19,24,29,34],num=10000)
    c = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv',header=False,cols=[2,3,4,5,6],num=10000)

    c.index = z_.index
    idxs = [i for i in range(10000)]
    z = dic['z'][0]
    template= (dic['F_bands'][0] + dic['F_bands'][1])/2
    
    # pdb.set_trace()
    data = [c.__getitem__(i) for i in idxs]
    data = np.array(data) 
    
    
    

    true_data = read_flux( 'obs_F.csv')
    true_tzm = read_file_tzm('true_tzm.csv')
    # data = data - np.min(data,axis=0,keepdims=True)
    # data = data * np.max(true_data[true_tzm[:,0]==1.],axis=0,keepdims=True)/np.max(data,axis=0,keepdims=True) - np.min(true_data[true_tzm[:,0]==1.],axis=0,keepdims=True)

    fig,ax = plt.subplots(1,3,figsize=size)
    for i in range(len(data[0])):
        d = [dat[i] for dat in data]
        ax[0].plot([i] * len(d),d)
    # assert len(template) == len(z)
    # for i in range(len(template[0])):
    #     d = [dat[i] for dat in template]
    #     x = [i]*len(d)
    #     ax[1].plot(x,d)
    #     for j,text in enumerate(z):
    #         ax[1].annotate(str(text), (x[j], d[j]))
    
    # for i in range(len(data)):
    #     d = data[i]
    #     ax[0].plot(range(5),d)

    for i in range(len(template)):
        d = template[i]
        x = range(5)
        ax[1].plot(x,d)
        # for j,text in enumerate(z):
        ax[1].annotate(str(z[i]), (x[-1], d[-1]))
    
    for i in range(len(true_data[0])):
        d = [dat[i] for dat in true_data]
        ax[2].plot([i] * len(d),d)
    
    
    # for i, txt in enumerate(z):
    fig.savefig(save_path)
    ### find index of ten thousands
    

    fig,ax = plt.subplots(1,10,figsize=(20,6))
    t = random.sample(range(len(template)),k=10)
    for j,ind in enumerate(t):
        d = template[ind]
        ax[j].plot(range(5),d)
        ax[j].set_title(f'z = {z[ind]}')
    fig.tight_layout(pad=3.0)
    fig.savefig('seperate_z.png')

    ### plot two line bound 
    fig,ax = plt.subplots(figsize=size)
    g = np.random.randint(low=0,high=len(z_))
    # g = 10000
    spec_z = z_[g]
    z_idx_1 = np.abs(np.array(z) - spec_z).argmin()
    # pdb.set_trace()
    print(z_idx_1)
    if z[z_idx_1] < spec_z:
        z_idx_2 = z_idx_1 + 1 
    else:
        z_idx_2 = z_idx_1-1
    ax.plot(range(5),data[g])
    ax.annotate(str(z_[g]), (4,data[g][-1]))
   
    ax.plot(range(5),template[z_idx_1],'--b')
    ax.annotate(str(z[z_idx_1]), (4, template[z_idx_1][-1]))
    ax.plot(range(5),template[z_idx_2],'--b')
    ax.annotate(str(z[z_idx_2]), (4, template[z_idx_2][-1]))

    if z_idx_1 < z_idx_2:
        z_idx_1 -=1
        z_idx_2 +=1
    else:
        z_idx_1 +=1
        z_idx_2 -=1
    ax.plot(range(5),template[z_idx_1],'--r')
    ax.annotate(str(z[z_idx_1]), (4, template[z_idx_1][-1]))
    ax.plot(range(5),template[z_idx_2],'--r')
    ax.annotate(str(z[z_idx_2]), (4, template[z_idx_2][-1]))

    if z_idx_1 < z_idx_2:
        z_idx_1 -=1
        z_idx_2 +=1
    else:
        z_idx_1 +=1
        z_idx_2 -=1
    ax.plot(range(5),template[z_idx_1],'--g')
    ax.annotate(str(z[z_idx_1]), (4, template[z_idx_1][-1]))
    ax.plot(range(5),template[z_idx_2],'--g')
    ax.annotate(str(z[z_idx_2]), (4, template[z_idx_2][-1]))
    
    fig.savefig('specific_z.png')


    ### plot new image with two type templates
    fig,ax = plt.subplots(figsize=size)
    z_idx_1 = np.mean((dic['F_bands'][0] - data[g])**2,axis=-1).argmin()
    # z_idx_1 = int(z_idx_1//5)
    # h = np.argmin(data[g])
    # z_idx_1 = z_idx_1[h]

    if z[z_idx_1] < spec_z:
        z_idx_2 = z_idx_1 + 1 
    else:
        z_idx_2 = z_idx_1 - 1
    
    ax.plot(range(5),data[g])
    ax.annotate(str(z_[g]), (4, data[g][-1]))
    ax.plot(range(5),dic['F_bands'][0][z_idx_1],'--g')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][0][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][0][z_idx_2],'--g',label='Type 0')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][0][z_idx_2][-1]))


    z_idx_1 = np.mean((dic['F_bands'][1] - data[g])**2,axis=-1).argmin()
    # z_idx_1 = int(z_idx_1//5)
    # h = np.argmin(data[g])
    # z_idx_1 = z_idx_1[h]

    if z[z_idx_1] < spec_z:
        z_idx_2 = z_idx_1 + 1 
    else:
        z_idx_2 = z_idx_1 - 1
    
    # ax.plot(range(5),data[g])
    # ax.annotate(str(z_[g]), (4, data[g][-1]))
    ax.plot(range(5),dic['F_bands'][1][z_idx_1],'--r')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][1][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][1][z_idx_2],'--r',label='Type 1')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][1][z_idx_2][-1]))

    ax.legend()
    fig.savefig('specific_z_new.png')

    ### plot new image with two type templates by z
    fig,ax = plt.subplots(figsize=size)
    z_idx_1 = np.abs(np.array(z) - spec_z).argmin()
    # z_idx_1 = int(z_idx_1//5)
    # h = np.argmin(data[g])
    # z_idx_1 = z_idx_1[h]

    if z[z_idx_1] < spec_z:
        z_idx_2 = z_idx_1 + 1 
    else:
        z_idx_2 = z_idx_1 - 1
    
    ax.plot(range(5),data[g])
    ax.annotate(str(z_[g]), (4, data[g][-1]))
    ax.plot(range(5),dic['F_bands'][0][z_idx_1],'--g')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][0][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][0][z_idx_2],'--g',label='Type 0')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][0][z_idx_2][-1]))


    
    # ax.plot(range(5),data[g])
    # ax.annotate(str(z_[g]), (4, data[g][-1]))
    ax.plot(range(5),dic['F_bands'][1][z_idx_1],'--r')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][1][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][1][z_idx_2],'--r',label='Type 1')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][1][z_idx_2][-1]))

    ax.legend()
    fig.savefig('specific_z_new_1.png')


    ### plot new image with two type templates by z

    # true_data = read_flux( 'obs_F.csv')
    # true_tzm = read_file_tzm('true_tzm.csv')
    g = np.random.randint(low=0,high=len(true_tzm))
    specific_t = true_tzm[g][0]
    specific_z = true_tzm[g][1]
    fig,ax = plt.subplots(figsize=size)
    z_idx_1 = np.abs(np.array(z) - specific_z).argmin()
    # z_idx_1 = int(z_idx_1//5)
    # h = np.argmin(data[g])
    # z_idx_1 = z_idx_1[h]

    if z[z_idx_1] < spec_z:
        z_idx_2 = z_idx_1 + 1 
    else:
        z_idx_2 = z_idx_1 - 1
    
    ax.plot(range(5),true_data[g])
    ax.annotate( f'{specific_z:.2f} type {specific_t:.1f}', (4, true_data[g][-1]))
    ax.plot(range(5),dic['F_bands'][0][z_idx_1],'--g')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][0][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][0][z_idx_2],'--g',label='Type 0')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][0][z_idx_2][-1]))


    
    
    ax.plot(range(5),dic['F_bands'][1][z_idx_1],'--r')
    ax.annotate(str(z[z_idx_1]), (4, dic['F_bands'][1][z_idx_1][-1]))

    ax.plot(range(5),dic['F_bands'][1][z_idx_2],'--r',label='Type 1')
    ax.annotate(str(z[z_idx_2]), (4, dic['F_bands'][1][z_idx_2][-1]))

    ax.legend()
    fig.savefig('specific_z_real_data.png')



    print('Saved images')
    pass


    
# plot_images(save_path='samples.png',size=(10,6))    


def combine_data(save_path):
    path_obs ='/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv'
    real_data_path = 'true_tzm.csv'
    path_real = 'obs_F.csv'
    df = pd.read_csv(path_obs,header=None)
    true_tzm = pd.read_csv(real_data_path,header=None,names=[0,1,-1])
    flux = pd.read_csv(path_real,header=None,names=[2,3,4,5,6])
    true_data = pd.concat((true_tzm,flux),axis=1).drop(columns=[-1])
    final_data = pd.concat((df,true_data),axis=0,ignore_index=True)
    final_data.to_csv(save_path,header=None,index=False)
    print(f"Saved file at:{save_path}")
    pass







    


    







