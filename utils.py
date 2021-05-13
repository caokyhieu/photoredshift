import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import pdb


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


    







