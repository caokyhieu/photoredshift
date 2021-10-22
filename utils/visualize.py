import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils.util_photometric import read_file_tzm
import pdb
from sklearn.metrics import confusion_matrix
from scipy import stats
import random
class Visualizer:
    def __init__(self,samples,data,bin_t,bin_z,bin_m,burn_in=0.2):
        """
        data: a dict compose 'tzm' and 'fraction' data
        tzm: will be a list compose tuple (t,z,m)
        fraction: a list of fractional ff . have size (N_t * N_z * M_m)
        bin_t,bin_z,bin_m: range of bin z, bin t bin m respectively
        burn_in: specify the burn in steps of N chain MCMC
        """

        # assert num_t * num_z * num_m == len(samples['fraction'][0]), 'Check again the tzm bins'
        l = int(burn_in * len(samples))
        self.data = samples
        self.real_data = np.array(data)
        
        self.bin_t,self.bin_z,self.bin_m = bin_t,bin_z,bin_m
        self.data['fraction'] = np.array(self.data['fraction'][l:]).reshape(-1,len(self.bin_t) -1 ,len(self.bin_z) -1 ,len(self.bin_m) -1 )
        self.data['tzm'] = np.array(self.data['tzm'][l:])

        print(f"Shape tzm : {self.data['tzm'].shape}, Shape fraction: {self.data['fraction'].shape}")

    def plot_tzm(self,name='tzm.png',num_z=10):
        """
        Function helps to plot tzm distribution from the last samples of MCMC
        """
        print(f"Numver of samples: {len(self.data['tzm'])}")
        df = pd.DataFrame(data=self.data['tzm'][-1],columns=['t','z','m'])
        df['t'] = (df['t'].values > 0.5).astype(int)
        real_df = pd.DataFrame(data=np.array(self.real_data),columns=['t','z','m'])
        real_df['t'] = (real_df['t'].values > 0.5).astype(int)
        # print(f"Max z: {real_df['z'].max()}, Max m: {real_df['m'].max()}")
        ## seperate z into bins
        bin_z = np.linspace(self.bin_z[0],self.bin_z[-1],num_z + 1)
        ## bin for sample and real data
        df['z'] = pd.cut(df['z'],bin_z,labels=[round(elem, 4) for elem in (bin_z[1:] - bin_z[:-1])/2 + bin_z[:-1]] )
        real_df['z'] = pd.cut(real_df['z'],bin_z,labels=[round(elem, 4) for elem in (bin_z[1:] - bin_z[:-1])/2 + bin_z[:-1]] )
        # real_df['z'] = pd.cut(real_df['z'],bin_z,labels=[round(elem, 4) for elem in (bin_z[1:] - bin_z[:-1])/2 + bin_z[:-1]] )
        fig,ax=plt.subplots(ncols=1,nrows=2,sharex=True,figsize=(20,10))
        # pdb.set_trace()
        ax[0].set_title('True tzm')
        sns.violinplot(data=real_df, x="z", y="m", hue="t",palette="muted",ax=ax[0])
        ax[1].set_title('Sample tzm')
        sns.violinplot(x="z", y="m", hue="t",data=df, palette="muted",ax=ax[1])
        
        fig.savefig(name, dpi=1200, format='png', bbox_inches='tight')
        # plt.close()

    def plot_ff(self,name='ff.png',num_z=10):
        """
        Function helps to plot tzm distribution from the last samples of MCMC
        fraction data : shape (n_samples,t,z,m)
        """
        # sample_ff = posterior.reshape(len(self.bin_t) -1 ,len(self.bin_z) -1 ,len(self.bin_m) -1 ) ## shape(t,z,m)
        # sample_ffz = pd.DataFrame(data=sample_ff.reshape(sample_ff.shape[0]* sample_ff.shape[1],sample_ff.shape[2]),\
        #                 index=pd.MultiIndex.from_product([[0, 1], [ i for i in (self.bin_z[1:] - self.bin_z[:-1])/2 + self.bin_z[:-1]]]),\
        #                     columns=(self.bin_m[1:] - self.bin_m[:-1])/2 + self.bin_m[:-1] )

        
        sample_ff = self.data['fraction'].sum(axis=-1) ## shape (N,t,z)
        # l = int(len(sample_ff) * burn_in)
        # l_ = range(l,len(sample_ff))
        # sample_ff = sample_ff[l_]
        sample_ffz = pd.DataFrame(data=sample_ff.reshape(sample_ff.shape[0]* sample_ff.shape[1],sample_ff.shape[2]),\
                        index=pd.MultiIndex.from_product([[i for i in range(len(sample_ff))], [0,1]]),\
                            columns=(self.bin_z[1:] - self.bin_z[:-1])/2 + self.bin_z[:-1] ) ## df shape (N,N_t,bin_z)
        # pdb.set_trace()
        # pdb.set_trace()
        # bin_z = np.linspace(self.bin_z[0],self.bin_z[-1],num_z + 1)
        ### calculate concurrence 
        tzm_raw_bin_counts, edges  = np.histogramdd(np.asanyarray(self.real_data), 
                                        bins=(self.bin_t,self.bin_z,self.bin_m), 
                                        normed=False, weights=None)
        
        tzm_raw_bin_counts = np.sum(tzm_raw_bin_counts,axis=2)
        fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(20,10))
        total = sample_ffz.sum(axis=1) ## shape (N,N_t)


        # sns.lineplot(sample_ffz.loc[0].index,total.loc[0].values/np.sum(total.loc[0].values), drawstyle='steps-pre',ax=ax[0],legend='brief',label='Sample')
        # sns.lineplot(sample_ffz.loc[0].index,tzm_raw_bin_counts[0]/np.sum(tzm_raw_bin_counts[0]),ax=ax[0],legend='brief',label='True')
        # pdb.set_trace()
        type_0_index = [(i,0) for i in range(len(sample_ff))]
        type_1_index = [(i,1) for i in range(len(sample_ff))]

        sns.lineplot(sample_ffz.columns,
                            np.mean((sample_ffz.loc[type_0_index].values/np.expand_dims(total.loc[type_0_index].values,axis=-1)),axis=0),
                             drawstyle='steps-pre',ax=ax[0],legend='brief',label='Sample')
        sns.lineplot(sample_ffz.columns,tzm_raw_bin_counts[0]/np.sum(tzm_raw_bin_counts[0]),ax=ax[0],legend='brief',label='True')
        
        ### sample data
        # sample_data = np.random.dirichlet(total.loc[0].values,size=100000)
        # sample_data = sample_data.T

        # ax[0].violinplot([data for data in sample_data], [x for x in sample_ffz.loc[0].index], points=60, widths=0.02,
        #              showmeans=True, showextrema=True, showmedians=True,
        #              bw_method='silverman')
        ax[0].violinplot([data for data in (sample_ffz.loc[type_0_index].values/np.expand_dims(total.loc[type_0_index].values,axis=-1)).T],
                             [x for x in sample_ffz.columns], points=60, widths=0.02,
                            showmeans=True, showextrema=True, showmedians=True,
                            bw_method=0.5)
        # sns.violinplot(x=np.tile(sample_ffz.loc[0].index.values ,len(sample_data)), y=sample_data.reshape(-1), ax=ax[0])
        ax[0].set_xlabel('z')
        ax[0].set_title('Type 0')

        # sns.lineplot(sample_ffz.loc[1].index,total.loc[1]/np.sum(total.loc[1]), drawstyle='steps-pre',ax=ax[1],legend='brief',label='Sample')
        # sns.lineplot(sample_ffz.loc[1].index,tzm_raw_bin_counts[1]/np.sum(tzm_raw_bin_counts[1]),ax=ax[1],legend='brief',label='True')

        sns.lineplot(sample_ffz.columns,
                            np.mean((sample_ffz.loc[type_1_index].values\
                            /np.expand_dims(total.loc[type_1_index].values,axis=-1)),axis=0),
                             drawstyle='steps-pre',ax=ax[1],legend='brief',label='Sample')
        sns.lineplot(sample_ffz.columns,tzm_raw_bin_counts[1]/np.sum(tzm_raw_bin_counts[1]),ax=ax[1],legend='brief',label='True')

        # sample_data = np.random.dirichlet(total.loc[1].values,size=100000)
        # sample_data = sample_data.T
        # ax[1].violinplot([data for data in sample_data], [x for x in sample_ffz.loc[1].index], points=60, widths=0.02,
        #              showmeans=True, showextrema=True, showmedians=True,
        #               bw_method='silverman')

        ax[1].violinplot([data for data in (sample_ffz.loc[type_1_index].values/np.expand_dims(total.loc[type_1_index].values,axis=-1)).T],
                             [x for x in sample_ffz.columns], points=60, widths=0.02,
                            showmeans=True, showextrema=True, showmedians=True,
                            bw_method=0.5)
        # sns.violinplot(x=np.tile(sample_ffz.loc[0].index.values,len(sample_data)), y=sample_data.reshape(-1), ax=ax[1])
        ax[1].set_title('Type 1')
        ax[1].set_ylabel('Frequency')
        fig.savefig(name, dpi=1200, format='png', bbox_inches='tight')
        pass 

    def MCMC_converge(self,save_path,figsize=(6,6),size=5):
        i = np.random.randint(low=0, high=len(self.bin_t) -1 ,size=size)
        j = random.sample(range(len(self.bin_z) -1),k=size) #np.random.randint(low=0, high=len(self.bin_z) -1 )
        k = random.sample(range(len(self.bin_m) -1),k=size)# np.random.randint(low=0, high=len(self.bin_m) -1)

        ff = self.data['fraction'][:,i,j,k]

        ### fig for ff
        fig,ax = plt.subplots(1,len(i),figsize=(figsize[0]*len(i),figsize[1]))
        for t in range(len(i)):
            ax[t].set_title(f"f{i[t],j[t],k[t]}")
            sns.lineplot(range(len(ff)),ff[:,t],ax=ax[t])
        
        fig.savefig(save_path + 'ff.png')

        ## fig for tzm
        g = random.sample(range(len(self.data['tzm'][0])),k=size) # np.random.randint(low=0, high=len(self.data['tzm'][0]),size=3)
        tzm = self.data['tzm'][:,g,:]

        fig,ax = plt.subplots(len(g),3,figsize=(figsize[0]* 3, figsize[1]* len(g)))
        for i in range(len(g)):
            for j in range(3):
                # ax[i,j].set_title(f"sample of tzm of galaxies{i}")
                sns.lineplot(range(len(self.data['tzm'])),self.data['tzm'][:,i,j],ax=ax[i,j])
        
        fig.savefig(save_path + 'tzm.png')

        pass 

    def scatter_z(self,save_path,figsize=(6,6)):
        """
        """
        
        fig,ax = plt.subplots(figsize=figsize)
        # pdb.set_trace()

        sns.scatterplot(x=self.real_data[:,1], y=np.mean(self.data['tzm'][:,:,1],axis=0),ax = ax)
        # pdb.set_trace()
        ax.set(xlabel='True z', ylabel='estimate z')
        fig.savefig(save_path + 'scater_z.png')

        pass 
   
    def confusion_t(self,save_path,figsize=(6,6)):
        """
        confusion matrix
        """
        estimate_t,_ = stats.mode(self.data['tzm'][:,:,0] > 0.5,axis=0)
        estimate_t = np.array(estimate_t,dtype=np.uint8).squeeze()
        print(estimate_t.shape)
        print(self.real_data[:,0].shape)
        conf = confusion_matrix(self.real_data[:,0],estimate_t)
        conf = conf/np.sum(conf)
        df_cm = pd.DataFrame(conf, range(len(conf)), range(len(conf)))
        fig,ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},ax=ax)
        fig.savefig(save_path + 'confusion_matrix.png')

        pass 

