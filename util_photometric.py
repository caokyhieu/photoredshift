import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import bisect

def mean_flux(tzm,template):
        """
        flux: {F_b}g, here we have N_b bands
           calculate P({F_b}g|t_g,z_g,m_g)
           not depend on m
        return mean flux at t_type and z_bin
        """
        t,z,m = tzm
        t = int(t)
        # idx = bisect.bisect_left(self.template['z'][t],z)
        z_idx_1 = np.abs(np.array(template['z'][t]) - z).argmin()
        if template['z'][t][z_idx_1] < z:
            z_idx_2 = z_idx_1+1 
        else:
            z_idx_2 = z_idx_1-1
        obs_mean_1 = np.array(template['F_bands'][t][z_idx_1])
        obs_mean_2 = np.array(template['F_bands'][t][z_idx_2])

        linear_weight = (z-template['z'][t][z_idx_1])/(template['z'][t][z_idx_2] - template['z'][t][z_idx_1])
        obs_mean = obs_mean_1 + linear_weight*(obs_mean_2-obs_mean_1) 

        return obs_mean

def find_tzm_index(tzm,bin_edge):
        t,z,m = tzm
        bin_t,bin_z,bin_m = bin_edge
        zbin = len(bin_z) - 1
        mbin = len(bin_m) - 1
        """
        this function helps to get index in f vector by t,z,m value
        """

        ff_samples_index = ((bisect.bisect_left(bin_t, t)-1)*zbin*mbin) \
                     + ((bisect.bisect_left(bin_z, z)-1)*mbin)     \
                     +   bisect.bisect_left(bin_m, m)
        return ff_samples_index


def read_file_tzm(path):
    " read true tzm.csv to a list of tzm"
    tzm_list = []
    with open(path,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split(',')
            tzm_list.append([float(t) for t in line])
    return tzm_list

def read_template(paths,columns = [1, 3, 5, 11, 13]):
    """
    read dat file and return dataframe with Flux and magnitude for each color band
    """
    data = {'z':[],'F_bands':[]}
    for i in range(len(paths)):
        df = pd.read_csv(paths[i],skiprows=1,sep=",\s+", header=None,engine='python')
        data['z'].append(df[0].values)
        data['F_bands'].append(df[columns].values)
    return data

def read_flux(path):
    data = pd.read_csv(path,header=None).values
    return data