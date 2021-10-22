from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from utils.util_photometric import read_template,read_flux,read_file_tzm,mean_flux
from utils.utils import DataGenerator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pdb
import torch
import torch.nn as nn
from torch.optim import Adam
import random


def define_function():
    S_N_ratio = 5.0
    num_samples = None
    X = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat',cols=[4],num=num_samples)
    y = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat',cols=[2,4,14],num=num_samples)



    template = read_template(paths=['/data/phuocbui/MCMC/reimplement/data/Ell2_flux_z_no_header.dat','/data/phuocbui/MCMC/reimplement/data/Ell5_flux_z_no_header.dat'])

    ## mabe use max_z = 1.3??
    # max_z = template['z'][1].max()
    max_z = 1.3
    print(max_z)
    index = [i for i,j in enumerate(y) if j[1] < max_z]
    y.index = np.array(y.index)[index]
    X.index = y.index
    print(f"Length y:{len(y)}, Length X: {len(X)}")

    ## pdb.set_trace()
    X = np.array([x for x in X])
    print(f"Min z: {np.min(X[:]):.2f}")
    print(f"Max z: {np.max(X[:]):.2f}")
    y = np.array([mean_flux(tzm,template) for tzm in y])
    y_std = np.array(list(map(lambda x:  [i/S_N_ratio if i > 0.0 else 1000.0 for i in x],y)))
    y_std = (1/y_std)**2


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LinearRegression().fit(X_train,y_train)
    print(f"Error:{clf.score(X_test,y_test)}")
    print(f"Iter: {1}, Intecept:{clf.intercept_},Coef: {clf.coef_}")
    np.savez('mean.npz', intecept=clf.intercept_, coef=clf.coef_)

# data = np.load('mat.npz')
# print data['name1']
# print data['name2']

    X_train, X_test, y_train, y_test = train_test_split(X, y_std, test_size=0.2, random_state=42)
    clf = LinearRegression().fit(X_train,y_train)
    print(f"Error:{clf.score(X_test,y_test)}")
    print(f"Iter: {1}, Intecept:{clf.intercept_},Coef: {clf.coef_}")
    np.savez('std.npz', intecept=clf.intercept_, coef=clf.coef_)

    
    
    pass

