from pyro_code.refactor import  run_experiment_basic_function
from itertools import product
import os
import numpy as np
# load_encoder('encoder.pth')

m1=-1.
m2=2.
sig1=1.
sig2=2.
a=1
b=2
c=3
n_l=100
n_u=100
p=np.array([1,5,9])
gamma=np.array([1,2,4])

run_experiment_basic_function(m1,m2,sig1,sig2,
                                a,b,c,n_l,n_u,
                                p,gamma)


