from pyro_code.run import run_exp
from itertools import product
import os
# load_encoder('encoder.pth')

m1= -1.
# m2= 4.
sig1=2**2
# sig2=4.
a=1
b=1.
c=1.
list_m2 = [ 10.]
list_sig2 = [2.**2]
for m2,sig2 in product(list_m2,list_sig2):
    name_folder = f"fig/miss_RBF_linear_m1={m1}_m2={m2}_sig1={sig1}_sig2={sig2}_a={a}_b={b}_c={c}"
    os.makedirs(name_folder,exist_ok=True)
    run_exp(m1,m2,sig1,sig2,a,b,c,path_fig=name_folder)