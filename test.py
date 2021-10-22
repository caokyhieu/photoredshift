from variational.stochastic_variational import train,train_,train_v2,load_encoder
from variational.params import define_function
from variational.HCMC import run
import pandas as pd
import json
import time
# load_encoder('encoder.pth')
start = time.time()
train(threshold=164.24800,encode=False) ##1860472803
# train_v2(threshold=1646852480.0 * 5,encode=False)
# train_(threshold=16673.,threshold2=6010430.5)
# run()
# def load_data(path):
#     with open(path) as file:
#         string = file.read()
#         obj = json.loads(string)
    
# define_function()
print(f"Finish train time: {time.time() - start}")