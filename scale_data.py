from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from util_photometric import read_template,read_flux,read_file_tzm,mean_flux
from utils import DataGenerator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pdb
import torch
import torch.nn as nn
from torch.optim import Adam
import random
num_samples = None
X = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat',cols=[2,4,14,19,24,29,34],num=num_samples)
y = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat',cols=[2,4,14],num=num_samples)



template = read_template(paths=['Ell2_flux_z_no_header.dat','Ell5_flux_z_no_header.dat'])

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
print(f"Min z: {np.min(X[:,1]):.2f}")
print(f"Max z: {np.max(X[:,1]):.2f}")
y = np.array([mean_flux(tzm,template) for tzm in y])
def index_in_range(args):
    x,min_,max_ = args
    result_list = []
    for i,j in enumerate(x):
        if j[1]<max_ and j[1] > min_:
            result_list.append(i)
    return result_list

temp_lin = np.linspace(0,1.3,13)
result_indexes = list(map(index_in_range,zip([X]*(len(temp_lin)-1),temp_lin[:-1],temp_lin[1:])))


# random_y = np.array([np.random.normal(scale=i,size=1) for i in j fonp.log10(y[:,-1])])
mask = y > 0
random_y = (mask * y/5. + 1000 * (1 - mask)).flatten()
random_y = np.array([np.random.normal(scale=i,size=1) for i in random_y]).reshape(*y.shape)
y =y + random_y

### use linear model
# clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_samples=0.5,min_samples_split=10,max_depth=5)).fit(X, y)
# clf = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=10,subsample=0.5)).fit(X, y)
# clf = MultiOutputRegressor(LinearRegression()).fit(X[:,1:], y)
# print(clf.score(X[:,1:], y))
# X[:,2:] = clf.predict(X[:,1:])



# pdb.set_trace()
X[:,2] = -99.
for idxs in result_indexes:
    print(f"Length data: {len(idxs)}")
    for i in range(1,5):
        # print(y.shape)
        # print(f"X shape: {X[:,[1,i+2]][idxs].shape}, y shape: {y[:,i][idxs].shape}")
        #clf = RandomForestRegressor(n_estimators=100,max_samples=0.5,min_samples_split=10,max_depth=5).fit(X[:,[1,i+2]][idxs],y[:,i][idxs])
        X_train, X_test, y_train, y_test = train_test_split(X[:,[1,i+2]][idxs], y[:,i][idxs], test_size=0.3, random_state=42)
        clf = LinearRegression().fit(X_train,y_train)

        
        print(f"Error:{clf.score(X_test,y_test)}")
        print(f"Iter: {i}, Intecept:{clf.intercept_},Coef: {clf.coef_}")
        X[idxs,i+2] = clf.predict(X[:,[1,i+2]][idxs])

    # clf = MultiOutputRegressor(LinearRegression()).fit(X[idxs,1:], y[idxs,:])
    # X[idxs,2:] = clf.predict(X[idxs,1:])
    # print(f"Score:{clf.score(X[idxs,1:], y[idxs,:])}")


        # print(clf.feature_importances_)


data = pd.DataFrame(X)
data.to_csv('/data/phuocbui/MCMC/reimplement/data/clean_data_v2.csv',header=None,index=False)

# for es in clf.estimators_:
#     print(es.coef_)

# for es in clf.estimators_:
#     print(es.intercept_)
# for es in clf.estimators_:
#     print(es.feature_importances_)

### use deep learning
# class SimpleModel(nn.Module):
#     def __init__(self,input_size,output_size,n_layers=8):
#         super().__init__()
#         self.input_size = input_size 
#         self.output_size = output_size
#         self.model = nn.Sequential()
#         start = self.input_size
#         end = self.output_size
#         for i in range(n_layers-1):
#             self.model.add_module(f'fc_{i}',nn.Linear(start,end))
#             self.model.add_module(f'Relu_{i}',nn.ReLU())
#             start = end
#         self.model.add_module(f'fc_{n_layers-1}',nn.Linear(start,end))

#     def forward(self,x):
#         x = self.model(x)
#         return x
        

# batchsize=100
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# index = [i for i in range(len(X))]
# n_epoch = 300
# input_size = 6
# output_size = 5
# n_layers = 20
# lr = 1e-2
# model = SimpleModel(input_size,output_size,n_layers)
# model = model.to(device)
# loss_function = nn.MSELoss()

# optim = Adam(model.parameters(),lr=lr)

# for e in range(n_epoch):
#     train_loss = []
#     for start in range(0,len(index),batchsize):
#         X_train = torch.from_numpy(X[index[start:start+batchsize]]).to(device).to(torch.float)
#         y_train = torch.from_numpy(y[index[start:start+batchsize]]).to(device).to(torch.float)
#         pred = model(X_train)
#         loss = loss_function(pred,y_train)

#         ## optimize
#         optim.zero_grad()
#         loss.backward()
#         optim.step()

#         train_loss.append(loss.cpu().detach().numpy())
    
#     random.shuffle(index)
    
#     print(f"Epoch {e}, Loss : {np.mean(train_loss):.2f}")

        
# X = DataGenerator(path='/data/phuocbui/MCMC/reimplement/data/MockCatalogJouvel.dat',cols=[2,4,14,19,24,29,34],num=num_samples)
# X = np.array([x for x in X])
# X[:,2:] = clf.predict(X[:,1:])
# data = pd.DataFrame(X)
# data.to_csv('/data/phuocbui/MCMC/reimplement/data/clean_data.csv',header=None,index=False)
## save clean data

