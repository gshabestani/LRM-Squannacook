# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:38:20 2021

@author: ghaza
"""

import Func_Lib as func
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
plt.style.use('seaborn-white')
import time
import os
from statsmodels.graphics.tsaplots import plot_acf
#%%


data=pd.read_excel('adjusted.xlsx')
# loading the observed and simulated flow
data['month']=pd.DatetimeIndex(data['date']).month

#data=data.drop(columns=['diff'])

data['lambda']=np.log(data['Qmodel']/data['Qgage'])
# generating emperical log-ratio errors

lamda=np.log(data['Qmodel']/data['Qgage'])




p=3
q=0
mod = ARMA(data['lambda'], order=(p,q))

res = mod.fit()

# performing an AR3 model on log-ratio errors

data['Ar_res']=res.resid


n=100

m=100

# m is the number of random AR parameter sets, Beta, generated for considering AR model uncertainity.
# n is the number of stochastich streamflow realizations generated for each Beta set.

Beta=np.random.multivariate_normal(np.array(res.params),np.array(res.cov_params()),m)
# random Beta sets for AR uncertainity consideration.

E=func.simple_bootstrap(np.array(data['Ar_res']),n*m).T
# Generating stochastic residuals with simple bootstrap from emperical AR residuals


l=np.zeros((len(data),m*n))

for i in range(0,m*n):
    
    l[0:4,i]=lamda[0:4]


for i in range(0,m):    
    B=Beta[i,:]
    for t in range(4,13514):
        l[t,i*n:((i+1)*n)]=B[0]+B[1]*l[t-1,i*n:((i+1)*n)]+B[2]*l[t-2,i*n:((i+1)*n)]+B[3]*l[t-3,i*n:((i+1)*n)]+E[t,i*n:((i+1)*n)]
        
# l is a matrix of [len(data),m*n] representing stochastic log-ratio errors         
        
Q=np.array(data['Qmodel'])/np.exp(l.T)

BCF=np.zeros(len(data))
for i in range(0,len(data)):
    BCF[i]=1/np.exp(-np.mean(l[i,:])+np.var(l[i,:])/2)
    
for t in range(0,len(data)):
    Q[:,t]=Q[:,t]*BCF[t]
    
# Q is stochasticly simulated streamflows matrix [m*n, len(data)]

# BCF is transformation bias correction factor.

np.save('Q.npy',Q)

np.save('StochasticE.npy',E)

np.save('L.npy',l)