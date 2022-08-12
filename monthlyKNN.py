# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:58:47 2022

@author: gshabe01
"""
import Func_Lib2 as func
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
plt.style.use('seaborn-white')
import time
from scipy.stats import pearson3
import os
from glob import glob
import sys
#%%


#%%
print('monthly calibrated')
data=pd.read_csv('adjusted.csv')
data['month']=pd.DatetimeIndex(data['date']).month
data['date']=pd.DatetimeIndex(data['date'])
data=data.drop(columns=['diff'])

data['lambda']=np.log(data['Qmodel']/data['Qgage'])
lamda=np.log(data['Qmodel']/data['Qgage'])

print(data.head(5))

p=3
q=0


mod = ARIMA(data['lambda'], order=(p,0,q))

res = mod.fit()
data['Ar_res']=res.resid
n=100
m=100
#K=[116, 200, 300, 500, 600, 700, 1000]
#print(int(sys.argv[1]))
#k=K[int(sys.argv[1])]
k=100
print(k)
E=np.zeros((len(data),m*n))
for i in range(0,len(data)):
    df=data.loc[data['month']== data['month'].loc[i]]
    df['Qmodel']=np.abs(df['Qmodel']-df['Qmodel'].loc[i])
    df=df.sort_values(by=['Qmodel'])
    I=np.random.randint(low=0,high=k, size=n*m)
    E[i,:]=np.array(df['Ar_res'])[I]



Beta=np.random.multivariate_normal(np.array(res.params),np.array(res.cov_params()),m)




l2=np.zeros((13514,m*n))

for i in range(0,m*n):
    
    l2[0:4,i]=lamda[0:4]


for i in range(0,m):    
    B=Beta[i,:]
    for t in range(4,13514):
        l2[t,i*n:((i+1)*n)]=B[0]+B[1]*l2[t-1,i*n:((i+1)*n)]+B[2]*l2[t-2,i*n:((i+1)*n)]+B[3]*l2[t-3,i*n:((i+1)*n)]+E[t,i*n:((i+1)*n)]

        



Q2=np.array(data['Qmodel'])/np.exp(l2.T)
BCF=1/np.exp(-np.mean(data['lambda'])+np.var(data['lambda'])/2)
Q=Q2*BCF
