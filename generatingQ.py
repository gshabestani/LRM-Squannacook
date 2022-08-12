# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:22:51 2022

@author: gshabe01
"""

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
import Func_Lib2 as func

data=pd.read_csv('adjusted.csv')
data['month']=pd.DatetimeIndex(data['date']).month
data['date']=pd.DatetimeIndex(data['date'])
data=data.drop(columns=['diff'])

data['lambda']=np.log(data['Qmodel']/data['Qgage'])
lamda=np.log(data['Qmodel']/data['Qgage'])

print(data.head(5))

p=3
q=0


mod = ARMA(data['lambda'], order=(p,q))

res = mod.fit()


Epsilon=pd.DataFrame(res.resid)
Epsilon['Qmodel']=np.array(data['Qmodel'])
Epsilon=np.array(Epsilon)
E=[]
for t in range(0,len(data)):
    E.append(Epsilon-[0,Epsilon[t,1]])

# the k nearest to the flow
#K=[116, 200, 300, 500, 600, 700, 1000, 3000]
k=700

Er=np.zeros((len(data),k))
for t in range(0,len(data)):
    E[t][:,1]=np.abs(E[t][:,1])
    x=E[t][np.argsort(E[t][:, 1])]
    Er[t,:]=x[0:k,0]
    
n=100

m=100
R=np.zeros((len(data),n*m))
for i in range(0,n*m):
    I= np.random.randint(low=0,high=k,size=len(data))
    R[:,i]= Er[[np.arange(0,len(data),1),I]]


Beta=np.random.multivariate_normal(np.array(res.params),np.array(res.cov_params()),m)




l2=np.zeros((13514,m*n))

for i in range(0,m*n):
    
    l2[0:4,i]=lamda[0:4]


for i in range(0,m):    
    B=Beta[i,:]
    for t in range(4,13514):
        l2[t,i*n:((i+1)*n)]=B[0]+B[1]*l2[t-1,i*n:((i+1)*n)]+B[2]*l2[t-2,i*n:((i+1)*n)]+B[3]*l2[t-3,i*n:((i+1)*n)]+R[t,i*n:((i+1)*n)]

        



Q2=np.array(data['Qmodel'])/np.exp(l2.T)

# BCF=np.zeros(len(data))
# for i in range(0,len(data)):
#     BCF[i]=1/np.exp(-np.mean(l[i,:])+np.var(l[i,:])/2)
    
# for t in range(0,len(data)):
#     Q[:,t]=Q[:,t]*BCF[t]



#Q=func.Q_noBCF(np.array(data['Qmodel']),np.array(data['month']),  l.T, mean_m, std_m )

BCF=1/np.exp(-np.mean(data['lambda'])+np.var(data['lambda'])/2)
Q=Q2*BCF

np.save('QKnn700.npy',Q)

