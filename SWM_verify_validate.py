# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:38:38 2022

@author: ghazal
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
import seaborn as sns
#%%


data=pd.read_excel('adjusted.xlsx')
# loading the observed and simulated flow
data['month']=pd.DatetimeIndex(data['date']).month

#data=data.drop(columns=['diff'])

data['lambda']=np.log(data['Qmodel']/data['Qgage'])
# generating emperical log-ratio errors


Q=np.load('Q.npy')

E=np.load('StochasticE.npy')

l=np.load('L.npy')

m=100
n=100

# m is the number of random AR parameter sets, Beta, generated for considering AR model uncertainity.
# n is the number of stochastich streamflow realizations generated for each Beta set.

Stochastic=func.Flow_Exceedance(Q,data,m*n)

#calculating inputs for a flow duration curve: Sorted stochastic flow, Exceedance , non Exceedance, rank and observatin columns.

storageSynt , AnnualVolume ,R=func.Storage_yield(Q, data,m*n)

# calculating information needed for a storage yield curve



Annualmax=func.annualmax(Q,data,m*n)
# generating annual maximums and their exceedance probability


low7=func.low7day(Q,data,m*n)
# generating minimum of 7-day averages and their exceedance probability  



mean_t,std_t,skew_t=func.t_stats(Q, data)

# generating unbiased monthly t-ratios of the stochastic simulation 


Q7_95,Q7_99,Q7_50, data=func.confidenceband_p(data,Q)

Qobs=np.array(data['Qgage'])
Qband50=np.zeros(len(Qobs))
Qband95=np.zeros(len(Qobs))
for t in range (0, len(Qobs)):
    
    Qband50[t]=(Qobs[t] >= Q7_50[0][t]) and (Qobs[t]<= Q7_50[1][t])
    Qband95[t]=(Qobs[t] >= Q7_95[0][t]) and (Qobs[t]<= Q7_95[1][t])
Band50_ratio=np.mean(Qband50)
Band95_ratio=np.mean(Qband95)
print('ratio of observations in the 95% confidence interval:',Band95_ratio)
print('ratio of observations in the 50% confidence interval:',Band50_ratio)

# calculating the coverage probability of the stochastic simulation for 50% and 95% confidence intervals

#%%
# first validation plot: Storage Yield Curce, flow duration curve, 7-day low flow exceedance probability and annual maximum exceedance probability
 
fig=plt.figure(figsize=(10,10),constrained_layout=True, dpi=300)
gs = fig.add_gridspec(8, 4, wspace=0.05,hspace=0.05)
    
sub1 = fig.add_subplot(gs[0:2,0:2])
Rt=np.linspace(1,R,len(storageSynt))


sub1.fill_between(Rt/R,(strg_max_min['min']/AnnualVolume),(strg_max_min[0]/AnnualVolume),color='silver')


sub1.plot(Rt/R,storageSynt[:,storageSynt.shape[1]-2]/AnnualVolume,color= 'g',label='Observation')
sub1.plot(Rt/R,storageSynt[:,storageSynt.shape[1]-1]/AnnualVolume,color='k',label='Deterministic')

sub1.set_xlabel('Yield / Annual Mean Flow')
sub1.set_ylabel('Storage / Annual Mean Volume')
# plt.title('Storage Yield Curve')
# sub1.grid()
sub1.plot([],[], 'silver', label='Stochastic')
sub1.legend(loc='best')
sub1.set_ylim(0,2.5)

sub2=fig.add_subplot(gs[0:2,2:4])
S=np.array(Stochastic)[:,0:10000]
S_max_min=pd.DataFrame(np.max(S,axis=1))
S_max_min['min']=np.min(S,axis=1)

sub2.fill_between(norm.ppf(Stochastic['Exeedance']),S_max_min['min'], S_max_min[0],color='silver')
ax1=sub2.plot(norm.ppf(Stochastic['Exeedance']),Stochastic['obs'], color='green',lw=2,label='Observation')
ax2=sub2.plot(norm.ppf(Stochastic['Exeedance']),Stochastic['histmodel'], color='k',lw=2,label='Deterministc')

#plt.title(title)
sub2.set_ylabel('Flow')
sub2.set_xlabel('Exceedance Probability')
plt.xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999])
sub2.set_xlim(-3,3)
sub2.set_yscale('log')
sub2.plot([],[], 'silver', label='Stochastic')
sub2.legend(loc='best')
# sub2.grid()
# plt.title('Flow Duration Curve')

sub3=fig.add_subplot(gs[2:4,0:2])


sub3.fill_between(norm.ppf(np.array(low7['non_Exceedance'])),np.array(low7)[:,0:10000].max(axis=1),np.array(low7)[:,0:10000].min(axis=1),color='silver')
sub3.plot(norm.ppf(low7['non_Exceedance']),low7['obs'], color='green',lw=2,label='Observation')
sub3.plot(norm.ppf(low7['non_Exceedance']),low7['histmodel'], color='k',lw=2,label='Deterministic')
# sub3.set_title('7-day Low Flow Duration curve')
sub3.set_ylabel('Annual 7-day Minimum Flow')
plt.xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999])
sub3.set_xlabel('Exceedance Probability')
# sub3.grid()
sub3.plot([],[], 'silver', label='Stochastic')
sub3.legend(loc='best')


sub4 = fig.add_subplot(gs[2:4,2:4])


sub4.fill_between(norm.ppf(np.array(Annualmax['Exeedance'])),np.array(Annualmax)[:,0:10000].max(axis=1),np.array(Annualmax)[:,0:10000].min(axis=1),color='silver')
sub4.plot(norm.ppf(Annualmax['Exeedance']),Annualmax['obs'], color='green',lw=2,label='Observation')
sub4.plot(norm.ppf(Annualmax['Exeedance']),Annualmax['histmodel'], color='k',lw=2,label='Deterministic')
# sub4.set_title('Annual Maximum Flow Duration')
sub4.set_ylabel('Annual Maximum Flow')
sub4.set_xlabel('Exceedance Probability')
plt.xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999])
sub4.set_yscale('log')
#plt.legend([ax1, ax2], ['deter-model','observation' ], loc='upper right')
# sub4.grid()
sub4.plot([],[], 'silver', label='Stochastic')
sub4.legend(loc='best')
fig.savefig("pannelsimple.png", bbox_inches='tight', dpi=1000)

#%%

# second validation plot: coverage probability timeseries, and  monthly t-ratio plots
data=data.reset_index()
fig=plt.figure(figsize=(10,6),constrained_layout=True, dpi=300)
gs = fig.add_gridspec(5, 6, wspace=0.05,hspace=0.05)
data=data.set_index('date')
sub1= fig.add_subplot(gs[0:3,:])

# for i in range(0,10):
#     plt.plot(l_hat['date'].loc['01-1995':'01-1996'],l_hat[r[i]].loc['01-1995':'01-1996'],color='silver')
data['Qgage'].loc['01-1994':'01-1996'].plot(color='green',label='Observation',ax=sub1)
data['Qmodel'].loc['01-1994':'01-1996'].plot(color='k',label='Deterministic',ax=sub1)
sub1.fill_between(data.loc['01-1994':'01-1996'].index, data['2.5 quantile'].loc['01-1994':'01-1996'], data['97.5 quantile'].loc['01-1994':'01-1996'], color='silver',label='95 CI')
sub1.fill_between(data.loc['01-1994':'01-1996'].index, data['25 quantile'].loc['01-1994':'01-1996'], data['75 quantile'].loc['01-1994':'01-1996'], color='yellow',label='50 CI')
sub1.set_ylabel('Flow')
sub1.plot([],[], 'silver', label='Stochastic')
plt.legend()
plt.yscale('log')
# plt.grid()
data=data.reset_index()

sub2 = fig.add_subplot(gs[3:5, 0:2])
sub2.plot([1,2,3,4,5,6,7,8,9,10,11,12],mean_t, color='k')
sub2.scatter([1,2,3,4,5,6,7,8,9,10,11,12],mean_t, color='k')
sub2.set_ylim(-2.5,2.5)
sub2.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
sub2.set_yticks([-2,0,2])
# sub2.set_title('Mean Monthly t-ratio')
sub2.set_xlabel('Months')
sub2.set_ylabel('Mean t Value')
# sub2.grid()

sub5 = fig.add_subplot(gs[3:5, 2:4])
sub5.plot([1,2,3,4,5,6,7,8,9,10,11,12],std_t, color='k')
sub5.scatter([1,2,3,4,5,6,7,8,9,10,11,12],std_t, color='k')
sub5.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
sub5.set_ylim(-2.5,2.5)
sub5.set_yticks([-2,0,2])
# sub5.set_title('Standard Deviation Monthly t-ratio')
sub5.set_xlabel('Months')
sub5.set_ylabel('Standard Deviation t value')
# sub5.grid()

sub6 = fig.add_subplot(gs[3:5, 4:6])
sub6.plot([1,2,3,4,5,6,7,8,9,10,11,12],skew_t, color='k')
sub6.scatter([1,2,3,4,5,6,7,8,9,10,11,12],skew_t, color='k')
sub6.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
sub6.set_ylim(-2.5,2.5)
sub6.set_yticks([-2,0,2])
# sub6.set_title('skew Monthly t-ratio')
sub6.set_xlabel('Months')
sub6.set_ylabel('Skewness t Value')
# sub6.grid()
fig.savefig("timeseriessimple.png", bbox_inches='tight', dpi=1000)
#%%
#validation plots 3 : Design flow histograms


QFrame=pd.DataFrame(Q.T,index=data['date'])

low7=QFrame.rolling(7).mean().resample('Y').min()
low7=pd.DataFrame(np.sort(np.array(low7),axis=0))
low7['Exceedance']=1-low7[0].rank()/len(low7)

low7Q10_hat=low7.loc[(0.89 <= low7['Exceedance']) & (low7['Exceedance'] <= 0.9)]

low7Q10_hat=low7Q10_hat.drop(columns=['Exceedance']).T



plt.figure()
sns.distplot(low7Q10_hat,kde=False, color='gray',label='Stochastic')
plt.plot([5.68,5.68],[0,600], color='g',linestyle='solid',label='Observation')
plt.plot([4.31,4.31],[0,600], color='k',linestyle='solid',label='Deterministic')
plt.plot([3.302,3.302],[0,600], color='g',linestyle='dashed')
plt.plot([9.78,9.78],[0,600], color='g',linestyle='dashed')
plt.xlabel('7Q10 Flow', fontsize=20)
plt.ylabel('Frequency',fontsize=20)

coverage7Q10=np.mean((3.302<=low7Q10_hat) & (low7Q10_hat <=9.78))
print('coverage probability:',coverage7Q10)




plt.legend(prop={'size':16}, loc='best')
plt.savefig('lambda10007Q10.png',dpi=1000)
# 7Q10 histogram and coverage 

QFrame['obs']=np.array(data['Qgage'])
QFrame['model']=np.array(data['Qmodel'])
ann_max=pd.DataFrame(np.sort(QFrame.resample('Y').max(),axis=0))
ann_max['non_Exceedance']=ann_max[0].rank()/len(ann_max)


### fitting a (pearson 3) distribution to annual maximums. and calculating the 100 year flood ( 0.99 quantile) with wilson-hilferty equation.
#X(p)=Mu + Sigma * (2 /skew *(1 + (skew * Zp)/6 -(skew **2)/36) **3 - 2/skew)
from scipy.stats import norm
def wilson(p,realization):
    zp=norm.ppf(p)
    Xp=realization.mean() + realization.std() * (2 /realization.skew() *(1 + (realization.skew() * zp)/6 -(realization.skew() **2)/36) **3 - 2/realization.skew())
    return Xp

Qmax=np.array(ann_max)

flood100=np.zeros(10000)
for i in range(0,10000):
    flood100[i]=wilson(0.99,pd.DataFrame(Qmax[:,i]))
flood100obs=wilson(0.99,ann_max[10000])
flood100model=wilson(0.99,ann_max[10001])


plt.figure()
sns.distplot(flood100,kde=False, color='gray',label='Stochastic')
plt.plot([flood100obs,flood100obs],[0,1500], color='g',linestyle='solid',label='Observation')
plt.plot([flood100model,flood100model],[0,1500], color='k',linestyle='solid',label='Deterministic')
plt.plot([2682,2682],[0,1500], color='g',linestyle='dashed')
plt.plot([9413,9413],[0,1500], color='g',linestyle='dashed')
plt.xlabel('100 Year Flood Flow',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xlim(0,20000)
plt.legend(prop={'size': 16})
coverage100=np.mean((2682<=flood100) & (flood100 <= 9413))
print('coverage probability:',coverage100)

plt.savefig('lambda_100y.png',dpi=1000)
# 100-year flood histogram and coverage probability


flood50=np.zeros(10000)
for i in range(0,10000):
    flood50[i]=wilson(0.98,pd.DataFrame(Qmax[:,i]))
flood50obs=wilson(0.98,ann_max[10000])
flood50model=wilson(0.98,ann_max[10001])


plt.figure()
sns.distplot(flood50,kde=False, color='gray',label='Stochastic')
plt.plot([flood50obs,flood50obs],[0,1500], color='g',linestyle='solid',label='Observation')
plt.plot([flood50model,flood50model],[0,1500], color='k',linestyle='solid',label='Deterministic')
plt.plot([2505,2505],[0,1500], color='g',linestyle='dashed')
plt.plot([7101,7101],[0,1500], color='g',linestyle='dashed')
plt.xlabel('50 Year Flood Flow', fontsize=20)
plt.ylabel('frequency', fontsize=20)
plt.xlim(0,20000)
plt.legend(prop={'size': 16})
coverage50=np.mean((2505<=flood50) & (flood50 <= 7101))
print('coverage probability:',coverage50)
plt.savefig('lambda50y.png',dpi=1000)

# 50 year histogram and coverage probability


flood500=np.zeros(10000)
for i in range(0,10000):
    flood500[i]=wilson(0.998,pd.DataFrame(Qmax[:,i]))
flood500obs=wilson(0.998,ann_max[10000])
flood500model=wilson(0.998,ann_max[10001])


plt.figure()
sns.distplot(flood500,kde=False, color='gray',label='Stochastic')

plt.plot([flood500obs,flood500obs],[0,1500], color='g',linestyle='solid',label='Observation')
plt.plot([flood500model,flood500model],[0,1500], color='k',linestyle='solid',label='Deterministic')
plt.plot([2949,2949],[0,1500], color='g',linestyle='dashed')
plt.plot([17592,17592],[0,1500], color='g',linestyle='dashed')
plt.xlabel('500 Year Flood Flow',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.legend(prop={'size': 16})
plt.xlim(0,40000)
coverage500=np.mean((2949<=flood500) & (flood500 <= 17592))
print('coverage probability:',coverage500)
plt.savefig('lambda500y.png',dpi=1000)

# 500 year histogram and coverage probability

flood2=np.zeros(10000)
for i in range(0,10000):
    flood2[i]=wilson(0.5,pd.DataFrame(Qmax[:,i]))
flood2obs=wilson(0.5,ann_max[10000])
flood2model=wilson(0.5,ann_max[10001])

plt.figure()

plt.figure()
sns.distplot(flood2,kde=False, color='gray',label='Stochastic')
plt.plot([flood2obs,flood2obs],[0,800], color='g',linestyle='solid',label='Observation')
plt.plot([flood2model,flood2model],[0,800], color='k',linestyle='solid',label='Deterministic')
plt.plot([899,899],[0,800], color='g',linestyle='dashed')
plt.plot([1499.57,1499.57],[0,800], color='g',linestyle='dashed')
plt.xlabel('2 Year Flood Flow',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xlim(0,2000)
plt.legend(prop={'size': 16})

coverage2=np.mean((899.18<=flood2) & (flood2 <=1499.57))
print('coverage probability:',coverage2)
plt.savefig('lambda2y.png',dpi=1000)

# 2 year histogram and coverage probability

flood10=np.zeros(10000)
for i in range(0,10000):
    flood10[i]=wilson(0.9,pd.DataFrame(Qmax[:,i]))
flood10obs=wilson(0.9,ann_max[10000])
flood10model=wilson(0.9,ann_max[10001])

plt.figure()

plt.figure()
sns.distplot(flood10,kde=False, color='gray',label='Stochastic')
plt.plot([flood10obs,flood10obs],[0,800], color='g',linestyle='solid',label='Observation')
plt.plot([flood10model,flood10model],[0,800], color='k',linestyle='solid',label='Deterministic')
plt.plot([1893.68,1893.68],[0,800], color='g',linestyle='dashed')
plt.plot([3566.99,3566.99],[0,800], color='g',linestyle='dashed')
plt.xlabel('10 Year Flood Flow', fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xlim(0,6000)
plt.legend(prop={'size': 16})
coverage10=np.mean((1893.68<=flood10) & (flood10 <= 3566.99))
print('coverage probability:',coverage10)
plt.savefig('lambda10y.png',dpi=1000)

# 10 year histogram and coverage probability
#%%
# verification 

data['lambda']=np.log(data['Qmodel']/data['Qgage'])
# generating emperical log-ratio errors

p=3
q=0
# ARMA model order, chosen by AIC or BIC before


mod = ARMA(data['lambda'], order=(p,q))

res = mod.fit()
# fitting the AR model
# performing an AR3 model on log-ratio errors

data['Ar_res']=res.resid

# checking AR residuals distribution
plt.figure()
stats.probplot(data['Ar_res'],plot=plt,rvalue=True)
plt.ylabel('ordered $\epsilon_t$', size=20)
plt.plot([0,0],[data['Ar_res'].min(),data['Ar_res'].max()],color='k')
plt.xlabel('quantiles of standard normal', size=20)
plt.xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999],rotation = 'vertical')
plt.save('epsilon_Exceedance.svg',dpi=500)

#checking heteroscedastisity and identical distribution

fig, ax = plt.subplots(2,figsize=(10,10))
ax[0].scatter(data['Qmodel'],data['Ar_res'], color='k')
ax[0].grid()
ax[0].set_xscale('log')
ax[0].set_ylabel('$\epsilon_t$',rotation=90, size= 20)

i=np.random.randint(0,10000)
ax[1].scatter(data['Qmodel'],E[:,i], color='k')
ax[1].grid()
ax[1].set_xscale('log')
ax[1].set_ylabel('$\~\epsilon_t$',rotation=90, size=20)

plt.savefig('AR_resflow.png',dpi=1000)


# checking for auto corelations

l_frame=pd.DataFrame(l)

lag=np.zeros((10000,11))

for i in range(0,11):
    for r in range(0,10000):
        lag[r,i]=l_frame[r].autocorr(lag=i)
       
lag_obs=np.zeros(11)
for i in range(0,11):
    lag_obs[i]=data['lambda'].autocorr(lag=i)
    
# varAr3=1/len(data)*(1+2*(lag_obs[0]**2+lag_obs[1]**2+lag_obs[2]**2))
# #varAr3=1/len(data)
# CI=(1/len(data)+1.96*varAr3**0.5)*np.ones(21)
# CI2=(1/len(data)-1.96*varAr3**0.5)*np.ones(21)

fig, ax = plt.subplots(1,figsize=(10,5))
ax[0].fill_between(np.arange(0,11,1),np.max(lag, axis=0),np.min(lag, axis=0), color = 'silver', label='Simple Bootstrap')
ax[0].scatter(np.arange(0,11,1),lag_obs, color='g')
ax[0].plot(np.arange(0,11,1),lag_obs, color='g', label='Observed')
ax[0].set_ylabel('Simple $\~\lambda}$', size=20)


ax[0].legend(loc='upper right')

ax[0].set_xticks(np.arange(0, 11, 1))


ax[0].set_xlabel('Lag (days)',size=15)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)


plt.savefig('errorAutocorelation.png',dpi=1000)



