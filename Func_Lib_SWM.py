# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:56:22 2021

@author: ghazal

"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

plt.style.use('seaborn-white')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm





def simple_bootstrap(Ar_residuals,n):
    # simple bootstrap of Ar model residuals
    Error=np.array(Ar_residuals)
    I= np.random.randint(low=0,high=len(Ar_residuals),size=[n,len(Ar_residuals)])
    E= Error[I]
    return E








def Exceedance_Error(Q,n, Qgage,Qmodel):
    '''


    Parameters
    ----------
    Q : np.array in shape of (n, len(data))
        stochastic streamflows.
    n: number of traces
    data : pandas DataFrame 
        input data including Qgage, Qmodel columns.
    -------
    Returns:
        diff: pandas data-frame with shape (len(data), n+5)
            Percent Error of ralizations, Exceedance , non Exceedance, rank and observatin columns.
    

    '''
    Q=Q.T
    #synthetic_frame['obs']=Qgage
    diff=pd.DataFrame(np.zeros(Q.shape))
    for i in range(0,n):
        diff[i]=100*(np.sort(Q[:,i])[::-1]-np.sort(Qgage)[::-1])/np.sort(Qgage)[::-1]
    diff['model Error']=100*(np.sort(Qmodel)[::-1]-np.sort(Qgage)[::-1])/np.sort(Qgage)[::-1]
    
    diff['obs']=np.sort(Qgage)[::-1]
    diff['Rank']=diff.reset_index().index + 1
    diff['Exeedance']=diff['Rank']/(len(diff)+1)
    diff['non_Exceedance']=1-diff['Exeedance']
    return diff


def plt_Exceedance_Error(diff,title):
    '''
    

    Parameters
    ----------
    diff : pandas data-frame with shape (len(data), n+5)
            Percent Error of ralizations, Exceedance , non Exceedance, rank and observatin columns.

    title: str plot title 
    

    '''
    
    for i in range(0,diff.shape[1]-5):
        plt.plot(diff['Exeedance'],diff[i], color='silver',linewidth=3)
    
    plt.plot(diff['Exeedance'],diff['model Error'], color='k')
    
    plt.plot([0,1],[0,0],'-g')
    plt.ylim((-100,50))
    plt.xlim([-0.01,0.99])
    plt.xticks([0.01,0.25,0.5,0.75,0.99])
    plt.xlabel('Exceedance Probability')
    plt.ylabel('Percent Error')
    plt.title(title)
    plt.grid()
    plt.plot([],[], 'silver', label='Stochastic')
    plt.legend(loc='best')
    
def Flow_Exceedance(Q,data,n):
       
   '''


    Parameters
    ----------
    Q : np.array in shape of (n, len(data))
        stochastic streamflows.
    n:  number of realizations
    data : pandas DataFrame 
        input data including Qgage, Qmodel columns.
    -------
    Returns:
        Simulation: pandas data-frame with shape (len(data), n+5)
            Sorted stochastic flow, Exceedance , non Exceedance, rank and observatin columns.
    

    '''
   Simulation=pd.DataFrame(np.zeros([len(data),n]))
   Simulation['obs']=np.sort(data['Qgage'])[::-1]
   Simulation['histmodel']=np.sort(data['Qmodel'])[::-1]
   Simulation['Rank']=Simulation.reset_index().index + 1
   Simulation['Exeedance']=Simulation['Rank']/(len(Simulation)+1)
   Simulation['non_Exceedance']=1-Simulation['Exeedance']
   synthetic_frame=pd.DataFrame(Q.T)
   for i in range(0,n):
       Simulation[i]=np.sort(synthetic_frame[i])[::-1]
    
   return Simulation

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def plt_FlowDuration(Simulation,title):
    
    '''
    

    Parameters
    ----------
    Simultion : pandas data-frame with shape (len(data), n+5)
            Sorted stochastic ralizations flows, Exceedance , non Exceedance, rank and observatin columns.

    Returns : plot of flows vs.Exceedance
    

    '''
    
    plt.figure()

    for i in range(0,Simulation.shape[1]-5):
        plt.plot(norm.ppf(Simulation['Exeedance']),Simulation[i],color='silver',lw=3)
    ax1=plt.plot(norm.ppf(Simulation['Exeedance']),Simulation['obs'], color='green',lw=2,label='Observation')
    ax2=plt.plot(norm.ppf(Simulation['Exeedance']),Simulation['model'], color='k',lw=2,label='Deterministc')
    
    plt.title(title)
    plt.ylabel('Flow')
    plt.xlabel('Exceedance')
    plt.xticks([-3,-2,-1,0,1,2,3], [0.001,0.02,0.15,0.5,0.84,0.98,0.999],rotation = 'vertical')
    plt.xlim(-3,3)
    plt.yscale('log')
    plt.plot([],[], 'silver', label='Stochastic')
    plt.legend(loc='best')
    plt.grid()
    
    
    
def Storage(r,Q,S0):
    '''
    Q: stream flow array (n,len(data))
    r: releas or yield delivered
    S0: initial storage
    '''
    
    # change Q to a vecor of 13500*1000
    s=np.zeros((Q.shape))
    s[0,:] = S0
    
    #omit t loop can I? Is it possible?
    
    for t in range(1,len(Q[:,0])):
        s[t,:]= s[t-1,:]+r-Q[t,:] 
        a = s[t,:]
        s[t,a<0] = 0
    # s[s<0] = 0
    S=np.max(s,axis=0)
    return S
    
def Storage_yield(Q, data,b):
    
   ''''
   Parameters
    ----------
    Q : np.array in shape of (n, len(data))
        stochastic streamflows.
    n: number of realizations
    data : pandas DataFrame 
        input data including Qgage, Qmodel columns.
    -------
    Returns:
     ------- 
     storageSynthetic: np.array shape (Ri,1002)
         synthetic realizations storage
         
     AnnualVolumMean: scaler 
         annual volum mean
         
         
     Ri: scaler
         average of observation annual means
     
     
    '''
    
   Synthetic=pd.DataFrame(Q.T,index=data.index)
   Synthetic['obs']=np.array(data['Qgage'])
   Synthetic['model']=np.array(data['Qmodel'])
    
   data=data.set_index('date')
   Ri=np.int(data['Qgage'].resample('y').mean().mean())
   AnnualVolumMean= data['Qgage'].resample('Y').mean().mean()*365
    
   storageSynthetic=np.zeros((Ri,b+2))
    
    
   Qs=np.array(Synthetic)
        
   for ri in range(0,Ri):
       storageSynthetic[ri,:]=Storage(ri,Qs,0)
    
   return storageSynthetic , AnnualVolumMean ,Ri


def plt_StorageYield(storageSynthetic , AnnualVolumMean, Ri,title):
    '''
    

    Parameters
    ----------
    storageSynthetic: np.array shape (Ri,1002)
         synthetic realizations storage
         
     AnnualVolumMean: scaler 
         annual volum mean
         
         
     Ri: scaler
         average of observation annual means

    Returns
    -------
    Storage/annual mean volume vs  yield/annual mean flow  plot

    '''
    
    plt.figure()
    Rt=np.linspace(1,Ri,len(storageSynthetic))

    for i in range(0,storageSynthetic.shape[1]-2):
        plt.plot(Rt/Ri,(storageSynthetic[:,i]/AnnualVolumMean),color='silver')
    
    
    plt.plot(Rt/Ri,storageSynthetic[:,storageSynthetic.shape[1]-2]/AnnualVolumMean,color= 'g')
    plt.plot(Rt/Ri,storageSynthetic[:,storageSynthetic.shape[1]-1]/AnnualVolumMean,color='k')
    
    plt.xlabel('yield / annual mean flow')
    plt.ylabel('storage / annual mean volume')
    plt.title(title)
    plt.grid()
    plt.plot([],[], 'silver', label='Stochastic')
    plt.legend(loc='best')
    
def confidenceband_p(data,Q):
    '''
    Parameters
    ----------
    
    Q : np.array with shape (n,len(data))
        n Stochastic flow realization
    data: input data as a pandas dataframe having date, Qgage,Qmodel, lambda 
        and month as columns
    Returns
    -------
    Qt95 : np.array shape(2,len(data)) 
        Qt95[0]: 0.025 quantile for each time step
        Qt95[1]: 0.975 quantile for each time step
    Qt99 : np.array shape(2,len(data)) 
        Qt99[0]: 0.005 quantile for each time step
        Qt99[1]: 0.995 quantile for each time step
    Qt50 : np.array shape(2,len(data)) 
        Qt50[0]: 0.25 quantile for each time step
        Qt50[1]: 0.75 quantile for each time step
    data: input data as a pandas dataframe having date, Qgage,Qmodel, lambda 
    month,97.5 quantile,2.5 quantile,25 quantile and 75 quantile as columns
    '''
    # confidence band generation
    Qt95=np.quantile(Q,[0.025,0.975],axis=0)
    Qt99=np.quantile(Q,[0.005,0.995],axis=0)
    Qt50=np.quantile(Q,[0.25,0.75],axis=0)
    
    data['97.5 quantile']=Qt95[1]
    data['2.5 quantile']=Qt95[0]
    data['25 quantile']=Qt50[0]
    data['75 quantile']=Qt50[1]
    
    return Qt95,Qt99,Qt50, data


def hydrograph_plt(data, start_date ,end_date):
    '''
    

    Parameters
    ----------
    data : input data as a pandas dataframe having date, Qgage,Qmodel, lambda 
    month,97.5 quantile,2.5 quantile,25 quantile and 75 quantile as columns
    start_date : str  
    plotting start date ex: '03-2010'
    end_date : str
    plotting end date ex: '03-2011'.

    Returns
    -------
    None.

    '''
    
    
    
    data=data.set_index('date')
    plt.figure(figsize=(20, 6))
    plt.grid()
    plt.yscale('log')
    plt.title(start_date+' to '+end_date+' Daily Flows')
    plt.xlabel('Date')
    plt.ylabel('Daily Flow')
    plt.fill_between(data.loc[start_date:end_date].index, data['2.5 quantile'].loc[start_date:end_date], data['97.5 quantile'].loc[start_date:end_date], color='silver',label='95 CI')
    plt.fill_between(data.loc[start_date:end_date].index, data['25 quantile'].loc[start_date:end_date], data['75 quantile'].loc[start_date:end_date], color='yellow',label='50 CI')
    plt.plot(data.loc[start_date:end_date].index, data['Qmodel'].loc[start_date:end_date],color='k',label='Model')
    plt.plot(data.loc[start_date:end_date].index, data['Qgage'].loc[start_date:end_date],color='g',label='Observation')
    #plt.plot(data.loc[start_date:end_date].index, data['QM'].loc[start_date:end_date],color='b',label='Quantile Mapped')
    plt.ylim(0,2000)
    plt.legend()
    #%%
def t_stats(Q, data):
    '''
    

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    mean_t : TYPE
        DESCRIPTION.
    std_t : TYPE
        DESCRIPTION.
    skew_t : TYPE
        DESCRIPTION.

    '''
    
    Q_R_1=data['Qgage'].loc[data['month'] == 1].drop(columns=['month'])

    Q_R_2=data['Qgage'].loc[data['month'] == 2].drop(columns=['month'])
    
    Q_R_3=data['Qgage'].loc[data['month'] == 3].drop(columns=['month'])
    
    Q_R_4=data['Qgage'].loc[data['month'] == 4].drop(columns=['month'])
    
    Q_R_5=data['Qgage'].loc[data['month'] == 5].drop(columns=['month'])
    
    Q_R_6=data['Qgage'].loc[data['month'] == 6].drop(columns=['month'])
    
    Q_R_7=data['Qgage'].loc[data['month'] == 7].drop(columns=['month'])
    
    Q_R_8=data['Qgage'].loc[data['month'] == 8].drop(columns=['month'])
    
    Q_R_9=data['Qgage'].loc[data['month'] == 9].drop(columns=['month'])
    
    Q_R_10=data['Qgage'].loc[data['month'] == 10].drop(columns=['month'])
    
    Q_R_11=data['Qgage'].loc[data['month'] == 11].drop(columns=['month'])
    
    Q_R_12=data['Qgage'].loc[data['month'] == 12].drop(columns=['month'])
    
    synthetic_frame=pd.DataFrame(Q.T).set_index(data['date'])
    synthetic_frame['month']=pd.DatetimeIndex(synthetic_frame.index).month
    synthetic_1=synthetic_frame.loc[synthetic_frame['month'] == 1].drop(columns=['month'])

    synthetic_2=synthetic_frame.loc[synthetic_frame['month'] == 2].drop(columns=['month'])
    
    synthetic_3=synthetic_frame.loc[synthetic_frame['month'] == 3].drop(columns=['month'])
    
    synthetic_4=synthetic_frame.loc[synthetic_frame['month'] == 4].drop(columns=['month'])
    
    synthetic_5=synthetic_frame.loc[synthetic_frame['month'] == 5].drop(columns=['month'])
    
    synthetic_6=synthetic_frame.loc[synthetic_frame['month'] == 6].drop(columns=['month'])
    
    synthetic_7=synthetic_frame.loc[synthetic_frame['month'] == 7].drop(columns=['month'])
    
    synthetic_8=synthetic_frame.loc[synthetic_frame['month'] == 8].drop(columns=['month'])
    
    synthetic_9=synthetic_frame.loc[synthetic_frame['month'] == 9].drop(columns=['month'])
    
    synthetic_10=synthetic_frame.loc[synthetic_frame['month'] == 10].drop(columns=['month'])
    
    synthetic_11=synthetic_frame.loc[synthetic_frame['month'] == 11].drop(columns=['month'])
    
    synthetic_12=synthetic_frame.loc[synthetic_frame['month'] == 12].drop(columns=['month'])
    
    
    obs_mean=[Q_R_1.mean(),Q_R_2.mean(),Q_R_3.mean(),Q_R_4.mean()\
              ,Q_R_5.mean(),Q_R_6.mean(),Q_R_7.mean(),Q_R_8.mean(),\
              Q_R_9.mean(),Q_R_10.mean(),Q_R_11.mean(),Q_R_12.mean()]
    obs_std=[Q_R_1.std(),Q_R_2.std(),Q_R_3.std(),Q_R_4.std()\
              ,Q_R_5.std(),Q_R_6.std(),Q_R_7.std(),Q_R_8.std(),\
              Q_R_9.std(),Q_R_10.std(),Q_R_11.std(),Q_R_12.std()]
    obs_skew=[Q_R_1.skew(),Q_R_2.skew(),Q_R_3.skew(),Q_R_4.skew()\
              ,Q_R_5.skew(),Q_R_6.skew(),Q_R_7.skew(),Q_R_8.skew(),\
              Q_R_9.skew(),Q_R_10.skew(),Q_R_11.skew(),Q_R_12.skew()]
    
    
    monthly_mean=[synthetic_1.mean(),synthetic_2.mean(),synthetic_3.mean(),synthetic_4.mean(),\
                  synthetic_5.mean(),synthetic_6.mean(),synthetic_7.mean(),synthetic_8.mean(),\
                  synthetic_9.mean(),synthetic_10.mean(),synthetic_11.mean(),synthetic_12.mean()]
    
    stdenger_monthly_std=[np.sqrt(np.mean((synthetic_1-Q_R_1.mean())**2)),np.sqrt(np.mean((synthetic_2-Q_R_2.mean())**2)),\
                          np.sqrt(np.mean((synthetic_3-Q_R_3.mean())**2)),np.sqrt((np.mean((synthetic_4-Q_R_4.mean())**2))),\
                              np.sqrt(np.mean((synthetic_5-Q_R_5.mean())**2)), np.sqrt(np.mean((synthetic_6-Q_R_6.mean())**2)),\
                                  np.sqrt(np.mean((synthetic_7-Q_R_7.mean())**2)),np.sqrt((np.mean((synthetic_8-Q_R_8.mean())**2))),\
                                  np.sqrt(np.mean((synthetic_9-Q_R_9.mean())**2)),np.sqrt(np.mean((synthetic_10-Q_R_10.mean())**2)),\
                                      np.sqrt(np.mean((synthetic_11-Q_R_11.mean())**2)),np.sqrt(np.mean((synthetic_12-Q_R_12.mean())**2))]
    
    
    
    synthetic_skew=[np.mean((synthetic_1-Q_R_1.mean())**3/Q_R_1.std()**3),np.mean((synthetic_2-Q_R_2.mean())**3/Q_R_2.std()**3),\
                    np.mean((synthetic_3-Q_R_3.mean())**3/Q_R_3.std()**3),np.mean((synthetic_4-Q_R_4.mean())**3/Q_R_4.std()**3),\
                        np.mean((synthetic_5-Q_R_5.mean())**3/Q_R_5.std()**3), \
                            np.mean((synthetic_6-Q_R_6.mean())**3/Q_R_6.std()**3),np.mean((synthetic_7-Q_R_7.mean())**3/Q_R_7.std()**3),\
                                np.mean((synthetic_8-Q_R_8.mean())**3/Q_R_8.std()**3), np.mean((synthetic_9-Q_R_9.mean())**3/Q_R_9.std()**3),\
                                    np.mean((synthetic_10-Q_R_10.mean())**3/Q_R_10.std()**3),np.mean((synthetic_11-Q_R_11.mean())**3/Q_R_11.std()**3),\
                                        np.mean((synthetic_12-Q_R_12.mean())**3/Q_R_12.std()**3)]


    mean_t=np.zeros(12)

    for i in range(0,12):
        mean_t[i]=(obs_mean[i]-monthly_mean[i].mean())/monthly_mean[i].std()
    # t stat for standard deviation
    std_t=np.zeros(12)
    
    for i in range(0,12):
        std_t[i]=(obs_std[i]-stdenger_monthly_std[i].mean())/stdenger_monthly_std[i].std()
        
    # t stat for skew
    skew_t=np.zeros(12)
    
    for i in range(0,12):
        skew_t[i]=(obs_skew[i]-synthetic_skew[i].mean())/synthetic_skew[i].std()
        
        
    return mean_t,std_t, skew_t
    
#%%
def low7day(Q,data,n):
    s=pd.DataFrame(Q.T,index=data['date'])
    data=data.set_index('date')
    low7day=np.array(s.rolling(7).mean().resample('Y').min())
    low7daysort=pd.DataFrame(np.zeros(low7day.shape))
    for i in range(0,n):
           low7daysort[i]=np.sort(low7day[:,i])[::-1]
    
    low7daysort['rank']=(low7daysort.index) + 1
    low7daysort['Exeedance']=low7daysort['rank']/(len(low7daysort)+1)
    low7daysort['non_Exceedance']=1-low7daysort['Exeedance']
    
    low7dayobs=data['Qgage'].rolling(7).mean().resample('Y').min()
    low7daymodel=data['Qmodel'].rolling(7).mean().resample('Y').min()
    low7daysort['obs']=np.sort(low7dayobs)[::-1]
    low7daysort['histmodel']=np.sort(low7daymodel)[::-1]
    l7=np.array(low7daysort)[:,0:n]
    from scipy.stats import norm
    plt.figure()
    # for i in range(0,n):
    #     plt.plot(norm.ppf(low7daysort['Exeedance']),low7daysort[i],color='silver',lw=1)
    plt.fill_between(norm.ppf(low7daysort['Exeedance']),np.min(l7,axis=1),np.max(l7,axis=1),color='silver')
    ax2=plt.plot(norm.ppf(low7daysort['Exeedance']),low7daysort['obs'], color='green',lw=2)
    ax1=plt.plot(norm.ppf(low7daysort['Exeedance']),low7daysort['histmodel'], color='k',lw=2)
    plt.title('7 Day Low Flow')
    plt.ylabel('flow')
    plt.xlabel('Exceedance Normal Inverse')
    plt.legend([ax1, ax2], ['deter-model','observation' ], loc='upper right')
    plt.grid()
    return low7daysort
#%%
def annualmax(Q,data,n):
    anual_max=np.array(pd.DataFrame(Q.T,index=data['date']).resample('Y').max())
    annualmaxsort=pd.DataFrame(np.zeros(anual_max.shape))
    for i in range(0,n):
        annualmaxsort[i]=np.sort(anual_max[:,i])[::-1]
    
    annualmaxsort['rank']=(annualmaxsort.index) + 1
    annualmaxsort['Exeedance']=annualmaxsort['rank']/(len(annualmaxsort)+1)
    annualmaxsort['non_Exceedance']=1-annualmaxsort['Exeedance']
    Data=data.set_index('date')
    annualmaxobs=Data['Qgage'].resample('Y').max()
    annualmaxmodel=Data['Qmodel'].resample('Y').max()
    annualmaxsort['obs']=np.sort(annualmaxobs)[::-1]
    annualmaxsort['histmodel']=np.sort(annualmaxmodel)[::-1]
    amax=np.array(annualmaxsort)[:,0:n]
    plt.figure()
    # for i in range(0,n):
    #     plt.plot(norm.ppf(annualmaxsort['Exeedance']),annualmaxsort[i],color='silver',lw=1)
    plt.fill_between(norm.ppf(annualmaxsort['Exeedance']),np.min(amax,axis=1),np.max(amax,axis=1),color='silver')
    ax2=plt.plot(norm.ppf(annualmaxsort['Exeedance']),annualmaxsort['obs'], color='green',lw=2)
    ax1=plt.plot(norm.ppf(annualmaxsort['Exeedance']),annualmaxsort['histmodel'], color='k',lw=2)
    plt.title('Annual Maximum')
    plt.ylabel('flow')
    plt.xlabel('Exceedance Normal Inverse')
    plt.legend([ax1, ax2], ['deter-model','observation' ], loc='upper right')
    plt.grid()
    plt.yscale('log')
    return annualmaxsort

