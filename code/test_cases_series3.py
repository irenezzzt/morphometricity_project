# %%
# load the script
from tkinter import W
import numpy as np
import math
from numpy import linalg
from numpy.core.numeric import Inf, identity
import pandas as pd
import csv
import itertools
import statsmodels.regression.mixed_linear_model as sm
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from random import gauss, seed
from random import choice


os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/code')
from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity

# %%

os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_backup/data')
full_dat = pd.read_csv('ses2.csv')


# %%
# From last set of simulations:
# from one sample, seems Gaussian kernel is pushing correlations to 0 (more sparse than linear kernel)
# on average, gaussian kernel and linear kernel provide similar distribution of pairwise correlations.
 
# %%

# try block of 1s and identity ASM
[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 11))
res_gau = np.ndarray(shape = (n_sim, 11))


for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z)
    ASM_gau = gauss_similarity(Z)

    ASM_sim = np.identity(n=N)

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_sim)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
 

    temp = morph_fit(y=y, X=X, K=ASM_lin, method="expected", max_iter=50)
    res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'],
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau, method="expected", max_iter=50)
    res_gau[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    
    print(i)

# %%
res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)
#%%

fig, ((ax1, ax2,ax3,ax4), ( ax5, ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4)
 

ax1.hist(res_lin[:,2], density=True)
ax1.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res_lin[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax5.hist(res_gau[:,2], density=True)
ax5.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax5.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax5.axvline(res_gau[:,2].mean(), color="blue",linestyle='-.')
ax5.set_xlim([0,1])



ax2.hist(res_lin[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('std err')


ax6.hist(res_gau[:,3], density=True)
ax6.set_title('std err')


ax3.hist(res_lin[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax3.set_title('AIC')
#ax4.set_xlim([4000,4400])
ax7.hist(res_gau[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax7.set_title('AIC')
#ax4.set_xlim([4000,4400])



ax4.hist(res_lin[:,9], density=True)
ax4.set_title('tr(S)')
ax8.hist(res_gau[:,9], density=True)
ax8.set_title('tr(S)')
fig.tight_layout()
#plt.show()
plt.savefig('sim_compare_component_identity_n100_morph80.png',dpi=150)



#%%
[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 11))
res_gau = np.ndarray(shape = (n_sim, 11))


for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z)
    ASM_gau = gauss_similarity(Z)

    ASM_sim = np.ones((N,N))

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_sim)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
 

    temp = morph_fit(y=y, X=X, K=ASM_lin, method="expected", max_iter=50)
    res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'],
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau, method="expected", max_iter=50)
    res_gau[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    
    print(i)

# %%
res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)


#%%
fig, ((ax1, ax2,ax3,ax4), ( ax5, ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4)
 

ax1.hist(res_lin[:,2], density=True)
ax1.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res_lin[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax5.hist(res_gau[:,2], density=True)
ax5.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax5.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax5.axvline(res_gau[:,2].mean(), color="blue",linestyle='-.')
ax5.set_xlim([0,1])



ax2.hist(res_lin[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('std err')


ax6.hist(res_gau[:,3], density=True)
ax6.set_title('std err')


ax3.hist(res_lin[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax3.set_title('AIC')
#ax4.set_xlim([4000,4400])
ax7.hist(res_gau[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax7.set_title('AIC')
#ax4.set_xlim([4000,4400])



ax4.hist(res_lin[:,9], density=True)
ax4.set_title('tr(S)')
ax8.hist(res_gau[:,9], density=True)
ax8.set_title('tr(S)')
fig.tight_layout()
#plt.show()
plt.savefig('sim_compare_component_ones_n100_morph80.png',dpi=150)
#%%
# when the data is generated from identity ASM, there is no difference between Va and Ve 
# hence the algorithm fails with any kernel

# when the data is generated from a block of ones ASM, all perfectly correlated
# the random effect becomes the same for all hence the estimate of Va is almost 0

# %%

# validate my guess: for any data, if we keep decrese the width, 
# will the estimated morphometricity keep increasing until 1?
[N,M,L] = [500, 100, 2]
[Va,Ve] = [2,8]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 11))
res_gau0= np.ndarray(shape = (n_sim, 11))
res_gau1 = np.ndarray(shape = (n_sim, 11))
res_gau2= np.ndarray(shape = (n_sim, 11))
res_gau3= np.ndarray(shape = (n_sim, 11))
 
# sim with width 1 analyze with width 2, 1, 1/2,... to see both direction
for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z)
    ASM_gau0 = gauss_similarity(Z, width=2)
    ASM_gau1 = gauss_similarity(Z, width=1)
    ASM_gau2 = gauss_similarity(Z, width=1/2)
    ASM_gau3 = gauss_similarity(Z, width=1/4)
    #ASM_iden = np.identity(n=N)


    ASM_sim =  ASM_lin

    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_sim)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
 

    temp = morph_fit(y=y, X=X, K=ASM_lin, method="expected", max_iter=50)
    res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'],
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau0, method="expected", max_iter=50)
    res_gau0[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    

    temp = morph_fit(y=y, X=X, K=ASM_gau1, method="expected", max_iter=50)
    res_gau1[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    
    temp = morph_fit(y=y, X=X, K=ASM_gau2, method="expected", max_iter=50)
    res_gau2[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    

    temp = morph_fit(y=y, X=X, K=ASM_gau3, method="expected", max_iter=50)
    res_gau3[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
        temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    
    
    #temp = morph_fit(y=y, X=X, K=ASM_iden, method="expected", max_iter=50)
    #res_iden[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
    #    temp['iteration'], temp['Estimated morphometricity'], 
    #    temp['Estimated standard error'], temp['Morphological variance'], 
    #    temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], 
    #   temp['BIC'] , temp['trace S'], temp['Sum of Residual']]
    # does not allow identity ASM input, singular matrix
    print(i)


# %%
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7,ax8,ax9,ax10),(ax11,ax12, ax13,ax14,ax15 )) = plt.subplots(nrows=3, ncols=5)
 

ax1.hist(res_lin[:,2], density=True)
ax1.set_title('m2 lin')
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res_lin[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res_gau0[:,2], density=True)
ax2.set_title('m2 gau 2')
ax2.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax2.axvline(res_gau0[:,2].mean(), color="blue",linestyle='-.')
ax2.set_xlim([0,1])

ax3.hist(res_gau1[:,2], density=True)
ax3.set_title('m2 gau 1')
ax3.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax3.axvline(res_gau1[:,2].mean(), color="blue",linestyle='-.')
ax3.set_xlim([0,1])


ax4.hist(res_gau2[:,2], density=True)
ax4.set_title('m2 gau 1/2')
ax4.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax4.axvline(res_gau2[:,2].mean(), color="blue",linestyle='-.')
ax4.set_xlim([0,1])

ax5.hist(res_gau3[:,2], density=True)
ax5.set_title('m2 gau 1/4')
ax5.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax5.axvline(res_gau3[:,2].mean(), color="blue",linestyle='-.')
ax5.set_xlim([0,1])

  
ax6.hist(res_lin[:,7], density=True)
ax6.set_title('AIC')
ax7.hist(res_gau0[:,7], density=True)
ax7.set_title('AIC')
 
ax8.hist(res_gau1[:,7], density=True)
ax8.set_title('AIC')
ax9.hist(res_gau2[:,7], density=True)
ax9.set_title('AIC')
ax10.hist(res_gau3[:,7], density=True)
ax10.set_title('AIC')

 


ax11.hist(res_lin[:,8], density=True)
ax11.set_title('BIC')
ax12.hist(res_gau0[:,8], density=True)
ax12.set_title('BIC')
 
ax13.hist(res_gau1[:,8], density=True)
ax13.set_title('BIC')
ax14.hist(res_gau2[:,8], density=True)
ax14.set_title('BIC')
ax15.hist(res_gau3[:,8], density=True)
ax15.set_title('BIC')

#fig.tight_layout()
fig.set_size_inches(18.5, 10.5)
#plt.show()
plt.savefig('sim_grid_gau_n100_morph20lin.png',dpi=150)

# %%
# check AIC/BIC selected kernel

AIC = np.column_stack( (res_lin[:,7], res_gau0[:,7], res_gau1[:,7], res_gau2[:,7], res_gau3[:,7]))
BIC = np.column_stack( (res_lin[:,8], res_gau0[:,8], res_gau1[:,8], res_gau2[:,8], res_gau3[:,8]))


np.argmin(AIC,axis=1)
np.argmin(BIC,axis=1)

# almost all 3
# AIC BIC fail...

 
# %%
