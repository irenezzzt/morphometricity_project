
# %%
# load the script
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
from random import seed
from random import choice


os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/code')
from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity

# %%

os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_backup/data')
full_dat = pd.read_csv('ses2.csv')

# %%
# check the estimates from linear and gaussian kernel

n = 100
subset = random.sample(sequence, n)
instance2_dat = full_dat.iloc[subset]
y = instance2_dat['age_at_recruitment']
X = instance2_dat['sex'].to_numpy().reshape(-1,1)

Z = instance2_dat.drop(['eid','age_at_recruitment','sex'], axis=1)
Z = (Z - np.mean(Z, axis=0))/np.std(Z, axis= 0)

ASM_gau = gauss_similarity(Z)
ASM_lin = np.corrcoef(Z)

#%%
def outlier(cor_matrix, n_sd):
    N = cor_matrix.shape[0]
    npair = math.comb(N,2)
    cor_coef = np.matrix.flatten(np.triu(cor_matrix, k=1))
    cor_coef = cor_coef[cor_coef!=0]
    mu = cor_coef.mean()
    sd = cor_coef.std()

    out =  ((cor_coef > mu + n_sd * sd) * 1).sum() + ((cor_coef < mu - n_sd * sd) * 1).sum()
    out_prop = out/npair
    return out_prop

#%%
print([outlier(ASM_lin, 1), outlier(ASM_gau, 1)])
print([outlier(ASM_lin, 2), outlier(ASM_gau, 2)])

# from one sample, seems Gaussian kernel is pushing correlations to 0 (more sparse than linear kernel)


# %%

sequence = [i for i in range(full_dat.shape[0])]
n_sim = 500
res = np.ndarray(shape = (n_sim, 4))

# %%
for i in range(n_sim):
    np.random.seed(i*13+7)
    subset = random.sample(sequence, 1000)
    instance2_dat = full_dat.iloc[subset]
    y = instance2_dat['age_at_recruitment']
    X = instance2_dat['sex'].to_numpy().reshape(-1,1)

    Z = instance2_dat.drop(['eid','age_at_recruitment','sex'], axis=1)
    Z = (Z - np.mean(Z, axis=0))/np.std(Z, axis= 0)
    
    ASM_gau = gauss_similarity(Z)
    ASM_lin = np.corrcoef(Z)
    print(i)
    res[i] = [outlier(ASM_lin,1), outlier(ASM_gau,1), outlier(ASM_lin,2), outlier(ASM_gau,2) ]

# %%
res.mean(axis=0)
# on average, gaussian kernel and linear kernel provide similar distribution of pairwise correlations.





#%% altering Gaussian kernel width then apply both the proposed linear and gaussian kernel
[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 9))
res_gau = np.ndarray(shape = (n_sim, 9))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z)
    ASM_gau = gauss_similarity(Z)

    ASM_sim = gauss_similarity(Z, width=1/2)

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
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'],temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau, method="expected", max_iter=50)
    res_gau[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]
    
    print(i)

res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)
# %%
(res_lin[:,7] < res_gau[:,7]).mean() # =0 : gaussian is always less than linear, preferrable but wrong
(res_lin[:,8] < res_gau[:,8]).mean()

# if the actual similarity is gaussian kernel half width 
# then linear is better in BIC (99% of the time) and gaussian is better in AIC (100% of the time)

#%%
res = res_lin
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
ax4.set_xlim([4000,4400])


ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
ax5.set_xlim([4100,5200])

fig.tight_layout()
#plt.show()
plt.savefig('sim_width0.5_lin_n500_morph80.png',dpi=150)

#%%
# plot the distribution of estimated morphometricity
res = res_gau
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
ax4.set_xlim([4000,4400])

ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
ax5.set_xlim([4100,5200])

fig.tight_layout()
#plt.show()
plt.savefig('sim_width0.5_gau_n500_morph80.png',dpi=150)








#%% altering Gaussian kernel width then apply both the proposed linear and gaussian kernel
[N,M,L] = [500, 100, 2]
[Va,Ve] = [8,2]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 9))
res_gau = np.ndarray(shape = (n_sim, 9))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z)
    ASM_gau = gauss_similarity(Z)

    ASM_sim = gauss_similarity(Z, width=2)

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
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'],temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau, method="expected", max_iter=50)
    res_gau[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]
    
    print(i)

res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)
# %%
(res_lin[:,7] < res_gau[:,7]).mean() # =0 : gaussian is always less than linear, preferrable but wrong
(res_lin[:,8] < res_gau[:,8]).mean()

# if the actual similarity is gaussian kernel double width 
# then gaussian kernel is always better than linear in AIC(100%), BIC(97%)

#%%
res = res_lin
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
#ax4.set_xlim([4000,4400])


ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
#ax5.set_xlim([4100,5200])

fig.tight_layout()
#plt.show()
plt.savefig('sim_width2_lin_n500_morph80.png',dpi=150)

#%%
# plot the distribution of estimated morphometricity
res = res_gau
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
ax0.hist(res[:,1], density=True)
ax0.set_title('iterations until converge')

ax1.hist(res[:,2], density=True)
ax1.set_title('estimated morphometricity')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('estimated standard error')


ax3.hist(res[:,6], density=True)
ax3.set_title('restricted log likelihood')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
#ax4.set_xlim([4000,4400])

ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
#ax5.set_xlim([4100,5200])

fig.tight_layout()
#plt.show()
plt.savefig('sim_width2_gau_n500_morph80.png',dpi=150)




# %%
# Check some of the AIC BIC component in simulations... 
# at least we know BIC is not always favor the more complex kernel...
# Simulate with linear ASM, model with both linear and gaussian ASM:

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
    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_lin)
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
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]
    
    print(i)

res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)




# %%
(res_lin[:,7] < res_gau[:,7]).mean() 
(res_lin[:,8] < res_gau[:,8]).mean()

#%%
res = res_lin
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
 

ax1.hist(res[:,2], density=True)
ax1.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('std err')


ax3.hist(res[:,5], density=True)
ax3.set_title('Ve')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
#ax4.set_xlim([4000,4400])


ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
#ax5.set_xlim([4100,5200])



ax6.hist(res[:,9], density=True)
ax6.set_title('tr(S)')
fig.tight_layout()
#plt.show()
plt.savefig('sim_compare_component_lin_n100_morph80.png',dpi=150)

# %%
res = res_gau
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
 

ax1.hist(res[:,2], density=True)
ax1.set_title('m2')
#txt = '='.join( ['mean',str( round(res[:,2].mean(), 4) )] )
#ax1.text(0.8,2,txt)
ax1.axvline(Va/(Va+Ve),color='red', linestyle='--')
ax1.axvline(res[:,2].mean(), color="blue",linestyle='-.')
ax1.set_xlim([0,1])


ax2.hist(res[:,3], density=True)
#txt = '='.join( ['mean se',str( round(res[:,3].mean(), 4) )] )
#ax2.text(1,1,txt)
ax2.set_title('std err')


ax3.hist(res[:,5], density=True)
ax3.set_title('Ve')

ax4.hist(res[:,7], density=True)
#txt = '='.join(['mean AIC', str( round(res[:,7].mean(), 4) )])
#ax4.text(-11100,0.001,txt)
ax4.set_title('AIC')
#ax4.set_xlim([4000,4400])


ax5.hist(res[:,8], density=True)
#txt = '='.join(['mean BIC', str( round(res[:,8].mean(), 4) )])
#ax5.text(-9100,0.001,txt)
ax5.set_title('BIC')
#ax5.set_xlim([4100,5200])



ax6.hist(res[:,9], density=True)
ax6.set_title('tr(S)')
fig.tight_layout()
#plt.show()
plt.savefig('sim_compare_component_gau_n100_morph80.png',dpi=150)

# %%

trace = res[:,9]
print([trace.mean(), trace.std()])

RSS = res[:,10]
print([RSS.mean(), RSS.std()])
# %%
# try other morphometricity...

[N,M,L] = [500, 100, 2]
[Va,Ve] = [2,8]
beta = np.random.normal(loc=0, scale=1, size = L)
n_sim = 100
res_lin = np.ndarray(shape = (n_sim, 11))
res_gau = np.ndarray(shape = (n_sim, 11))

for i in range(n_sim):
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM_lin = np.corrcoef(Z) 
    ASM_gau = gauss_similarity(Z)
    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_lin)
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
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]
    
    print(i)

res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)


# %%

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
plt.savefig('sim_compare_component_gau_n100_morph20.png',dpi=150)

# %%

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
    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM_lin)
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
        temp['BIC'], temp['trace S'], temp['Sum of Residual'] ]
    
    print(i)

res_lin = res_lin[res_lin[:,0]==1] # subset those who converged  
res_lin = res_lin[np.isnan(res_lin[:,3])==False]
print(res_lin.shape[0]/n_sim)

res_gau = res_gau[res_gau[:,0]==1] # subset those who converged  
res_gau = res_gau[np.isnan(res_gau[:,3])==False]
print(res_gau.shape[0]/n_sim)


# %%

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
plt.savefig('sim_compare_component_gau_n100_morph80.png',dpi=150)

# %%
