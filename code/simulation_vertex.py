
# does aggregated version really give higher morphometricity?
from zipfile import ZIP_LZMA
import numpy as np
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
from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity



[N,M,L] = [500, 100, 2]
[Va,Ve] = [5,5]
n_sim = 500
res_lin = np.ndarray(shape = (n_sim, 9))
res_gau = np.ndarray(shape = (n_sim, 9))
res_noise1 = np.ndarray(shape = (n_sim, 9))
res_noise2 = np.ndarray(shape = (n_sim, 9))
res_noise4 = np.ndarray(shape = (n_sim, 9))
# adding more brain imaging measurements... 


for i in range(n_sim):
    print(i)
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))   
    ASM = np.corrcoef(Z)
    # generate some new Z with noise (so the similarity level between individuals reduced.)
    Z1 = Z + np.random.normal(0,1, size = (N,M))
    Z2 = Z + np.random.normal(0,2, size = (N,M))
    Z4 = Z + np.random.normal(0,4, size = (N,M))

    ASM_gau = gauss_similarity(Z, width=1)
    ASM_n1 = np.corrcoef(Z1)
    ASM_n2 = np.corrcoef(Z2)
    ASM_n4 = np.corrcoef(Z4)



    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)


    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
     
    temp = morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=200)
    res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_gau, method="observed", max_iter=200)
    res_gau[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_n1, method="observed", max_iter=200)
    res_noise1[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_n2, method="observed", max_iter=200)
    res_noise2[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_n4, method="observed", max_iter=200)
    res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]
 
    
# When the noise has same covariance structure as Z, the estimated morphometricity is inflated by too much
# otherwise, more noise == more inflation in estimated morphometricity
# same as applying gaussian kernel when the data generating function is from linear kernel

# basically, this method only applies when you have strong confidence/prior knowledge 
#   on the data generating model and the covariance structure.

# show in theoretical level: if adding many not significant variables[]
# try something else::



[N,M,L,ngroup] = [500, 1000, 2, 100]
[Va,Ve] = [5,5]
n_sim = 100
res_ori = np.ndarray(shape = (n_sim, 9))
res_agg = np.ndarray(shape = (n_sim, 9))

for i in range(n_sim):
    print(i)
    np.random.seed(i*13+7)
    Z = np.random.normal(0, 2, size = (N,M))
    ASM = np.corrcoef(Z)
    # generate some new Z with noise (so the similarity level between individuals reduced.)
    group  = list(map(str, [*range(ngroup)]*(int(M/ngroup)))) 
    random.shuffle(group)
    group = np.array(group)
    temp = np.ndarray(shape=(N,))
    for gg in np.unique(group):
        temp= np.vstack((temp, np.sum(Z[:,group==gg], axis=1)))   
  
    Zz = np.transpose(temp)
    ASM_agg = np.corrcoef(Zz)


    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

    X = np.concatenate((age, sex), axis=1)

    beta = np.random.normal(loc=0, scale=1, size = L)
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)
    y = beta0i + beta.dot(X.T) + eps
     
    temp = morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=200)
    res_ori[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    temp = morph_fit(y=y, X=X, K=ASM_agg, method="observed", max_iter=200)
    res_agg[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
        temp['iteration'], temp['Estimated morphometricity'], 
        temp['Estimated standard error'], temp['Morphological variance'], 
        temp['Residual variance'], temp['ReML likelihood'], temp['AIC'], temp['BIC'] ]

    
# as expected, if the data is generated from the vertex level, and we are using summary level data to fit the LME,
# it will under-estimate the morphometricity (to a degree depending on the aggregation level. Now 1000 -> 100)
# vice versa.

# then why the BLUP random effect model cannot achieve the same level of morphometricity?
# did they remove some of the vertex?    
