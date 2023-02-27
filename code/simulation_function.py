# functions for simulation
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
import os

os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/code')
from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity

def sim(N, M, L, m2, n_sim, kernel = "linear", fisher="expected"):
    Va = m2*10; Ve = (1-m2)*10
    res_lin = np.ndarray(shape = (n_sim, 9))
    res_gau0 = np.ndarray(shape = (n_sim, 9))
    res_gau1 = np.ndarray(shape = (n_sim, 9))
    res_gau2 = np.ndarray(shape = (n_sim, 9))
    res_gau3 = np.ndarray(shape = (n_sim, 9))

    beta = np.random.normal(loc=0, scale=1, size = L)  # fixed effect

    if kernel == "linear":
        for i in range(n_sim):
            np.random.seed(i*13+7)
            Z = np.random.normal(0, 2, size = (N,M)) # brain imaging 
            ASM = np.corrcoef(Z) 

            age = np.random.normal(56, 8 ,size=(N,1))
            sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

            X = np.concatenate((age, sex), axis=1) # covariates
            beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)  # random effect
            eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
            y = beta0i + beta.dot(X.T) + eps # response            
            
            ASM_lin = ASM 
            ASM_gau0 = gauss_similarity(Z, width=2)
            ASM_gau1 = gauss_similarity(Z, width=1)
            ASM_gau2 = gauss_similarity(Z, width=1/2)
            ASM_gau3 = gauss_similarity(Z, width=1/4)

            temp = morph_fit(y=y, X=X, K=ASM_lin, method=fisher, max_iter=100)   
            res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]    

            temp = morph_fit(y=y, X=X, K=ASM_gau0, method=fisher, max_iter=100)   
            res_gau0[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
        
            temp = morph_fit(y=y, X=X, K=ASM_gau1, method=fisher, max_iter=100)     
            res_gau1[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
       
            temp = morph_fit(y=y, X=X, K=ASM_gau2, method=fisher, max_iter=100)   
            res_gau2[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
          
            temp = morph_fit(y=y, X=X, K=ASM_gau3, method=fisher, max_iter=100)   
            res_gau3[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
             
        return{'res_lin': res_lin, 'res_gau0': res_gau0, 'res_gau1':res_gau1, 'res_gau2':res_gau2, 'res_gau3':res_gau3}
    
    elif kernel =="gaussian":
        for i in range(n_sim):
            np.random.seed(i*13+7)
            Z = np.random.normal(0, 2, size = (N,M)) # brain imaging 
            ASM = gauss_similarity(Z, width=1)

            age = np.random.normal(56, 8 ,size=(N,1))
            sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

            X = np.concatenate((age, sex), axis=1) # covariates
            beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)  # random effect
            eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
            y = beta0i + beta.dot(X.T) + eps # response
             
            ASM_lin = np.corrcoef(Z)
            ASM_gau0 = gauss_similarity(Z, width=1/2)
            ASM_gau1 = gauss_similarity(Z, width=1)
            ASM_gau2 = gauss_similarity(Z, width=2)
            ASM_gau3 = gauss_similarity(Z, width=4)

            temp = morph_fit(y=y, X=X, K=ASM_lin, method=fisher, max_iter=100)   
            res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]    
        

            temp = morph_fit(y=y, X=X, K=ASM_gau0, method=fisher, max_iter=100)   
            res_gau0[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
        
            temp = morph_fit(y=y, X=X, K=ASM_gau1, method=fisher, max_iter=100)     
            res_gau1[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
       
            temp = morph_fit(y=y, X=X, K=ASM_gau2, method=fisher, max_iter=100)   
            res_gau2[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
          
            temp = morph_fit(y=y, X=X, K=ASM_gau3, method=fisher, max_iter=100)   
            res_gau3[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'], 
                temp['Estimated standard error'], temp['Morphological variance'], 
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]
             
        #res_lin = res_lin[res_lin[:,0]==1] 
        #res_lin = res_lin[np.isnan(res_lin[:,3])==False] # subset of the iters converge and with positive variance estimates
        
        #res_gau0 = res_gau0[res_gau0[:,0]==1] 
        #res_gau0 = res_gau0[np.isnan(res_gau0[:,3])==False] # subset of the iters converge and with positive variance estimates

        #res_gau1 = res_gau1[res_gau1[:,0]==1] 
        #res_gau1 = res_gau1[np.isnan(res_gau1[:,3])==False] # subset of the iters converge and with positive variance estimates

        #res_gau2 = res_gau2[res_gau2[:,0]==1] 
        #res_gau2 = res_gau2[np.isnan(res_gau2[:,3])==False] # subset of the iters converge and with positive variance estimates
            
        #res_gau3 = res_gau3[res_gau3[:,0]==1] 
        #res_gau3 = res_gau3[np.isnan(res_gau3[:,3])==False] # subset of the iters converge and with positive variance estimates
            
        return{'res_lin': res_lin, 'res_gau0': res_gau0, 'res_gau1':res_gau1, 'res_gau2':res_gau2, 'res_gau3':res_gau3}

    else:
        return['Input kernel is not supported']

