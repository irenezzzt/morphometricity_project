#!/usr/bin/env python
# coding: utf-8

# %%
# libraries
import numpy as np
from numpy import linalg
from numpy.core.numeric import Inf, identity
import pandas as pd
import csv
import itertools
import statsmodels.regression.mixed_linear_model as sm
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing




# %%
def compute_Score(y, P, K):
    '''
    This is a helper function to compute the score function
    Input:
        - y: nx1 array, phenotype
        - P: nxn array, projection matrix that maps y to yhat
        - K: anatomic similarity matrix
    Output:
        - Score: 2x1 array for two parameters Va(anatomical variance) and Ve(residual variance)
    '''
    Sg = -0.5 * np.trace(P.dot(K)) + 0.5 * y.T.dot(P).dot(K).dot(P).dot(y)
    Se = -0.5 * np.trace(P) + 0.5 * y.T.dot(P).dot(P).dot(y)
    return [Sg, Se]
# %%
def compute_FisherInfo(y, P, K, method):
    '''
    This is a helper function to compute the Fisher information matrix 
    Input:
        - y: nx1 array, phenotype
        - P: nxn array, projection matrix that maps y to yhat
        - K: anatomic similarity matrix
        - method: string, "average", "expected", or "observed" Fisher information
    Output:
        - Info: 2x2 array for two parameters Va(anatomical variance) and Ve(residual variance)
    '''
    if method == "average" :
        Info_gg = 0.5*y.T.dot(P).dot(K).dot(P).dot(K).dot(P).dot(y)
        Info_ge = 0.5*y.T.dot(P).dot(K).dot(P).dot(P).dot(y)
        Info_ee = 0.5*y.T.dot(P).dot(P).dot(P).dot(y)
    elif method == "expected" :
        Info_gg = 0.5*np.trace(P.dot(K).dot(P).dot(K))
        Info_ge = 0.5*np.trace(P.dot(K).dot(P))
        Info_ee = 0.5*np.trace(P.dot(P))
    elif method == "observed" : 
        Info_gg = -0.5*np.trace(P.dot(K).dot(P).dot(K)) +y.T.dot(P).dot(K).dot(P).dot(K).dot(P).dot(y)
        Info_ge = -0.5*np.trace(P.dot(K).dot(P)) + y.T.dot(P).dot(K).dot(P).dot(P).dot(y)
        Info_ee = -0.5*np.trace(P.dot(P)) + y.T.dot(P).dot(P).dot(P).dot(y)
    else:
        return 'Method for fisher information is not accepted'
    
    Info = [[Info_gg, Info_ge], [Info_ge, Info_ee]]
    return Info
# %%
def EM_update(y, X, K, Va, Ve):
    N = K.shape[0]
    # update covariance matrix V of y:
    V = Va * K + Ve * np.identity(n = N)

    # update projection matrix P:

    inv_V = np.linalg.inv(V)
    temp = np.linalg.solve(V, X).dot( np.linalg.solve(X.T.dot(inv_V).dot(X), X.T) )
    P = (np.identity(n = N) - temp).dot(inv_V)  

    # update logliklihood:
    if np.isnan(np.sum(V)):
        m2 = std_err = Va = Ve = lik_new = float('NaN')
        print('invalid covariance estimates, NaN ')
    else:
        E = np.linalg.eigvals(V)
        log_detV=sum(np.log(E + 2**(-52)))
        lik_new = - 0.5*log_detV - 0.5*np.log(np.linalg.det(X.T.dot(inv_V.dot(X)))) - 0.5*y.T.dot(P).dot(y)

    return [V, P, lik_new]
# %%
def morph_fit(y, X, K, method, max_iter=100, tol=10**(-4)):
    '''
    The function fit the linear mixed effect model (1) by EM algorithm and estimate the morphometricity together with its standard error
        y = Xb + a + e    (1)
    where Cov(a) = Va * K, e ~ N(0, Ve) i.i.d.

    Input of Linear mixed effect model:
        - y nx1 array: phenotype
        - X nxl array: l covariates such as age, sex, site etc.  
        - K nxn array: anatomic similarity matrix, positive semi-definite.
        - method str: "average", "expected", "observed" information to be used in ReML
        - max_iter int: maximum iteration if not converged, default 100
        - tol int: convergence threshold, default 10^(-6)

    Output (parameters to be estimated):
        - beta estimated fixed effect, standard error, and the associated hypothesis test statistic, p-value, significance (Add later)
        - Va variance explained by the ASM (random intercept)
        - Ve residual variance
        - m2  estimated morphometricity, defined as Va/(Va+Ve)
        - std_err  standard error of estimated m2
        - lik_new  reML likelihood
    '''

    N = X.shape[0]

    # standardize X, y to have mean 0, var 1 for each column 
    #X = (X - np.mean(X, axis=0))/np.std(X, axis= 0)
    #y = (y - y.mean())/(y.std())

    # reconstruct ASM if having negative eigenvalues
    D, U = np.linalg.eigh(K)
    D = np.diag(D)
    if min(np.diagonal(D)) < 0 :
        D[D < 0] = 0
        K = U.dot(D).dot(U.T) 
    else:
        K=K

    # initialize the anatomic variance, residual variance, and projection matrix.
    Vp = np.var(y)
    Va = Ve = 1/2*Vp
    V, P, lik= EM_update(y = y, X = X, K = K, Va = Va, Ve = Ve)

    # initial update of [Va, Ve] by EM:
    Va = ( Va**2 * y.T.dot(P).dot(K).dot(P).dot(y) + np.trace(Va*np.identity(n = N) - Va**2 * P.dot(K)) )/N
    Ve = ( Ve**2 * y.T.dot(P).dot(P).dot(y) + np.trace(Ve*np.identity(n = N) - Ve**2 * P) )/N
    
    # set values if negative and normalize  
    T = np.array([Va, Ve])
    T[T < 0] = 10e-6 * Vp
    # Va, Ve = T/sum(T)
    Va, Ve = T

    # initial update covariance and projection
    lik_old = float('inf')
    V, P, lik_new = EM_update(y = y, X = X, K = K, Va = Va, Ve = Ve)
    
    # ReML interative update by EM:
    iter = 0
    while np.abs(lik_new-lik_old) >= tol and iter < max_iter :
        iter = iter + 1
        lik_old = lik_new

        # compute the first order derivative of likelihood (score functions) and second derivative (fisher information)
        Score = compute_Score(y = y, P = P, K = K)
        Info = compute_FisherInfo(y = y, P = P, K = K, method = method)
        
        # update Va, Ve
        T = np.array([Va,Ve]) + np.linalg.solve(Info, Score)
        # set variance values to be 1e-6*Vp if negative (Vp=1 for normalized y) and normalize
        T[T < 0] = 1e-6  
        if method == "observed":
             Va, Ve = T/sum(T)
        else:
            Va, Ve = T
        # update covariance V, projection matrix P and likelihood  
        V, P, lik_new = EM_update(y = y, X = X, K = K, Va = Va, Ve = Ve)
    
    # after the ReML converges, compute morphometricity and standard error
    m2 = Va/(Va+Ve) 
    Info = compute_FisherInfo(y = y, P = P, K = K, method = method)

    inv_Info = np.linalg.inv(Info)
    std_err = np.sqrt( (m2/Va)**2 * (1-m2)**2 * inv_Info[0][0] - 2 * (1-m2) * m2 * inv_Info[0][1] + m2**2 * inv_Info[1][1] ) 
    # m2 = Va/(Va+Ve) 

    # diagnosis of convergence
    if iter == max_iter and abs(lik_new - lik_old)>=tol :
        res = 'ReML algorithm did not converge'
    else:
        res = 'ReML algorithm has converged'
    # model selection cretiria
   
    S = np.identity(n = N) - Ve * P
    RSS = Ve * (N-np.trace(S))
    AIC = N*np.log(RSS) + 2*np.trace(S)
    BIC = N*np.log(RSS) + np.log(N)*np.trace(S)
    
    return { 
        'flag': res,
        'iteration': iter,
        'Estimated morphometricity': m2, 
        'Estimated standard error': std_err,
        'Morphological variance': Va,
        'Residual variance': Ve,
        'ReML likelihood': lik_new,
        'AIC':AIC,
        'BIC':BIC
        }





def gauss_ker(vec,mat, S2):
    '''
    This is a helper function to compute the similarity between an individual and the rest
    Input:
        - mat: matrix of size n x M, n individuals, M imaging measures
        - vec: vector of size M, observation for one specific individual 
        - S2: vector of size M, pre-calculated variance for each imaging measures
    Output:
        - r: vector of size N, similarity between zi and each individual in Z
    '''
    n, M = mat.shape 
    r = np.exp(- np.divide( (mat-vec)**2, S2*M).sum(axis=1))
    
    return r

def gauss_similarity(Z, width=1):
    '''
    This is the function is to compute the ASM matrix by default gauss kernel 
    Input:
        - Z, matrix of size n x M, each row [i, ] is an individual Xi
    Output:
        - R, matrix of size n x n, 
            each entry [i,j] is a real number, similarity between Xi and Xj
    '''
    n, M = Z.shape
    Z_copy = Z
    S2 = Z.std(axis=0)**2
    R = np.apply_along_axis(gauss_ker, axis=1, arr=Z, mat=Z_copy, S2=(S2*width))
    return R
    
 

# %%
