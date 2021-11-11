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
    Se = -0.5 * np.trace(P) + 0.5*y.T.dot(P).dot(P).dot(y)
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
    if method =="average" :
        Info_gg = 0.5*y.T.dot(P).dot(K).dot(P).dot(K).dot(P).dot(y)
        Info_ee = 0.5*y.T.dot(P).dot(K).dot(P).dot(P).dot(y)
        Info_ge = 0.5*y.T.dot(P).dot(P).dot(P).dot(y)
    elif method == "expected" :
        Info_ge = 0.5*np.trace(P.dot(K).dot(P).dot(K))
        Info_ee = 0.5*np.trace(P.dot(K).dot(P))
        Info_ge = 0.5*np.trace(P.dot(P))
    elif method == "observed" : 
        Info_gg = -0.5*np.trace(P.dot(K).dot(P).dot(K)) +y.T.dot(P).dot(K).dot(P).dot(K).dot(P).dot(y)
        Info_ee = -0.5*np.trace(P.dot(K).dot(P)) + y.T.dot(P).dot(K).dot(P).dot(P).dot(y)
        Info_ge = -0.5*np.trace(P.dot(P)) + y.T.dot(P).dot(P).dot(P).dot(y)
    else:
        return 'Method for fisher information is not accepted'
    
    Info = [[Info_gg, Info_ge], [Info_ge, Info_ee]]
    return Info
# %%
def EM_update(y, K, Va, Ve):
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
        lik_new = - 0.5*log_detV - 0.5*np.log(np.linalg.det(X.T.dot(np.linalg.inv(V).dot(X)))) - 0.5*y.T.dot(P).dot(y)

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
        - Va variance explained by the ASM (random intercept)
        - Ve residual variance
        - m2  estimated morphometricity, defined as Va/(Va+Ve)
        - std_err  standard error of estimated m2
        - lik_new  reML likelihood
    '''

    N = X.shape[0]

    # standardize X, y to have mean 0, var 1 for each column 
    X = (X - np.mean(X, axis=0))/np.std(X, axis= 0)
    y = (y - y.mean())/(y.std())

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
    V, P, lik= EM_update(y = y, K = K, Va = Va, Ve = Ve)

    # initial update of [Va, Ve] by EM:
    Va = ( Va**2 * y.T.dot(P).dot(K).dot(P).dot(y) + np.trace(Va*np.identity(n = N) - Va**2 * P.dot(K)) )/N
    Ve = ( Ve**2 * y.T.dot(P).dot(P).dot(y) + np.trace(Ve*np.identity(n = N) - Ve**2 * P) )/N
    
    # set values if negative and normalize  
    T = np.array([Va, Ve])
    T[T < 0] = 10e-6 * Vp
    Va, Ve = T/sum(T)

    # initial update covariance and projection
    lik_old = float('inf')
    V, P, lik_new = EM_update(y = y, K = K, Va = Va, Ve = Ve)
    
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
        Va, Ve = T/sum(T)

        # update covariance V, projection matrix P and likelihood  
        V, P, lik_new = EM_update(y = y, K = K, Va = Va, Ve = Ve)
    
    # after the ReML converges, compute morphometricity and standard error
    m2 = Va/(Va+Ve) 
    Info = compute_FisherInfo(y = y, P = P, K = K, method = method)

    inv_Info = np.linalg.inv(Info)
    std_err = np.sqrt( (m2/Va)**2 * (1-m2)**2 * inv_Info[0][0] - 2 * (1-m2) * m2 * inv_Info[0][1] + m2**2 * inv_Info[1][1] ) 
    print(std_err)

    # diagnosis of convergence
    if iter == max_iter and abs(lik_new - lik_old)>=tol :
        res = 'ReML algorithm did not converge'
    else:
        res = 'ReML algorithm has converged'
    
    print(res)
    return { 
        'iteration': iter,
        'Estimated morphometricity': m2, 
        'Estimated standard error': std_err,
        'Morphological variance': Va,
        'Residual variance': Ve,
        'ReML likelihood': lik_new}




# %%
# test cases with simulated data:

# y = beta0 + beta0i + sum Xij betaj  + ei

# b0 = 1
# beta0i ~ N(0, ASM) (as a vector)
# betai ~ N(0, 1) i.i.d
# ei ~ N(0,2) i.i.d.

Z = np.array([[1,2,3], [2,1,2], [3,5,1], [1,0,-3], [0,5,2]]) #N=5, M=3
ASM = pd.DataFrame(Z.T).corr() 

X = np.array([[0,1,20],[1,1,30],[1,0,40],[0,1,35], [0,0,25]])

N, L = X.shape
beta = np.random.normal(loc=0, scale=10, size = L)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 100*ASM)
eps = np.random.normal(loc=0, scale=1.5, size = N)
y = 1 + beta0i + beta.dot(X.T) + eps


 
# %%
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=10)
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=20)
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=50)
morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=100)



# %%
# simulation with real Z:

sub_dat = pd.read_csv("sub_dat.csv")

# generate anatomic similarity matrix 
subject = sub_dat["eid"]
Z = sub_dat.drop(["eid","Unnamed: 0"], axis=1)
N = Z.shape[0]
# Z.shape = (100, 124) N=100, M=124
ASM = pd.DataFrame(Z.T).corr() 
ASM = ASM.values 

# %%
sex = np.random.binomial(n = 1, p=0.4, size = N)
age = np.random.normal(loc = 45, scale = 5, size = N)
X = np.array([sex, age]).T
# normalize X
N, L = X.shape

beta = np.random.normal(loc=0, scale=10, size = L)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 100*ASM)
eps = np.random.normal(loc=0, scale=1.5, size = N)
y = 1 + beta0i + beta.dot(X.T) + eps

X = (X - np.mean(X, axis=0))/np.std(X, axis= 0)
y = (y - y.mean())/(y.std())
# %%
# phenotype data, get a list and ask jerome for help

morph_fit(y=y, X=X, K=ASM, method="observed", max_iter=300)
# new error message: 
# if np.isnan(np.sum(V)):
# f"The truth value of a {type(self).__name__} is ambiguous. "
# The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
# %%