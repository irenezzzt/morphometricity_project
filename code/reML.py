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
# read in data and filter ROI relavant to thickness and Volumn
iter_csv = pd.read_csv('ukb47552.csv', iterator=True, chunksize=10000,error_bad_lines=False)
data = pd.concat ([chunk.dropna(how='all') for chunk in iter_csv] )
data.shape #502462 subjects, 2545-1 measurements 

# %%
# filter data by the field id related to "thickness" and "Volumn"
url = "https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=196"
field_id = pd.DataFrame(pd.read_html(url)[0])
# still having problem with read_html, error: lmxl is not installed (but I definitely have it installed)
rows = [s for s, x in enumerate(field_id["Description"]) if "thickness" in x or "Volume" in x]
filtered_id = field_id.loc[rows]["Field ID"].apply(str)

# %%
instance2_dat = instance3_dat = pd.DataFrame(data["eid"])

for col in data.columns[1:]:
    field, instance = col.split("-")
    if any(filtered_id == field):
        if instance == "2.0":
            instance2_dat[field] = data[col]
            if instance == "3:0":
                instance3_dat[field] = data[col] 
 

instance2_dat.dropna(axis = 0, how = "any", inplace=True)
instance3_dat.dropna(axis = 0, how = "any", inplace=True)
print(instance2_dat.shape, instance3_dat.shape) 
# Consider only complete cases, each instances have 43107 subjects with 62 

# %%
# Take a subset of instance2 first
sub_dat = instance2_dat.iloc[0:100 ]
sub_dat.to_csv("sub_dat.csv", index=True)

 
# %%

# Week of Oct 25th. Goal:

# 1.1 simluation to confirm the estimation
# 1.2 subset ~1000 to test code of EM+LME 
# 2. twist the regression model by normalizing X, y
# 3. compute on CC, save and reload to the script (memory concern?)
 
# Existing libraries doesn't allow user specify the correlation structure of random intercept

# Input of Linear mixed effect model:
# - y nx1 array: phenotype
# - X nxm array: m brain measurements in matrix, normalized by feature
# - K nxn array: anatomic similarity matrix
# - method str: "average", "expected", "observed" information to be used in ReML
# - max_iter int: maximum iteration if not converged, default 100
# - tol int: convergence threshold, default 10^(-6)

# Output (parameters to be estimated):
# - Va 
# - Ve
# - m2  estimated morphometricity, defined as Va/(Va+Ve)
# - std_err  standard error of estimated m2
# - lik_new  reML likelihood


# %%
def morph_fit(y, X, K, method, max_iter=100, tol=10**(-6)):
    N, M = X.shape
    
    Vp = np.var(y)
    
    # initialize the anatomic and residual variance
    Va = Ve = Vp/2
    V = Va * K + Ve * np.identity(n = N)
    temp = np.linalg.solve(V, X).dot( X.T.dot(np.linalg.inv(V)).dot(X) ).dot(X.T)
    # initialize the projection matrix 
    P = (np.identity(n = N) - temp).dot(np.linalg.inv(V))
    
    # initial update by EM:
    Va = ( Va**2 * y.T.dot(P).dot(K).dot(P).dot(y) + np.trace(Va*np.identity(n = N) - Va**2 * P.dot(K)) )/N
    Ve = ( Ve**2 * y.T.dot(P).dot(P).dot(y) + np.trace(Ve*np.identity(n = N) - Ve**2 * P) )/N
    
    # set values if negative
    if Va < 0: 
        Va = 10**(-6) * Vp
    if Ve < 0:
        Ve = 10**(-6) * Vp

    # update covariance and projection
    V = Va * K + Ve*np.identity(n = N)
    temp = np.linalg.solve(V,X).dot(np.linalg.inv( X.T.dot(np.linalg.inv(V)).dot(X) ) ).dot(X.T)
    P = (np.identity(n = N) - temp).dot(np.linalg.inv(V))

    log_detV = np.log(np.linalg.det(V))
    lik_old = float('inf')
    lik_new = -0.5 *log_detV - 0.5* np.log(np.linalg.det(X.T.dot(np.linalg.inv(V)).dot(X))) - 0.5*y.T.dot(P).dot(y)
   
    # ReML:
    iter = 0
    while np.abs(lik_new-lik_old) >= tol and iter < max_iter :
        iter = iter + 1
        lik_old = lik_new

        # update the first order derivative of the reML likelihood (score functions)
        Sg = -0.5 * np.trace(P.dot(K)) + 0.5 * y.T.dot(P).dot(K).dot(P).dot(y)
        Se = -0.5 * np.trace(P) + 0.5*y.T.dot(P).dot(P).dot(y)
        Score = [Sg, Se]

        # update the second order derivative (fisher information)
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
        
        # the full fisher information matrix
        Info = [[Info_gg, Info_ge], [Info_ge, Info_ee]]
        # update variance params
        temp = np.linalg.solve(Info, Score)
        Va = Va + temp[0]
        Ve = Ve + temp[1]

        # set variance values if negative
        if Va < 0: 
            Va = 10**(-6) * Vp
        if Ve < 0:
            Ve = 10**(-6) * Vp
        
        # update the covariance and projection matrix
        V = Va * K + Ve * np.identity(n = N)
        temp = np.linalg.solve(V,X).dot(np.linalg.inv( X.T.dot(np.linalg.inv(V)).dot(X) ) ).dot(X.T)
        P = (np.identity(n = N) - temp).dot(np.linalg.inv(V))

        # update the ReML likelihood if V is not NaN
        if np.isnan(np.sum(V)):
            m2 = std_err = Va = Ve = lik_new = float('NaN')
            return 'ReML algorithm does not converge'
        else:
            log_detV = np.log(np.linalg.det(V))
            lik_new = - 0.5*log_detV - 0.5*np.log(np.linalg.det(X.T.dot(np.linalg.inv(V).dot(X)))) - 0.5*y.T.dot(P).dot(y)
    # after the ReML converges, compute morphometricity
    m2 = Va/(Va+Ve) 

    # compute the final fisher info
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
    Info = [[Info_gg, Info_ge], [Info_ge, Info_ee]]
    inv_Info = np.linalg.inv(Info)
    std_err = np.sqrt( (m2/Va)**2 * ( (1-m2)**2*inv_Info[0][0]- 2*(1-m2)*m2*inv_Info[0][1] + m2**2*inv_Info[1][1]) )
    
    # diagnosis of convergence
    if iter == max_iter and abs(lik_new - lik_old)>=tol :
        res = 'ReML algorithm does not converge'
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



# y = beta0 + beta0i + sum Xi betai  + ei

# b0 = 1
# beta0i ~ N(0, ASM) (as a vector)
# betai ~ N(0, 1) i.i.d
# ei ~ N(0,2) i.i.d.

X = np.array([[1,2,3], [2,1,2], [3,5,1], [1,0,-3], [0,5,2]])
ASM = pd.DataFrame(X.T).corr() 
D, U = np.linalg.eigh(ASM)
D = np.diag(D)
if min(np.diagonal(D)) < 0 :
    D[D < 0] = 0
    K = U.dot(D).dot(U.T) 
    # reconstruct ASM if having negative eigenvalues
else:
    K=ASM

N, M = X.shape
beta = np.random.normal(loc=0, scale=10, size = M)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 100*K)
eps = np.random.normal(loc=0, scale=1.5, size = N)
y = 1 + beta0i + beta.dot(X.T) + eps

morph_fit(y=y, X=X, K=K, method="observed", max_iter=10)


# the algorithm works with simulated data


# %%
# simulation with real X:

sub_dat = pd.read_csv("sub_dat.csv")

# generate anatomic similarity matrix 
subject = sub_dat["eid"]
X = sub_dat.drop(["eid","Unnamed: 0"], axis=1)

# %%

# normalize X first
X = preprocessing.normalize(X.iloc[0:10], axis= 0)
ASM = pd.DataFrame(X.T).corr() 
# Check semi-pos definite of ASM:
D, U = np.linalg.eigh(ASM)
D = np.diag(D)
if min(np.diagonal(D)) < 0 :
    D[D < 0] = 0
    K = U.dot(D).dot(U.T) 
    # reconstruct ASM if having negative eigenvalues
else:
    K=ASM


# %%
N, M = X.shape
beta = np.random.normal(loc=0, scale=10, size = M)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = 100*K)
eps = np.random.normal(loc=0, scale=1.5, size = N)
y = 1 + beta0i + beta.dot(X.T) + eps


# %%
# phenotype data, get a list and ask jerome for help

morph_fit(y=y, X=X, K=K, method="observed", max_iter=10)
# with the real X, it always produce 0 determinant of X'V^(-1)X.  when computing the likelihood 
# %%
# phenotype data, get a list and ask jerome for help












# %%
