#test cases for morphometricity.py
#%%
import numpy as np
import pandas as pd
import itertools
import morphometricity
import matplotlib.pyplot as plt
import simulation_function
# %%
# hyper parameters 
#   N: number of subjects
#   M: number of morphological measures
#   L: number of covariates (sex, age, etc.)
#   true_morph: proportion of variance in outcome that is explained by brain morphology 
#   width: gaussian kernel bandwidth ranges from 0 to inf.
[N,M,L] = [500, 100, 2]
true_morph = 0.4
Va = true_morph*10; Ve = (1-true_morph)*10
# generate fixed effect
np.random.seed(1011)
beta = np.random.normal(loc=0, scale=1, size = L)  # fixed effect

# %%
# estimate morphometricity through 50 replicates:
# 1) linear ASM
n_sim = 50
expected_t = expected_f = average_t = average_f = observed_t = observed_f = []

for ind in range(n_sim):
    print(ind)
    # generate data
    Z = np.random.normal(0, 2, size = (N,M))
    K = np.corrcoef(Z)
    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb
    X = np.concatenate((age, sex), axis=1)
    
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*K)   # random effect
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
    y = beta0i + beta.dot(X.T) + eps

    expected_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4), standardize=True)]
    expected_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4), standardize=False)]

    average_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="average", max_iter=100, tol=10**(-4), standardize=True)]
    average_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="average", max_iter=100, tol=10**(-4), standardize=False)]

    observed_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="observed", max_iter=100, tol=10**(-4), standardize=True)]
    observed_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="observed", max_iter=100, tol=10**(-4), standardize=False)]

# %% show the distribution of estimated Va(its value should be between 0 and 10):
expected_t_va = [expected_t[i]['Morphological variance'] for i in range(n_sim)]
expected_f_va = [expected_t[i]['Morphological variance'] for i in range(n_sim)]
average_t_va = [average_t[i]['Morphological variance'] for i in range(n_sim)]
average_f_va = [average_t[i]['Morphological variance'] for i in range(n_sim)]
observed_t_va = [observed_t[i]['Morphological variance'] for i in range(n_sim)]
observed_f_va = [observed_t[i]['Morphological variance'] for i in range(n_sim)]

fig, axes = plt.subplots(nrows=2, ncols=3) 
axes = axes.flatten()

axes[0].hist(expected_t_va)       
axes[1].hist(average_t_va)
axes[2].hist(observed_t_va)  

axes[3].hist(expected_f_va)         
axes[4].hist(average_f_va)        
axes[5].hist(observed_f_va)    

plt.show()
 


# %% 2) gaussian ASM

gau_expected_t = gau_expected_f = gau_average_t = gau_average_f = gau_observed_t = gau_observed_f = []

for ind in range(n_sim):
    print(ind)
    # generate data
    Z = np.random.normal(0, 2, size = (N,M))
    K = morphometricity.gauss_similarity(Z, width=1)
    age = np.random.normal(56, 8 ,size=(N,1))
    sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb
    X = np.concatenate((age, sex), axis=1)
    
    beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*K)   # random effect
    eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
    y = beta0i + beta.dot(X.T) + eps

    gau_expected_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4), standardize=True)]
    gau_expected_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4), standardize=False)]

    gau_average_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="average", max_iter=100, tol=10**(-4), standardize=True)]
    gau_average_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="average", max_iter=100, tol=10**(-4), standardize=False)]

    gau_observed_t += [morphometricity.morph_fit(y=y, X=X, K=K, method="observed", max_iter=100, tol=10**(-4), standardize=True)]
    gau_observed_f += [morphometricity.morph_fit(y=y, X=X, K=K, method="observed", max_iter=100, tol=10**(-4), standardize=False)]


# linear ASM works fine for all 3 fisher info and the results are consistent with or without standardization
# gaussian ASM average fisher info has problem of singular matrix in the step of computing det[Va,Ve] without standardization
# gaussian ASM observed fisher info has Va > Vy in the results if without standardization