#test cases for morphometricity.py
#%%
import numpy as np
from numpy.core.numeric import Inf, identity
import pandas as pd
import itertools
import morphometricity
 
# %%
# hyper parameters 
#   N: number of subjects
#   M: number of morphological measures
#   L: number of covariates (sex, age, etc.)
#   true_morph: proportion of variance in outcome that is explained by brain morphology 
#   width: gaussian kernel bandwidth ranges from 0 to inf.
[N,M,L] = [500, 100, 2]
true_morph = 0.6

# %%
# simulate morphological measures Z and covariates X
np.random.seed(1011)
Z = np.random.normal(0, 2, size = (N,M)) # brain imaging
age = np.random.normal(56, 8 ,size=(N,1))
sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb
X = np.concatenate((age, sex), axis=1)

#%%
# compute ASM by gaussian kernel:

#     (1)  non-empty Z and positive width
ASM = morphometricity.gauss_similarity(Z = Z, width=0.5)
print([np.shape(ASM) == (N, N), np.all(np.linalg.eigvals(ASM) > 0)])
#%%
#     (2） non-empty Z and no width specified (default 1)
ASM = morphometricity.gauss_similarity(Z = Z)
print([np.shape(ASM) == (N, N), np.all(np.linalg.eigvals(ASM) > 0)])
#%%
#     (3) non-empty Z and negative width
ASM = morphometricity.gauss_similarity(Z = Z, width=-1)
print([np.shape(ASM) == (N, N), np.all(np.linalg.eigvals(ASM) > 0)]) 
# False, not positive semi-definite but it is allowed in morph_fit() 
#%%
#     (4) empty Z (N=0)
Z = np.random.normal(0, 2, size = (0, M)) 
ASM = morphometricity.gauss_similarity(Z = Z) # showing error for 0 dim: 'Cannot apply_along_axis when any iteration dimensions are 0'
#%%
#     (4) empty Z (M=0)
Z = np.random.normal(0, 2, size = (N, 0)) 
ASM = morphometricity.gauss_similarity(Z = Z)
print([np.shape(ASM) == (N, N), np.all(np.linalg.eigvals(ASM) > 0)]) # False, ASM is matrix of all 1 s

# %%
# compute morphometricity:
np.random.seed(1233)
K = morphometricity.gauss_similarity(Z = np.random.normal(0, 2, size = (N,M)) , width=1)
beta0i = np.random.multivariate_normal(mean = [0] * N, cov = true_morph*K)  # random effect
beta = np.random.normal(loc=0, scale=1, size = L)  # fixed effect
eps = np.random.normal(loc=0, scale=(1-true_morph)**(1/2), size = N)  # random error
y = beta0i + beta.dot(X.T) + eps # response

# gauss ASM, positive semi definite:
K = morphometricity.gauss_similarity(Z = np.random.normal(0, 2, size = (N,M)) , width=1)
res = morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4))
res['flag']+' in '+str(res['iteration'])+' iterations'


# %%
# gauss ASM, not positive semi definite:
K = morphometricity.gauss_similarity(Z = np.random.normal(0, 2, size = (N,M)) , width=-1)
res = morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4))
res['flag']+' in '+str(res['iteration'])+' iterations'

# %%
# linear ASM
Z = np.random.normal(0, 2, size = (N,M))
K = np.corrcoef(Z)
res = morphometricity.morph_fit(y=y, X=X, K=K, method="expected", max_iter=100, tol=10**(-4))
res['flag']+' in '+str(res['iteration'])+' iterations'

# %%
# linear ASM, use average Fisher Info in EM

res = morphometricity.morph_fit(y=y, X=X, K=K, method="average", max_iter=100, tol=10**(-4))
res['flag']+' in '+str(res['iteration'])+' iterations'

# %%
# linear ASM, use observed Fisher

res = morphometricity.morph_fit(y=y, X=X, K=K, method="observed", max_iter=100, tol=10**(-4))
res['flag']+'in '+str(res['iteration'])+' iterations'

# %%
# linear ASM, wrong Fisher info method specified

res = morphometricity.morph_fit(y=y, X=X, K=K, method="whatever", max_iter=100, tol=10**(-4))

