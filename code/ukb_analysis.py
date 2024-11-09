# %%
# analyzing the ukb data for 7 traits
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

os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity/data')


from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity
data = pd.read_csv("ses2.csv")
# %% 
# age at recruitement
age_data = data.drop(columns=['eid','20191-0.0', '20544-0.2', '20544-0.12','20544-0.14','21003-2.0'])
age_data = age_data.dropna(axis=0, how="any")
age_data.shape

# y for phenotype: age at recruitment, 
# x for covariates sex (other analyses will include age) 
# z for brain morphological measures

y=age_data['age_at_recruitment'].to_numpy()
x=age_data['sex'].to_numpy().reshape(-1,1)
z=age_data.drop(columns=['age_at_recruitment','sex']) 
# 627 subjects, 125 image features


# generate ASMs from z with different kernels: 
# 1. linear 
# 2. standard Gaussian 
# 3. Gaussian bandwidth 0.5 
# 4. Gaussian bandwidth 0.25
# (not including larger bandwidth because the simulation results shows that smaller bandwidth tends to give higher m2)
#   
K = np.corrcoef(z) 
r1=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1)
r2=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/2)
r3=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/4)
r4=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

# %%
age_res = np.array([ [r1['Estimated morphometricity'], r1['Estimated standard error'], r1['AIC']],
                     [r2['Estimated morphometricity'], r2['Estimated standard error'], r2['AIC']],
                     [r3['Estimated morphometricity'], r3['Estimated standard error'], r3['AIC']],
                     [r4['Estimated morphometricity'], r4['Estimated standard error'], r4['AIC']]])
# %% 
# age at instance 2, repeat the same procedure
age_data = data.drop(columns=['eid','age_at_recruitment','20191-0.0', '20544-0.2', '20544-0.12','20544-0.14'])
age_data = age_data.dropna(axis=0, how="any")
age_data.shape

y=age_data['21003-2.0'].to_numpy()
x=age_data['sex'].to_numpy().reshape(-1,1)
z=age_data.drop(columns=['21003-2.0','sex']) 
# 627 subjects, 125 image features
K = np.corrcoef(z) 
r1=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1)
r2=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/2)
r3=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/4)
r4=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

age_ses2_res = np.array([ [r1['Estimated morphometricity'], r1['AIC']],
                     [r2['Estimated morphometricity'], r2['AIC']],
                     [r3['Estimated morphometricity'], r3['AIC']],
                     [r4['Estimated morphometricity'], r4['AIC']]])


# %% 
# fluid intelligence
IQ_data = data.drop(columns=['eid','20544-0.2', '20544-0.12','20544-0.14','21003-2.0'])
IQ_data = IQ_data.dropna(axis=0, how="any")
IQ_data.shape

y=IQ_data['20191-0.0'].to_numpy()
x=IQ_data[['age_at_recruitment','sex']].to_numpy()
z=IQ_data.drop(columns=['age_at_recruitment','sex','20191-0.0']) 
# 343 subjects, 125 image features

K = np.corrcoef(z) 
r1=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1)
r2=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/2)
r3=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/4)
r4=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)


IQ_res = np.array([ [r1['Estimated morphometricity'], r1['AIC']],
                     [r2['Estimated morphometricity'], r2['AIC']],
                     [r3['Estimated morphometricity'], r3['AIC']],
                     [r4['Estimated morphometricity'], r4['AIC']]])


# %% 
# schizophrenia
scz_data = data.drop(columns=['eid', '20544-0.12','20544-0.14','21003-2.0', '20191-0.0'])
scz_data = scz_data.dropna(axis=0, how="any")
scz_data.shape

y=scz_data['20544-0.2'].to_numpy()
x=scz_data[['age_at_recruitment','sex']].to_numpy()
z=scz_data.drop(columns=['age_at_recruitment','sex','20544-0.2']) 
z.shape
# 46 subjects, 125 image features

K = np.corrcoef(z) 
r1=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1)
r2=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/2)
r3 = morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

K = gauss_similarity(Z=z, width=1/4)
r4=morph_fit(y=y, X=x, K=K, method="expected", max_iter=50)

scz_res = np.array([ [r1['Estimated morphometricity'], r1['AIC']],
                     [r2['Estimated morphometricity'], r2['AIC']],
                     [r3['Estimated morphometricity'], r3['AIC']],
                     [r4['Estimated morphometricity'], r4['AIC']]])





# %% 
# the overlapped individuals all Nan in autism, adhd, the diagnosed cases are too few.

ad_data = data.drop(columns=['eid', '20544-0.2','20544-0.14','21003-2.0', '20191-0.0'])
ad_data = ad_data.dropna(axis=0, how="any")
ad_data.shape

# %%
# plotting

X = ['Age recruit','Age ses2','Fluid Intelligence','Schizophrenia']
Y0 = [1, 1, 0.95, 0.5]
Y1 = [age_res[0][0]-0.7, age_ses2_res[0][0]-0.7, IQ_res[0][0]-0.7, scz_res[0][0]-0.4]
Y2 = [age_res[1][0], age_ses2_res[1][0],IQ_res[1][0], scz_res[1][0]]
Y3 = [age_res[2][0], age_ses2_res[2][0],IQ_res[2][0], scz_res[2][0]]
Y4 = [age_res[3][0], age_ses2_res[3][0],IQ_res[3][0], scz_res[3][0]]
Y5 = [0.89, 0.89, 0.15, 0.71]  

# Y0 are the reported estimated morphometricity from Sabuncu et al.
# Y5 are the highest prediction R2 from the most recent study (from MRI data, not just morphological measures)
# age:     Irene Cumplido-Mayoral 2022 (UKB)
# fluid intelligence:       Bruno HeblingVieira 2022 (systematic review)
# sczophrenia:      Matthew Bracher-Smith 2022 (UKB)

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y1, 0.1, label = 'Linear')
plt.bar(X_axis - 0.1, Y2, 0.1, label = 'Gaussian 1')
plt.bar(X_axis, Y3, 0.1, label = 'Gaussian 1/2')
plt.bar(X_axis + 0.1, Y4, 0.1, label = 'Gaussian 1/4')
plt.bar(X_axis + 0.2, Y5, 0.1, label = 'Literature')
    
plt.xticks(X_axis, X)
plt.xlabel("Traits")
plt.ylabel("Prediction morphometricity")
plt.legend()
plt.savefig('ukb_traits_morph.png',dpi=150)

# %%
# Comments:
# AIC BIC being criticized for similar reason
# include the method use in past literature