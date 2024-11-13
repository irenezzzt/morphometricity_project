# %%
# load the script

import numpy as np
import math
from numpy._core.numeric import inf, identity
import pandas as pd
from matplotlib import ticker
import csv
import itertools
import matplotlib.pyplot as plt
import random
import matplotlib.mlab as mlab
from random import gauss, seed
from random import choice
from tempfile import TemporaryFile
import os

import morphometricity 
import simulation_function 
#%%
# Simulation 1: to verify the method - when model is correctly specified  
# data generated from linear kernel
# N:        sample size
# M:        number of brain mophological measures
# L:        number of covariates (age and sex)
# true_morph: true morphometricity in data generating process, ranging from 0 to 1 
# n_sim:    number of repeats

# outputs, currently we are only recording 
#   1. mean_m2: the estimated morphometricity m2, averaged across n_sim times of simualtion 
#   2. est_sd: theoretical standard deviation of m3, for each simulation, there is one, and average across n_sim times of simulation
#   3. sd_m2 : sample standard deviation of n_sim number of m2, obtained from n_sim times of simulation
#   4. AIC_choice: the proportion of each kernel chosen by AIC across n_sim times of simulation
#   5. BIC_choice: the proportion of each kernel chosen by BIC across n_sim times of simulation

[N,M,L] = [500, 100, 2]
true_morph = np.linspace(0,1,11) # from 0% to 100%, generate true morph every 10%
n_sim = 1000
#input dictionay => same loop output dic
mean_m2 = np.ndarray(shape = (11, 5)) # 11 true morph, 5 different kernels
est_sd = np.ndarray(shape = (11, 5))
sd_m2 = np.ndarray(shape = (11, 5))
AIC_choice = np.ndarray(shape = (11, 5))
BIC_choice = np.ndarray(shape = (11, 5))

#%%
output_dict = {}

for m2 in true_morph:
    print(m2) # the actual proportion of variance explained, used to simulate data
    
    # the input of sim() kernel="linear" indicates the data generating process uses linear kernel
    # the outputs are estimated using linear kernel(res_lin), gaussian kernel width 0.5(res_gau0), width 1(res_gau1), width 2(res_gau2), and width 4(res_gau4)
    res = sim(N=N, M=M, L=L, m2=m2, n_sim=n_sim, kernel = "linear")
    
    mean_m2 = [ res['res_lin']['estimated m2'].mean(), res['res_gau0']['estimated m2'].mean(), res['res_gau1']['estimated m2'].mean(),
                    res['res_gau2']['estimated m2'].mean(), res['res_gau3']['estimated m2'].mean()]
    est_sd =  [ res['res_lin']['estimated sd'].std(), res['res_gau0']['estimated sd'].std(), res['res_gau1']['estimated sd'].std(),
                    res['res_gau2']['estimated sd'].std(), res['res_gau3']['estimated sd'].std()]
    theoretical_sd   = [res['res_lin']['theoretical sd'].mean(), res['res_gau0']['theoretical var'].mean(), 
                   res['res_gau1']['theoretical sd'].mean(),res['res_gau2']['theoretical var'].mean(), 
                   res['res_gau3']['theoretical sd'].mean()]
    
    AIC = np.column_stack( (res['res_lin']['aic'], res['res_gau0']['aic'], res['res_gau1']['aic'], 
                             res['res_gau2']['aic'], res['res_gau3']['aic']))
    temp  =  np.argmin(AIC,axis=1)
    AIC_choice = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]
   
    BIC = np.column_stack( (res['res_lin']['bic'], res['res_gau0']['bic'], res['res_gau1']['bic'], 
                             res['res_gau2']['bic'], res['res_gau3']['bic']))
    temp = np.argmin(BIC, axis=1)
    BIC_choice = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]
    output_dict[m2]={'estimated m2': mean_m2, 
                     'estimated sd': est_sd,
                     'theoretical_sd': theoretical_sd,
                     'aic': AIC_choice,
                     'bic': BIC_choice}


#%%
from matplotlib import ticker

labels = ['linear', 'gauss bw 1/2', 'gauss bw 1', 'gauss bw 2', 'gauss bw 4']
AIC_res = []
for key in output_dict:
    AIC_res.append(output_dict[key]['aic'])
df = pd.DataFrame(AIC_res, index= true_morph, columns = labels)


df.plot(kind='bar')
plt.ylabel('proportion selected')
plt.xlabel('true morphometricity')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_tick_params(rotation=0)

plt.legend(prop={'size': 7})
plt.savefig('sim_fig2_lin_AIC.png', dpi=300)


#%%

# save each result into a seperate csv file
#out = os.getcwd()
#df_map = {'fig1_aic': AIC_choice,'fig2_bic': BIC_choice, 'fig1_m2': mean_m2, 
#'fig1_est_sd': est_sd, 'fig1_emp_sd':sd_m2}
#for name, df in df_map.items():
#    np.savetxt(os.path.join(out, f'{name}.csv'), df, delimiter=",") 

# %%
# plot Fig 1.

 
fig, ax = plt.subplots()

est_m2_res = []
for key in output_dict:
    est_m2_res.append(output_dict[key]['estimated m2'])
df = pd.DataFrame(est_m2_res, index= true_morph, columns = labels)



ax.plot(true_morph, mean_m2['linear'],label = "linear", )
ax.plot(true_morph, mean_m2['gauss bw 1/2'],label = "gauss bw 1/2")
ax.plot(true_morph, mean_m2['gauss bw 1'],label = "gauss bw 1")
ax.plot(true_morph, mean_m2['gauss bw 2'],label = "gauss bw 2")
ax.plot(true_morph, mean_m2['gauss bw 4'],label = "gauss bw 4")
ax.plot(true_morph, true_morph, label = "true morphometricity", linestyle="--", color="black")
ax.set_ylim([0, 1])
#ax.errorbar(true_morph, mean_m2[:,4],yerr=est_sd[:,4],fmt='-o',label = "gauss bw0.5")

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(10, 6)
plt.savefig('sim_fig1_lin.png',dpi=150)

# %%
# creat table for AIC BIC choice:
tab1_lin = TemporaryFile()
np.savez(tab1_lin, AIC = AIC_choice, BIC= BIC_choice)
_ = tab1_lin.seek(0)

npzfile = np.load(tab1_lin)
sorted(npzfile.files)
npzfile['AIC']
npzfile['BIC']

# %% 
# heat map for AIC BIC choice? X
# change to stacked bar plot or overlapping bar plots (for 3-5 true m2)

kernels=["linear", 'gauss bw 1/2', 'gauss bw 1', 'gauss bw 2', 'gauss bw 4']
true_morph=['0','0.1','0.2','0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0']

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
 
im = ax1.imshow(npzfile['AIC'],"PuOr") 

# Show all ticks and label them with the respective list entries
ax1.set_yticks(np.arange(11))
ax1.set_yticklabels(true_morph)
ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(kernels)

# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(true_morph)):
    for j in range(len(kernels)):
        text = ax1.text(j, i, npzfile['AIC'][i, j],
                       ha="center", va="center", color="w")
 
ax1.set_title("Kernel chosen by cAIC (proportion)")


im2 = ax2.imshow(npzfile['BIC'],"PuOr")
 
ax2.set_yticks(np.arange(11))
ax2.set_yticklabels(true_morph)
ax2.set_xticks(np.arange(5))
ax2.set_xticklabels(kernels)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(true_morph)):
    for j in range(len(kernels)):
        text = ax2.text(j, i, npzfile['AIC'][i, j],
                       ha="center", va="center", color="w")
ax2.set_title("Kernel chosen by BIC (proportion)")


fig.set_size_inches(10, 11)
plt.savefig('sim_fig2_lin.png',dpi=150)


# %%
# data generated from standard gaussian asm

[N,M,L] = [500, 100, 2]
true_morph = np.linspace(0,1,11)
 
n_sim = 1000

mean_m2 = np.ndarray(shape = (11, 5))
est_sd = np.ndarray(shape = (11, 5))
sd_m2 = np.ndarray(shape = (11, 5))
AIC_choice = np.ndarray(shape = (11, 5))
BIC_choice = np.ndarray(shape = (11, 5))


for i in range(11):
    print(i)
    m2 = true_morph[i]
    res = sim(N=N, M=M, L=L, m2=m2, n_sim=n_sim, kernel = "gaussian")
    mean_m2[i] = [ res['res_lin'][:,2].mean(), res['res_gau0'][:,2].mean(), res['res_gau1'][:,2].mean(),
                    res['res_gau2'][:,2].mean(), res['res_gau3'][:,2].mean()]
    est_sd[i] =  [ res['res_lin'][:,2].std(), res['res_gau0'][:,2].std(), res['res_gau1'][:,2].std(),
                    res['res_gau2'][:,2].std(), res['res_gau3'][:,2].std()]
    sd_m2[i]   = [ res['res_lin'][:,3].mean(), res['res_gau0'][:,3].mean(), res['res_gau1'][:,3].mean(),
                    res['res_gau2'][:,3].mean(), res['res_gau3'][:,3].mean()]
    
    AIC = np.column_stack( (res['res_lin'][:,7], res['res_gau0'][:,7], res['res_gau1'][:,7], 
                             res['res_gau2'][:,7], res['res_gau3'][:,7]))
    temp  =  np.argmin(AIC,axis=1)
    AIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]
   
    BIC = np.column_stack( (res['res_lin'][:,8], res['res_gau0'][:,8], res['res_gau1'][:,8], 
                             res['res_gau2'][:,8], res['res_gau3'][:,8]))
    temp = np.argmin(BIC, axis=1)
    BIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]

# %%
# plot Fig 1.

 
fig, ax = plt.subplots()

ax.errorbar(true_morph, mean_m2[:,0],yerr=est_sd[:,0],fmt='-o',label = "linear")
ax.errorbar(true_morph, mean_m2[:,1],yerr=est_sd[:,1],fmt='-o',label = "gauss bw4")
ax.errorbar(true_morph, mean_m2[:,2],yerr=est_sd[:,2],fmt='-o',label = "gauss bw2")
ax.errorbar(true_morph, mean_m2[:,3],yerr=est_sd[:,3],fmt='-o',label = "gauss bw1")
ax.errorbar(true_morph, mean_m2[:,4],yerr=est_sd[:,4],fmt='-o',label = "gauss bw0.5")

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

#plt.show()

fig.set_size_inches(15, 8)
#plt.show()
plt.savefig('sim_fig1_gau.png',dpi=150)

# %%
# creat table for AIC BIC choice:
tab1_gau = TemporaryFile()
np.savez(tab1_gau, AIC = AIC_choice, BIC= BIC_choice)
_ = tab1_gau.seek(0)

npzfile = np.load(tab1_gau)
sorted(npzfile.files)
npzfile['AIC']
npzfile['BIC']

#%% heat map for AIC BIC choice?
kernels=["linear", 'gauss bw4', 'gauss bw2', 'gauss bw1', 'gauss bw0.5']
true_morph=['0','0.1','0.2','0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0']

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
 
im = ax1.imshow(npzfile['AIC'],"PuOr")

# Show all ticks and label them with the respective list entries
ax1.set_yticks(np.arange(11))
ax1.set_yticklabels(true_morph)
ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(kernels)

# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(true_morph)):
    for j in range(len(kernels)):
        text = ax1.text(j, i, npzfile['AIC'][i, j],
                       ha="center", va="center", color="w")
 
ax1.set_title("Kernel chosen by cAIC (proportion)")


im2 = ax2.imshow(npzfile['BIC'],"PuOr")
 
ax2.set_yticks(np.arange(11))
ax2.set_yticklabels(true_morph)
ax2.set_xticks(np.arange(5))
ax2.set_xticklabels(kernels)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(true_morph)):
    for j in range(len(kernels)):
        text = ax2.text(j, i, npzfile['AIC'][i, j],
                       ha="center", va="center", color="w")
ax2.set_title("Kernel chosen by BIC (proportion)")


fig.set_size_inches(10, 11)
plt.savefig('sim_fig2_gau.png',dpi=150)

# %%

# instead of empirical sd, use mean estimated sd?

fig, ax = plt.subplots()

ax.errorbar(true_morph, mean_m2[:,0],yerr=sd_m2[:,0],fmt='-o',label = "linear")
ax.errorbar(true_morph, mean_m2[:,1],yerr=sd_m2[:,1],fmt='-o',label = "gauss bw4")
ax.errorbar(true_morph, mean_m2[:,2],yerr=sd_m2[:,2],fmt='-o',label = "gauss bw2")
ax.errorbar(true_morph, mean_m2[:,3],yerr=sd_m2[:,3],fmt='-o',label = "gauss bw1")
ax.errorbar(true_morph, mean_m2[:,4],yerr=sd_m2[:,4],fmt='-o',label = "gauss bw0.5")

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(15, 8)
plt.savefig('sim_fig1_lin_sd2.png',dpi=150)

#  error bar affects the visual presentation, include in a table?


# Simulation 2: to check the performance under wrongly specified model and the AIC/BIC
# data generated from linear kernel
[N,M,L] = [50, 100, 2]
true_morph = np.linspace(0,1,11)
n_sim = 1000

mean_m2 = np.ndarray(shape = (11, 5))
est_sd = np.ndarray(shape = (11, 5))
sd_m2 = np.ndarray(shape = (11, 5))
AIC_choice = np.ndarray(shape = (11, 5))
BIC_choice = np.ndarray(shape = (11, 5))

for i in range(11):
    print(i)
    m2 = true_morph[i]
    res = simulation_function.sim(N=N, M=M, L=L, m2=m2, n_sim=n_sim, kernel = "linear")
    mean_m2[i] = [ sum(res['res_lin']['estimated m2'])/n_sim, sum(res['res_gau0']['estimated m2'])/n_sim, 
                  sum(res['res_gau1']['estimated m2'])/n_sim, sum(res['res_gau2']['estimated m2'])/n_sim,
                  sum(res['res_gau3']['estimated m2'])/n_sim]
    est_sd[i] =  [ np.array(res['res_lin']['estimated m2']).std(), np.array(res['res_gau0']['estimated m2']).std(), 
                  np.array(res['res_gau1']['estimated m2']).std(), np.array(res['res_gau2']['estimated m2']).std(), 
                  np.array(res['res_gau3']['estimated m2']).std()]
    sd_m2[i]   = [ sum(res['res_lin']['estimated sd'])/n_sim, sum(res['res_gau0']['estimated sd'])/n_sim, 
                  sum(res['res_gau1']['estimated sd'])/n_sim, sum(res['res_gau2']['estimated sd'])/n_sim,
                  sum(res['res_gau3']['estimated sd'])/n_sim]
    
    AIC = np.column_stack( (res['res_lin']['aic'], res['res_gau0']['aic'], res['res_gau1']['aic'], 
                             res['res_gau2']['aic'], res['res_gau3']['aic']))
    temp  =  np.argmin(AIC,axis=1)
    AIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]
   
    BIC = np.column_stack( (res['res_lin']['bic'], res['res_gau0']['bic'], res['res_gau1']['bic'], 
                             res['res_gau2']['bic'], res['res_gau3']['bic']))
    temp = np.argmin(BIC, axis=1)
    BIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]

#%%

# save each result into a seperate csv file
out = os.getcwd()
df_map = {'fig1_aic': AIC_choice,'fig2_bic': BIC_choice, 'fig1_m2': mean_m2, 
'fig1_est_sd': est_sd, 'fig1_emp_sd':sd_m2}
for name, df in df_map.items():
    np.savetxt(os.path.join(out, f'{name}.csv'), df, delimiter=",") 

# %%
# plot Fig 1.

 
fig, ax = plt.subplots()

ax.plot(true_morph, mean_m2[:,0],marker='o', label = "linear")
ax.plot(true_morph, mean_m2[:,1],marker='o', label = "gauss bw 2")
ax.plot(true_morph, mean_m2[:,2],marker='o', label = "gauss bw 1")
ax.plot(true_morph, mean_m2[:,3],marker='o', label = "gauss bw 1/2")
ax.plot(true_morph, mean_m2[:,4],marker='o', label = "gauss bw 1/4")


ax.plot(true_morph, true_morph, label = "true morphometricity", linestyle="--", color="black")
ax.set_ylim([0, 1])
#ax.errorbar(true_morph, mean_m2[:,4],yerr=est_sd[:,4],fmt='-o',label = "gauss bw0.5")

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(10, 6)
fig.show()
plt.savefig('sim_fig1_lin.png',dpi=150)
# %%
fig, ax = plt.subplots()

ax.errorbar(true_morph, mean_m2[:, 0], yerr=est_sd[:,0], capsize=3, label="sample sd")
ax.errorbar(true_morph, mean_m2[:, 0], yerr=sd_m2[:,0], capsize=3, label="estimated sd")


ax.plot(true_morph, true_morph,label = "true morphometricity", linestyle="--", color="black")
 

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(10, 6)
fig.show()
#plt.savefig('sim_fig_S1_lin.png',dpi=150)

# %%
# creat table for AIC BIC choice:
tab1_lin = TemporaryFile()
np.savez(tab1_lin, AIC = AIC_choice, BIC= BIC_choice)
_ = tab1_lin.seek(0)

npzfile = np.load(tab1_lin)
sorted(npzfile.files)
npzfile['AIC']
npzfile['BIC']



#%%
kernels=["linear", 'gauss bw 4', 'gauss bw 2', 'gauss bw 1', 'gauss bw 1/2']
true_morph=['0','0.1','0.2','0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0']
a = npzfile['AIC']
 
aa = pd.DataFrame(a,columns=kernels, index=true_morph)
aa.plot(kind='bar')
plt.ylabel('Proportion of Selection')
plt.xlabel('True morphometricity')
plt.legend(loc=2, prop={'size': 7})


plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.gca().xaxis.set_tick_params(rotation=0)
plt.savefig('sim_fig2_lin_AIC.png',dpi=150)
#%% heat map for AIC BIC choice?
# change to stacked bar plot or overlapping bar plots (for 3-5 true m2)
kernels=["linear", 'gauss bw 2', 'gauss bw 1', 'gauss bw 1/2', 'gauss bw 1/4']
true_morph=['0','0.1','0.2','0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0']
a = npzfile['AIC']
b = npzfile['BIC']

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
 
ax1.bar(true_morph, a[:,0], width=0.2, label="linear") 
ax1.bar(true_morph, a[:,1], width=0.2, label="gauss 2", bottom = a[:,0])
ax1.bar(true_morph, a[:,2], width=0.2, label="gauss 1", bottom = a[:,0]+a[:,1])
ax1.bar(true_morph, a[:,3], width=0.2, label="gauss 1/2", bottom = a[:,0]+a[:,1]+a[:,2])
ax1.bar(true_morph, a[:,4], width=0.2, label="gauss 1/4", bottom = a[:,0]+a[:,1]+a[:,2]++a[:,3])
ax1.set_title("Kernel chosen by cAIC (proportion)")
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

ax2.bar(true_morph, b[:,0], width=0.2, label="linear") 
ax2.bar(true_morph, b[:,1], width=0.2, label="gauss 4", bottom = b[:,0])
ax2.bar(true_morph, b[:,2], width=0.2, label="gauss 2", bottom = b[:,0]+b[:,1])
ax2.bar(true_morph, b[:,3], width=0.2, label="gauss 1", bottom = b[:,0]+b[:,1]+b[:,2])
ax2.bar(true_morph, b[:,4], width=0.2, label="gauss 0.5", bottom = b[:,0]+b[:,1]+b[:,2]+b[:,3])
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

ax2.set_title("Kernel chosen by BIC (proportion)")
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5)

fig.set_size_inches(14, 6)
fig.show()
plt.savefig('sim_fig2_lin.png',dpi=150)


# %%
# data generated from standard gaussian asm

[N,M,L] = [50, 100, 2]
true_morph = np.linspace(0,1,11)
 
n_sim = 1000

mean_m2 = np.ndarray(shape = (11, 5))
est_sd = np.ndarray(shape = (11, 5))
sd_m2 = np.ndarray(shape = (11, 5))
AIC_choice = np.ndarray(shape = (11, 5))
BIC_choice = np.ndarray(shape = (11, 5))


for i in range(11):
    print(i)
    m2 = true_morph[i]
    res = simulation_function.sim(N=N, M=M, L=L, m2=m2, n_sim=n_sim, kernel = "gaussian")
    mean_m2[i] = [ sum(res['res_lin']['estimated m2'])/n_sim, sum(res['res_gau0']['estimated m2'])/n_sim, 
                  sum(res['res_gau1']['estimated m2'])/n_sim, sum(res['res_gau2']['estimated m2'])/n_sim,
                  sum(res['res_gau3']['estimated m2'])/n_sim]
    est_sd[i] =  [ np.array(res['res_lin']['estimated m2']).std(), np.array(res['res_gau0']['estimated m2']).std(), 
                  np.array(res['res_gau1']['estimated m2']).std(), np.array(res['res_gau2']['estimated m2']).std(), 
                  np.array(res['res_gau3']['estimated m2']).std()]
    sd_m2[i]   = [ sum(res['res_lin']['estimated sd'])/n_sim, sum(res['res_gau0']['estimated sd'])/n_sim, 
                  sum(res['res_gau1']['estimated sd'])/n_sim, sum(res['res_gau2']['estimated sd'])/n_sim,
                  sum(res['res_gau3']['estimated sd'])/n_sim]
    
    AIC = np.column_stack( (res['res_lin']['aic'], res['res_gau0']['aic'], res['res_gau1']['aic'], 
                             res['res_gau2']['aic'], res['res_gau3']['aic']))
    temp  =  np.argmin(AIC,axis=1)
    AIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]
   
    BIC = np.column_stack( (res['res_lin']['bic'], res['res_gau0']['bic'], res['res_gau1']['bic'], 
                             res['res_gau2']['bic'], res['res_gau3']['bic']))
    temp = np.argmin(BIC, axis=1)
    BIC_choice[i] = [(temp ==0).mean(),(temp ==1).mean(),(temp ==2).mean(),
                      (temp ==3).mean(),(temp ==4).mean() ]





 # %%
# plot Fig 3.

 
fig, ax = plt.subplots()

ax.plot(true_morph, mean_m2[:,0],marker='o', label = "linear")
ax.plot(true_morph, mean_m2[:,1],marker='o', label = "gauss bw 2")
ax.plot(true_morph, mean_m2[:,2],marker='o', label = "gauss bw 1")
ax.plot(true_morph, mean_m2[:,3],marker='o', label = "gauss bw 1/2")
ax.plot(true_morph, mean_m2[:,4],marker='o', label = "gauss bw 1/4")


ax.plot(true_morph, true_morph, label = "true morphometricity", linestyle="--", color="black")
ax.set_ylim([0, 1])
#ax.errorbar(true_morph, mean_m2[:,4],yerr=est_sd[:,4],fmt='-o',label = "gauss bw0.5")

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(10, 6)
fig.show()
#plt.savefig('sim_fig1_gau.png',dpi=150)
# %%
fig, ax = plt.subplots()

ax.errorbar(true_morph, mean_m2[:, 0], yerr=est_sd[:,0], capsize=3, label="sample sd")

ax.plot(true_morph, true_morph,label = "true morphometricity", linestyle="--", color="black")
 

ax.legend()
ax.set_xlabel('true morphometricity')
ax.set_ylabel('estimated morphometricity')
ax.set_title('')

 
fig.set_size_inches(10, 6)
plt.savefig('sim_fig_S1_gau.png',dpi=150)

# %%
# creat table for AIC BIC choice:
tab1_gau = TemporaryFile()
np.savez(tab1_gau, AIC = AIC_choice, BIC= BIC_choice)
_ = tab1_gau.seek(0)

npzfile = np.load(tab1_gau)
sorted(npzfile.files)
npzfile['AIC']
npzfile['BIC']

#%% heat map for AIC BIC choice?
# change to stacked bar plot or overlapping bar plots (for 3-5 true m2)
kernels=["linear", 'gauss bw2', 'gauss bw1', 'gauss bw 1/2', 'gauss bw 1/4']
true_morph=['0','0.1','0.2','0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0']
a = npzfile['AIC']
b = npzfile['BIC']

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
 
ax1.bar(true_morph, a[:,0], width=0.2, label="linear") 
ax1.bar(true_morph, a[:,1], width=0.2, label="gauss 4", bottom = a[:,0])
ax1.bar(true_morph, a[:,2], width=0.2, label="gauss 2", bottom = a[:,0]+a[:,1])
ax1.bar(true_morph, a[:,3], width=0.2, label="gauss 1", bottom = a[:,0]+a[:,1]+a[:,2])
ax1.bar(true_morph, a[:,4], width=0.2, label="gauss 0.5", bottom = a[:,0]+a[:,1]+a[:,2]++a[:,3])
ax1.set_title("Kernel chosen by cAIC (proportion)")

ax2.bar(true_morph, b[:,0], width=0.2, label="linear") 
ax2.bar(true_morph, b[:,1], width=0.2, label="gauss 4", bottom = b[:,0])
ax2.bar(true_morph, b[:,2], width=0.2, label="gauss 2", bottom = b[:,0]+b[:,1])
ax2.bar(true_morph, b[:,3], width=0.2, label="gauss 1", bottom = b[:,0]+b[:,1]+b[:,2])
ax2.bar(true_morph, b[:,4], width=0.2, label="gauss 0.5", bottom = b[:,0]+b[:,1]+b[:,2]+b[:,3])

ax2.set_title("Kernel chosen by BIC (proportion)")
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5)

fig.set_size_inches(14, 6)
plt.savefig('sim_fig2_gau.png',dpi=150)
# %%
