# %%
# load the script
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


os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/code')
from morphometricity import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity
from simulation_func import sim1
#%%
# Simulation 1: to verify the method - when model is correctly specified  

[N,M,L] = [50, 100, 2]
[Va,Ve] = [8,2]
n_sim = 500
res = np.ndarray(shape = (n_sim, 7))

sim1(N=N, M=M, L=L,m2=0.2, n_sim=10)
# %%
