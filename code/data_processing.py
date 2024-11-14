# read in data and filter ROI relavant to thickness and Volumn

import numpy as np
from numpy import linalg
from numpy._core.numeric import identity
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
import os

DATAPATH = 'data/irene-phd/ukb-data'
datadir = os.path.join(os.path.expanduser('~'),DATAPATH)
# need to check if datadir exists and if not create it
# if not os.path.exists(datadir):
#    os.path.mkdir(datadir)
os.chdir(datadir)

# %%
#iter_csv = pd.read_csv('ukb47552.csv', iterator=True, chunksize=10000,error_bad_lines=False)
iter_csv = pd.read_csv('ukb47552.csv', iterator=True, chunksize=10000,on_bad_lines='error')
data = pd.concat([chunk.dropna(how='all') for chunk in iter_csv] )
data.shape #502462 subjects, 2545-1 measurements

# %%
# filter data by the field id related to "thickness" and "Volumn"
url = "https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=196"
field_id = pd.DataFrame(pd.read_html(url)[0])
# still having problem with read_html, error: lmxl is not installed (but I definitely have it installed)
rows = [s for s, x in enumerate(field_id["Description"]) if "thickness" in x or "Volume" in x]
filtered_id = field_id.loc[rows]["Field ID"].apply(str)
# JB: str is undefined ? 

# %%
iter_csv = pd.read_csv('ukb46307_demographics.csv', iterator=True, chunksize=10000,on_bad_lines='error')
data_demographics = pd.concat ([chunk.dropna(how='all') for chunk in iter_csv] )
data_demographics.shape #502462 subjects, 11 measurements
                        # JB : finds (502485, 11)
data_demographics = data_demographics[['eid','age_at_recruitment', 'sex']]
# %%
merge_data = data.merge(right = data_demographics, how="inner", on="eid")
merge_data.shape
# JB : len(merge_data) : 502462

# %%
# extract more phenotypes
# ukb field 20544 #2(SCZ), #12(Autism), #14(ADD/ADHD)
#     field 20191 fluid intelligence at

column_codes = {
    "eid":"eid",
    "21003-2.0": "age_at_ses2",
    "20191-0.0":"fluid intelligence",
    "20544-0.2":"scz",
    "20544-0.12":"autism",
    "20544-0.14":"adhd"}

phenotype = pd.read_csv("current.csv", usecols=column_codes.keys())
phenotype = phenotype.dropna(axis=0, how="all")
phenotype.shape
# 6801 subjects in the phenotype data

# %%
merge_data2 = merge_data.merge(right=phenotype, how="inner", on="eid")
merge_data2.shape
#(6801, 2552)
# assert(merge_data.shape==(6801,2552)) ? 
print(merge_data.shape)
traits = merge_data2[['eid', 'age_at_recruitment', 'sex', '20191-0.0', '20544-0.2', '20544-0.12','20544-0.14', '21003-2.0']]

# 
# In [31]: traits.shape
# Out[31]: (6801, 8)


# JB: some quick comment here ? why 2545 ? 
instance2_dat = instance3_dat = pd.DataFrame(traits)
for col in merge_data.columns[1:2545]:
    field, instance = col.split("-")
    if any(filtered_id == field):
        if instance == "2.0":
            instance2_dat[field] = data[col]
        if instance == "3:0":
            instance3_dat[field] = data[col]

col_to_drop = ['eid', 'age_at_recruitment', 'sex', '20191-0.0', '20544-0.2', '20544-0.12','20544-0.14', '21003-2.0']

instance2_dat.drop(columns=col_to_drop, axis=1)
instance3_dat.drop(columns=col_to_drop, axis=1)

instance2_dat.dropna(axis = 0, how = "all", inplace=True)
instance3_dat.dropna(axis = 0, how = "all", inplace=True)

# Consider only complete cases, each instances have 6801 subjects with 132 features (including morphological measurs and phenotypes)
print(instance2_dat.shape, instance3_dat.shape)
# Consider only complete cases, each instances have 43107 subjects with 62
# JB: why the print is returning:  
#:  print(instance2_dat.shape, instance3_dat.shape) 
#: (6801, 132) (6801, 132)

# %%
# Take a subset of instance2 first
sub_dat = instance2_dat.iloc[0:1000]

# write 3 files in the current directory : sub_dat.csv, ses2.csv and ses3.csv
# os.chdir(datadir)
# out = os.getcwd()

df_map = {'sub_dat': sub_dat,'ses2': instance2_dat, 'ses3': instance3_dat}
for name, df in df_map.items():
    df.to_csv(os.path.join(datadir, f'{name}.csv'))

# %%
# application data:

