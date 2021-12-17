# %%
# read in data and filter ROI relavant to thickness and Volumn

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


os.chdir('/Users/tingzhang/Documents/GitHub/morphometricity_project/data')

# %%
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
iter_csv = pd.read_csv('ukb46307_demographics.csv', iterator=True, chunksize=10000,error_bad_lines=False)
data_demographics = pd.concat ([chunk.dropna(how='all') for chunk in iter_csv] )
data_demographics.shape #502462 subjects, 11 measurements 
data_demographics = data_demographics[['eid','age_at_recruitment', 'sex']]
# %%
merge_data = data.merge(right = data_demographics, how="inner", on="eid")

# %%
instance2_dat = instance3_dat = pd.DataFrame(merge_data[["eid","age_at_recruitment","sex"]])
for col in merge_data.columns[1:2545]:
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
sub_dat = instance2_dat.iloc[0:1000]


out = os.getcwd()
df_map = {'sub_dat': sub_dat,'ses2': instance2_dat, 'ses3': instance3_dat}
for name, df in df_map.items():
    df.to_csv(os.path.join(out, f'{name}.csv')) 
# %%
