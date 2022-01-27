# load the script
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
from random import seed
from random import choice


from reLM import compute_Score, compute_FisherInfo, EM_update, morph_fit, gauss_ker, gauss_similarity
