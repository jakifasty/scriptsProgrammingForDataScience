#Will mean-value imputation have the 
# same effect, if performed before 
# normalization, on the distribution of
# the normalized values for those values
# that were originally not missing, for 
# min-max normalization and z-normalization? 
# Explain your reasoning.

import pandas as pd
import numpy as np

df1 = pd.DataFrame({"values": np.random.randn(100)}) #create/load DataFrame of random values

#mean-value imputation
mean = df1["values"].mean() #same as df.mean() because it is a single column

#normalization (min-max)
min = df1["values"].min()
max = df1["values"].max()
min_max_norm = [(x-min)/(max-min) for x in df1["values"]]

#normalization (z-normalization) with mean=0 and std=1
mean = df1["values"].mean() #same as df.mean() because it is a single column
std = df1["values"].std()
z_norm = df1["values"].apply(lambda x: (x-mean)/std)

df2 = pd.DataFrame({"values": np.random.randn(100)}) #create/load DataFrame of random values

#normalization (min-max)
min = df2["values"].min()
max = df2["values"].max()
min_max_norm = [(x-min)/(max-min) for x in df2["values"]]

#normalization (z-normalization)
mean = df2["values"].mean() #same as df.mean() because it is a single column
std = df2["values"].std()
z_norm = df2["values"].apply(lambda x: (x-mean)/std)

#mean-value imputation
mean = df2["values"].mean() #same as df.mean() because it is a single column
