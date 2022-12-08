#dropping rows or columns with missing values 
import pandas as pd
import numpy as np

#dropping rows or columns with missing values
df1 = pd.DataFrame({"id": [np.nan, 2, 3, 4, 5],
                "grade": [np.nan, "b", np.nan, "c", np.nan],
                "award": [np.nan, "gold", "silver", "bronze", np.nan]})
df1.dropna(how="any") #drop those lines with NaN values
df1.dropna(how="all", subset=["grade", "award"]) #drop all those lines 

#imputing missing values
df2 = pd.DataFrame({"id": [np.nan, 2,3,4,5],
                    "grade": [np.nan, "b", np.nan, "c", np.nan],
                    "award": [np.nan, "gold", "silver", 
                    "bronze", np.nan]})
values = {"grade": "e", "award": "iron"}
df2.fillna(value=values) #fill NaN values with the values in the dictionary
df2["id"].fillna(df2["id"].mean(), inplace=True) #inplace marks if DataFrame is modifyed (True) or another one is created
df2["award"].fillna(df2["award"].mode()[0], inplace=True) #The mode of a set of values is the value that appears most often. It can be multiple values

#bining in DataFrames
randDf = pd.DataFrame({"values": np.random.rand(100)}) #assign random values to a DataFrame of 100 positions
res, bins = pd.cut(randDf["values"], 4, retbins=True) #cut the values in 4 bins
bins
res
new_res = pd.cut(randDf["values"], bins) #used to go from a continious variable to a categorical variable

#equal-sized binning (using qcut)
df = pd.DataFrame({"values": np.random.rand(100)})
bins
res

#encoding features: normalization

#min-max normalization
df = pd.DataFrame({"values": np.random.randn(100)})
min = df["values"].min()
max = df["values"].max()
df["values"] = [(x-min)/(max-min) for x in df["values"]]

#z-normalization: mean = 0, std = 1
df = pd.DataFrame({"values": np.random.randn(100)})
mean = df["values"].mean() #same as df.mean() because it is a single column
std = df["values"].std()
df["values"] = df["values"].apply(lambda x: (x-mean)/std)

#encoding features: dimensionality reduction    
#selection of top-ranked categorical features (Don't understand this so good)
for col in df2.columns:
        df2[col] = df2[col].astype("category") #this casts the column to a categorical variable

res = [(col,[df2[col].groupby(["award"]).size().values for (n,g) in df2.groupby(["award"])]) 
        for col in df2.columns.drop([0,1])] #groups DataFrame using the column and counts the number of values in each group
scores = [(col,scores(res)) for (col,r) in res]
sorted_scores = sorted(scores,key=lambda tup: tup[1],reverse=True)
filtered = [col for (col,score) in sorted_scores[:2]]
new_df = df.loc[:,filtered]


