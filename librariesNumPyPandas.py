import numpy as np #import the library numpy as np
import numpy.ma as ma

#functions available in numpy
np.sqrt(9) #square root of 9 = 3
np.square(3) #squareroot ot 3 = 9, i.e 3**2
np.maximum(2,3) #returns the maximum of the two numbers
np.absolute(-10) #returns the absolute value of the number, i.e. 10
np.sign(-10) #returns the sign of the number, i.e. -1

#create arrays using numpy
a1 = np.array([1,2,3,4]) #create an array of 1 row and 4 columns with the numbers 1 to 4
isinstance(a1, np.ndarray) #check if a1 is an array
a1.shape #returns the shape of the array, i.e. (4,) which means 4 rows and 0 columns

a2 = np.array([[1,2,3,4],[5,6,7,8]]) #create an array with 2 rows and 4 columns
a2.shape #returns shape of the array
a2.dtype #returns the type of the array, i.e. int64

a3 = np.arange([1,2,3]).reshape(3,3,3)
a3.shape #returns shape of the array

a4 = np.zeros((5,2), dtype=bool) #create array of zeros with 5 rows and 2 columns
a5 = np.empty(10) #uninitialized vector of float values

#datatypes int, float, bool, object, may be inferred or declared
a6 = np.array([[1,2], [3,4], [5]], dtype=object) #create an array with 3 rows and 2 columns
a6.shape
a6.dtype

a7 = np.array([1,2,3.14]) #create an array with 3 rows and 1 column

#accessing Numpy arrays
a = np.array([1,2,3,4])
a[0] #returns 1
a[1:3] #IMPORTANT: returns value on positions 1 and 2, i.e. [1,2,3)
a[:3] #returns values on positions 0, 1 and 2, i.e. [1,2,3)
a[2:] #returns values on positions 2, 3, i.e. [3,4]
a[[1,1]] #returns values on position 1 and 1, i.e. [2,2]
a[[True, False, False, True]]

b = np.arange(12).reshape(4,3) #reate array of 12 index positions and reshape it to 4 rows and 3 columns
b[3,2] #return value on position 3,2, i.e. 11
b[[0,3], [0,2]] #return both values on positions 0,3 and 0,2, i.e. [0,11]
b[1:3, :2] #returns values in rows 1 and 2 and columns 0 and 1

#assigning values to array elements
a[0] +=1 #same as a[0] = a[0] + 1
a[2:] = np.array([5,6]) #assign values to positions 0 and 1
a[:2] = [3,4]
b = a[1:3]
b[0] = 3 #this changes the value of a[1] to 3

c = a.copy() #copy array a to array c
c[1] = 7 #assign value 7 to position 1 of array c

#operations involving arrays
s = np.array([1,2,3])
np.square(s) #make the square of each element of the array

#operations involving multiple arrays of same size
a = np.array([1,2,3])
b = np.array([4,5,6])
c = a+b #sum of the positions of each array

#braodcasting. operations involving operands of different size(broadcasting):
#works when differing size equals 1 for one operand
a = np.array([1,2])
b = np.array([4,5,6])
c = a+b #returns error as the sizes of the arrays are different

d = 5*b
e = b > 5
f = np.array([[1,2],[3,4]])
g = a+f

h = np.array([1],[2])
i = np.array([1,2])
j = h-i #in this case the substraction is made for each element of the array, i.e. 1-1 and 1-2, and 2-1 and 2-2

#operating on multiple array elements with apply_along_axis
a = np.arange(15).reshape(5,3)

np.apply_along_axis(lambda x: np.sum(x),0,a) #sum over rows
def f(x):
    return np.sum(x)

np.apply_along_axis(f, 1, a) #sum over columns
np.apply_along_axis(np.sum, 1, a) #equivalent

a = (np.arange(15)*2).reshape(5,3)
np.argmax(a) #returns the index of the maximum values of the array a
np.argmax(a,0) #returns the index of the maximum value in each column
np.argmax(a,1) #returns the index of the maximum value in each row

a = np.array([1,2,3,4], dtype=float)
a[2] = np.nan #assign nan to position 2 of array a
b = np.array([1,2,3,4])
b[2] = np.nan

c = np.array([True, np.nan, False, False])

np.nan == np.nan #returns false
np.nan is np.nan #returns true as nan is equal to nan
np.isnan(np.nan) #returns true

#counting with missing values
np.sum(np.array([1,2,np.nan,4])) 
np.nansum(np.array([1,2,np.nan,4])) #sum of the values of the array
np.nanprod(np.array([1,2,np.nan,4])) #product of the values of the array
np.nanmin(np.array([1,2,np.nan,4])) #return minimum value of the array
np.nanmax(np.array([1,2,np.nan,4])) #return maximum value of the array
np.nanmean(np.array([1,2,np.nan,4])) #return mean value of the array

x = np.array([1, 2, 3, -1, 5])
mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])

#creating DataFrames with pandas
import pandas as pd

values = np.arange(15).reshape(5,3)
df1 = pd.DataFrame(values, columns=["A", "B", "C"]) #create a dataframe with 5 rows and 3 columns
print(df1)

df2 = pd.DataFrame(values, index=list("abcde"), columns=list("ABC"))
print(df2)

df1["B"] #get value of column B
isinstance(df1["B"], pd.Series) #returns true
df1["B"].values
df1["B"].dtype

df1.loc[1:3, ["B", "C"]] #get values of rows 1,2 and 3 and columns B and C
df1.iloc[1:4, 1:]
df1.iloc[1:3]
df1.iloc[:,1:] #return all rows, subset of cols

#accessing DataFrames
df2.loc["a", ["C", "B"]]
df2.loc["a":"d"]

df2.loc[:,["A","B"]].iloc[0:3]
df2.loc["a":"c",["A","B"]]

df2.loc[[True,True,False,False,False],[True,False,True]] #return two rows, and two columns
df2[[True, True, False, False, False]]

df2.loc["a", ["C", "B"]] #get first row and columns C and B
df2.loc["a":"d"] #all but last row, all columns

df2.iloc[0,0]
df2.loc["a","A"] #same as above, but using values and not indexes

#accessing DataFrames through boolean indexing
df1[[True, False, True, False, True]]

df1[df1["B"] % 2 == 0] #return rows where column B is even
df1[df1["B"] % 2 == 0] #return rows where column B is even and columns A and C

df1[df1["C"].isin([5,8,11])] #return rows where column C is 5, 8 or 11
df1[(df1["A"] > 3) & (df1["C"].isin([5,8,11]))] #return rows where column A is greater than 3 and column C is 5, 8 or 11

df1[(df1["B"]%2 == 0) | (df1["C"] > 10)] #return rows where column B is even or column C is 10

#updating dataframes. Assigning values to DataFrames
df1["B"] = [True,True,False,False,False]
df1["B"].dtype #returns type bool
df1["D"] = np.arange(5, dtype=float) #add column D of floats 5.0 to df1
df1["E"] = 1 #add column E of ones to df1
df1.loc[1, "A"] = np.nan #A gets the type float
df1.loc[2, "B"] = np.nan #B gets the type float
df1.isnull().values.any() #returns true if there are any missing values in the dataframe

#dropping rows and columns in DataFrames
df1.drop(columns="A", inplace=True) #remove column A from df1
df2 = df1.drop(columns="A", errors="ignore")
df3 = df2.drop(index=[2,3])

#merging DataFrames (in a SQL style)
df1 = pd.DataFrame({"LKey": list("abcdef"), "A": [0,0,0,1,1,1]})
df2 = pd.DataFrame({"RKey": list("fedcba"), "B": [0,0,0,1,1,1]})
df1.merge(df2, how="outer", left_on="LKey", right_on="RKey") #merging df1 with df2

#creating groupings in DataFrames
df = pd.DataFrame({"A": list("ababab"), "B": [1,2,3,4,5,6],
                  "C": [10,20,30,40,50,60]})
g = df.groupby("A")
g.get_group("a")
g.sum()

df = pd.DataFrame({"A": list("ababab"), "B": [0,0,0,1,1,1],
                   "C": [10,20,30,40,50,60]})
g = df.groupby(["A", "B"])
g.get_group(("a", 0, 10))
g.size()

#defining and using categorical values in DataFrames
df = pd.DataFrame({"id": [1,2,3,4,5], "award": ["silver", "gold", "silver", "silver", "gold"]})
df["award"] = df["award"].astype("category")

df["award"] = df["award"].cat.set_categories(["gold", "silver", "bronze"])
g = df.groupby("award").size()

g.get("iron", 0)

#importing and exporting DataFrames
#reading and writing to comma separated text (csv) files
df = pd.DataFrame({"id": [np.nan, 2,3,4,5],
                    "grade": [np.nan, "b", np.nan, "c", "a"]})
df.to_csv("my_file.csv", index=False)
df2 = pd.read_csv("my_file.csv")

