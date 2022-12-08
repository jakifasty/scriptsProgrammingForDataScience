#this script is just for testing purposes

#declaring a variable
a = 3.6

#different types of numbers
i = 314 #integer number
f = 3.14e2 #floating point number, or float
z = 2+3j

#check the type of a variable
typei = isinstance(i, int)
print(typei) #prints True, or False if it is not a float
typef = isinstance(f, float)
print(typef) #prints True, or False if it is not a float
typez = isinstance(z, complex)
print(typef) #prints True, or False if it is not a float

#create strings
s = 'Hello World'
b = isinstance(s, str) #this returns True, or False if it is not a string

#declare a boolean
b1 = True
b2 = isinstance(b1, bool) #this returns True as b1 is a boolean with value True
b3 = isinstance("True", bool) #returns False as "True" is a string

#casting consists in specifying the type of a variable, i.e make a variable of a specific type,
#even if previously it is not of that type
cast_i = int(3.15)
print(cast_i) #prints 3
cast_f = float(3)
print(cast_f) #prints 3.0
cast_s = str(3.13)
print(cast_s) #prints 3.13
cast_f = float(cast_s)

b = bool("hello") #returns True
b = bool("") #returns False
b = bool(5) #returns True
b = bool(0.0) #returns False

#operators are used to make operations
v = 2**3 #exponentiation, i.e 2^3
print(v) #prints 8

#assigning values to variables
x = 12
x+=1 #same as x=x+1
x-=1 #same as x=x-1
x*=2 #same as x=x*2

#comparison operators
b = (2.0 == 2) #returns True

#logical operators
b = ((1+1 == 2) and not (2+2 == 4)) #returns True

#identity operators: is, is not
b_val = 1+1 is 2 #returns True

#lists: indexed lists, ordered lists, changable lists
lang = ["Python", "C", "C++", "Julia", "C#"]
first = lang[0] #returns "Python", as it is the first element of the list
second = lang[1] #returns "C", as it is the second element of the list

first_two = lang[0:2] #returns ["Python", "C"] as we say we start at position 0 and we take the first two elements
first_three = lang[0:3]
all_but_first = lang[1:] #returns ["C", "C++", "Julia", "C#"] as we start at position 1 and we take all the elements from position 1 onwards
all_but_last = lang[:-1] #returns ["Python", "C", "C++", "Julia"] as we start at position 0 and we take all the elements up to the last one, without including the last one
lang[1] = "R" #subsitutes the second element of the list with "R"
lang += ["S"] #add new element to the list (same as doing lang.append("S")) or lang = lang + ["S"] )
lang.insert(1, "Java") #inserts "Java" at position 1

#declaring a new list
l1 = ["a", "b", "c"]
len(l1) #returns 3, as there are 3 elements in the list
l1.count("b") #returns 1, as there is one "b" in the list
l1.append("d") #adds element "d" to the list
l1.remove("a") #removes element "a" from the list
l1.reverse() #reverses the list
l1.extend(["d", "e"]) #adds elements "d" and "e" to the list
l2 = [1,2,3]+[4,5] #creates single list from two arrays concatenated

#tuples: indexed, ordered, unchangable
t = ("a", "b", "c") #declaring a tuple
t_1 = t[1] #returns "b" as it's the value on the position of b
"c" in f #returns True as "c" is in the tuple

#sets: unindexed(not indexed), unordered, changable (no duplicates)
s = {"a", "b", "b", "c"} #declaring a set
s.remove("a")
s.add("d")
s = s.union(set(lang)) #union of two sets 
s = s.difference(set(["b", "c", "d"])) #difference returns the elements in s that are not in the defined set
s = s.intersection(set(["Python", "F#"])) #intersection returns the elements in s that are also in lang
"S" in s #returns True as "S" is in the set
s<set(lang) #returns True as s is a subset of lang

#dicctionaries: indexed, unordered, changable. Used to store data values in keys, i.e. in {key:value} pairs
thisdict = {
  "brand": "Ford", #<key>:<value>, this represents an item in the dictionary
  "model": "Mustang",
  "year": 1964
}
print(thisdict["brand"])

d = {
    "Python": "1994",
    "R": "1995",
    "Julia": "2018",
}
y = d["R"] #returns "1995"
d["S"] = 1976 #adds new item to the dictionary
list(d.keys()) #returns the keys ["Python", "R", "Julia", "S"]
list(d.values()) #returns the values ["1994", "1995", "2018", 1976]
t = d.get("T") #returns None as "T" is not in the dictionary
t = d.get("T", 0)

#declaring new dicctionary
d2 = {
    ("a",1):500, 
    ("b",2):250
}
d2[("b",2)] #returns 250, as it is the value of the key ("b",2)

#if statements (with elif and else)
n = 4
if n>5:
    print("more than 5")
elif n ==5:
    print("equal to 5")
else:
    print("less than 5")

## for loops
for i in range(3):
    print(i) #prints 0, 1, 2, as range starts at 0 and ends at 2

for i in [1,2,3]:
    print(i) #prints 1, 2, 3

for i in "hello":
    print(i) #prints h, e, l, l, o

for e in enumerate(["a", "b", "c"]): #enumerate returns the index and the value of the element in the list
    print(e)

for e in enumerate(["a", "b", "c"]):
    if e[0] % 2 == 0: #if the index is even
        print(e[1]) #prints (0, "a"), (1, "b"), (2, "c")

for i in [1,2,3]:
    if i%2 == 0: #if the number is even
        break #breaks the loop
    print(i) #pif not even, prints i

for i in [1,2,3]:
    if i%2 == 0: #if the number is even
        continue #goes to the next iteration of the loop
    print(i) #if not even, prints i

## while loops (with break and continue)
i = 1
while i<4:
    print(i)
    i+=1

i = 1
while i<4:
    if i%2 == 0:
        break
    print(i)
    i+=1

#prints 1 and then enters infinite loop
i=1
while i < 4:
   if i % 2 == 0:
      continue
   print(i)
   i += 1

#List comprehensions: creating lists without for/while loops
nl = []
for la in lang:
    nl += [la.lower()] 

nl = [la.lower() for la in lang] #same as before, but more efficient

nl = [la.lower() for la in lang if len(la) > 1] #same as before, but including only if the length of the element is greater than 1, i.e items with multiple characters

nl = [la.lower() if len(la) > 1 else la for la in lang] #convert items only with multiple characters to lower case, and leave the rest as they are

nl = [c 
    for la in lang
        for c in la
    ] #generate a list with all the characters 

## functions: using def and return
def add_one_and_print(a):
    a+=1
    print(a)
    return a
b = 1
c = add_one_and_print(b) #prints 2 and returns 2 as the value of c
print(b)

def add_two_to_second(l1): #pass as argument the list 1
    l1[1] += 2

l = [1,2,3,4,5]
r = add_two_to_second(l)
r is None

#functions with default arguments
def difference(a=10, b=20):
    return a-b

d0 = difference()
d1 = difference(3, 5)
d2 = difference(b=5)
d3 = difference(b=2, a=3) #by saying that b is 2 and a is 3 we're actually changing the value of the arguments

#lambda functions: anonymous functions with one expression
r = (lambda x: x+1)(5)

f = lambda x,y: x+y
sum = f(2,3)

def derivation(f,x,h):
    return (f(x+h)-f(x))/h

derivation(lambda x: x**2,8,1e-10)

## classes and objects: class definitions using class
class DSLang:
    def __init__(self, name, year):
        self.name = name #this is an class attribute from the class DSLang
        self.year = year

l1 = DSLang("Python", 1994) #initializing an object
l2 = DSLang("R", 1995)
print(l1.name) #prints "Python"

## methods
class DSLang2:
    def __init__(self, name, year):
        self.name = name #this is an class attribute from the class DSLang
        self.year = year
    
    def age(self, current_year): #this is a method from the class DSLang2
        return current_year - self.year #current_year is a parameter of the method
    
l2 = DSLang2("Julia", 2018) #this is an instance of the class DSLang2
print(l2.age(2019))

##special methods
class Super:
    def __init__(self, age):
        self.age = age
    def __str__(self): #self is used to access the attributes of the object
        return "My age is " + str(self.age) + " years old."
    def __eq__(self, value):
        return self.age == value
    def __len__(self):
        return self.age

o = Super(5)
print(o) #prints "My age is 5"
print(o == 5) #prints True

##inheritance
class Sub(Super): #sub comes from subclass, super comes from superclass
    def __init__(self, age=3):
        self.age = age

s = Sub()
print(s) #prints "My age is 3"
len(s)

## modules: define a modeule by placing the code in a file, named with the extension .py
import my_definitions #importing the module in file my_definitions.py
lo = my_definitions.DSLang("R", 1995) #using the definitions of the module

import my_definitions as md #importing the module in file my_definitions.py and giving it the alias md
lo = md.DSLang("R", 1995) #using the definitions of the module

from my_definitions import DSLang #importing the class DSLang from the module my_definitions
lo = DSLang("R", 1995) #using the definitions of the module

#reloading a module, after editing its definitions
from importlib import reload #importing the reload function from the importlib module
reload(my_definitions)

## input/output: write to standard output

#read from standard input, s will be assigned a string
s = input() 

#write to standard output. I don't understand this
print("R", 1995)
print("N:{} Y:{}".format("R", 1995))
print("N:{} Y:{}".format("R", 1995), file=open("output.txt", "w")) #write to a file
print("F: {:.2f}".format(31.41592)) #print a float with 0 decimals
print("F: {:.4f}".format(31.41592)) #print a float with 4 decimals

#write to files
f = open("temp.txt", "w") #open/create a file for writing
result = [1,2,3]
f.write(str(result)) #write a string to the file. Actually this convers "result" to a string
f.close() #close the file

f = open("temp2.txt", "a") #opens/creates a file for appending text
f.write("Hello world!") #appends the string to the file
f.close() #close the file

#next block is for testing purposes
# Copy and paste functions from Assignment 1 here that you need for this assignment

def column_filter(df): 

    filtered_df = df.copy() #copy input dataframe

    #iterate through all columns and consider to drop a column only if it is not labeled "CLASS" or "ID"
    #you may check the number of unique (non-missing) values in a column by applying the pandas functions
    #dropna and unique to drop missing values and get the unique (remaining) values
    filtered_df = filtered_df.dropna(how = 'all', axis = 1)
    for col in filtered_df.columns:
        if col != "CLASS" and col != "ID":
            if filtered_df[col].dropna().unique().size == 1:
                filtered_df = filtered_df.drop(col, axis=1)

    column_filter = filtered_df.columns #list of the names of the remaining columns, including "CLASS" and "ID"
    
    return filtered_df, column_filter

def apply_column_filter(df, column_filter):

    filtered_new_df = df.copy() #copy input dataframe

    #drop each column that is not included in column_filter
    for col in filtered_new_df.columns:
        if col not in column_filter:
            filtered_new_df = filtered_new_df.drop(col, axis=1)

    return filtered_new_df

def imputation(df):
    df_temp = df.copy()
    values = {}
    for column in df_temp:
        #print('Column Name : ', column)
        columnSeriesObj = df_temp[column]
        if columnSeriesObj.dtype == int or columnSeriesObj.dtype == float:
             values[column] = columnSeriesObj.mean()
        elif columnSeriesObj.dtype == object:
             values[column] = columnSeriesObj.mode()[0]

    #print(values)
    df_temp.fillna(value=values, inplace=True)

    return df_temp, values

def apply_imputation(df,imputation):
    df_temp = df.copy()
    values = imputation
    #print(values)
    df_temp.fillna(value=values, inplace=True)
    return df_temp

def normalization(df, normalizationtype): # minmax (default) or zscore 

    new_df = df.copy() #copy input dataframe
    normalization = {}

    #a mapping (dictionary) from each column name to a triple, consisting of ("minmax",min_value,max_value) or ("zscore",mean,std)
    #consider columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID")
    if normalizationtype == "minmax":
        for col in new_df.columns:
            if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
                #normalization[col] = ({"CLASS": ("minmax", 0, 1), "ID": ("minmax", 0, 1)})
                normalization[col] = ("minmax", new_df[col].min(), new_df[col].max())
                new_df[col] = [(x-new_df[col].min())/(new_df[col].max()-new_df[col].min()) for x in new_df[col]]
    elif normalizationtype == "zscore":
        for col in new_df.columns:
            if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
                #normalization[col] = ({"CLASS": ("zscore", 0, 1), "ID": ("zscore", 0, 1)})
                normalization[col] = ("zscore", new_df[col].mean(), new_df[col].std())
                new_df[col] = new_df[col].apply(lambda x:(x-new_df[col].mean())/new_df[col].std())

    return new_df, normalization

def apply_normalization(df, normalization):

    new_df = df.copy() #copy input dataframe
    
    for col in new_df.columns:
        if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
            if normalization[col][0] == "minmax":
                new_df[col] = (new_df[col] - normalization[col][1])/(normalization[col][2] - normalization[col][1])
            elif normalization[col][0] == "zscore":
                new_df[col] = (new_df[col] - normalization[col][1])/normalization[col][2]

    return new_df

def one_hot(df):

    new_df = df.copy() #copy input dataframe

    one_hot = {} #a mapping (dictionary) from column name to a set of categories (possible values for the feature)

    for col in new_df.columns:
        if (new_df[col].dtype == "object" or new_df[col].dtype == "category") and col != "CLASS" and col !="ID":
            one_hot[col] = set(new_df[col])
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df, one_hot

def apply_one_hot(df, one_hot):
    
    new_df = df.copy() #copy input dataframe

    for col in new_df.columns:
        if new_df[col].dtype == "category" and col != "CLASS" and col !="ID":
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df

def accuracy(df, correctlabels):
    df_temp = df.copy()
    count = 0
    outputlabels = df_temp.idxmax(axis = 1)
    for i in range(outputlabels.size):
        if correctlabels[i] == outputlabels[i]:
            count += 1
    accuracy = count/outputlabels.size
        
    return accuracy

def brier_score(df, correctlabels):
    df_temp = df.copy()
    brier_score = 0
    mean = 0
    df_correct = pd.DataFrame(np.zeros((len(df), len(np.unique(correctlabels)))), columns=np.unique(correctlabels))
    for i in range(len(correctlabels)):
        df_correct.loc[i, correctlabels[i]] = 1
    #print(df_correct)
    for column in df_correct:
        columnSeriesObj = df_correct[column]
        for i in range(columnSeriesObj.size):
            brier_score += (df_correct.loc[i, column] - df_temp.loc[i, column])**2
    brier_score = brier_score/len(df)
    return brier_score

def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def create_bins(df,nobins = 10,bintype = 'equal-width'): # defining the function
    df_temp = df.copy() # creating a copy of the input dataFrame
    binning = {} 
    for column in df_temp: # running trough the colums in the copy of the input dataFrame
        # selecting only the requested columns
        if df_temp[column].dtype == 'int64' or 'float64' or 'int32' or 'float32':
            if column != "CLASS" and column != "ID":
                # differentiating for the two cases equal-width and equal-size, using pd.cut and pd.qcut functions respectively 
                if bintype == 'equal-width':
                    cat, bins = pd.cut(df_temp[column],bins=nobins,retbins=True,labels=False)
                elif bintype == 'equal-size':
                    cat, bins = pd.qcut(df_temp[column],q=nobins,retbins=True,labels=False,duplicates='drop')
                # replacing the extremes of the bins with the required -np.inf and np.inf
                bins[0] = -np.inf
                bins[-1] = np.inf
                # filling the dictionary and the dataFrame
                binning[column] = bins
                df_temp[column] = cat.astype('category')
    return df_temp,binning

def apply_bins(df,binning):
    df_temp = df.copy()
    for column in df_temp:
        if df_temp[column].dtype == 'int64' or 'float64' or 'int32' or 'float32':
            if column != "CLASS" and column != "ID":
                cat, bins = pd.cut(df_temp[column],bins=binning[column],retbins=True,labels=False)
                df_temp[column] = cat.astype('category')
    return df_temp

def getfrequencies(classlabels):
    totalcount = classlabels.value_counts().to_dict()
    totalindex = sum(totalcount.values())
    #total = sum(my_dict.values())
    result = {key: value / totalindex for key, value in totalcount.items()}
    return result

def transform(df, column_filter, imputation, normalization, one_hot): #this function applies all the transformations to the input dataframe
    
    new_df = df.copy() #copy input dataframe

    #apply the column_filter, imputation, normalization, and one_hot transformations to the input dataframe
    new_df = column_filter(new_df)
    new_df = imputation(new_df)
    new_df = normalization(new_df)
    new_df = one_hot(new_df)

    return new_df

def get_nearest_neighbor_predictions(row, k):
    print("test get_nearest_neighbor_predictions")
    return row, k

#from here onwards

# Copy and paste functions from Assignment 1 here that you need for this assignment

def column_filter(df): 

    filtered_df = df.copy() #copy input dataframe

    #iterate through all columns and consider to drop a column only if it is not labeled "CLASS" or "ID"
    #you may check the number of unique (non-missing) values in a column by applying the pandas functions
    #dropna and unique to drop missing values and get the unique (remaining) values
    filtered_df = filtered_df.dropna(how = 'all', axis = 1)
    for col in filtered_df.columns:
        if col != "CLASS" and col != "ID":
            if filtered_df[col].dropna().unique().size == 1:
                filtered_df = filtered_df.drop(col, axis=1)

    column_filter = filtered_df.columns #list of the names of the remaining columns, including "CLASS" and "ID"
    
    return filtered_df, column_filter

def apply_column_filter(df, column_filter):

    filtered_new_df = df.copy() #copy input dataframe

    #drop each column that is not included in column_filter
    for col in filtered_new_df.columns:
        if col not in column_filter:
            filtered_new_df = filtered_new_df.drop(col, axis=1)

    return filtered_new_df

def imputation(df):
    df_temp = df.copy()
    values = {}
    for column in df_temp:
        #print('Column Name : ', column)
        columnSeriesObj = df_temp[column]
        if columnSeriesObj.dtype == int or columnSeriesObj.dtype == float:
             values[column] = columnSeriesObj.mean()
        elif columnSeriesObj.dtype == object:
             values[column] = columnSeriesObj.mode()[0]

    #print(values)
    df_temp.fillna(value=values, inplace=True)

    return df_temp, values

def apply_imputation(df,imputation):
    df_temp = df.copy()
    values = imputation
    #print(values)
    df_temp.fillna(value=values, inplace=True)
    return df_temp

def normalization(df, normalizationtype): # minmax (default) or zscore 

    new_df = df.copy() #copy input dataframe
    normalization = {}

    #a mapping (dictionary) from each column name to a triple, consisting of ("minmax",min_value,max_value) or ("zscore",mean,std)
    #consider columns of type "float" or "int" only (and which are not labeled "CLASS" or "ID")
    if normalizationtype == "minmax":
        for col in new_df.columns:
            if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
                #normalization[col] = ({"CLASS": ("minmax", 0, 1), "ID": ("minmax", 0, 1)})
                normalization[col] = ("minmax", new_df[col].min(), new_df[col].max())
                new_df[col] = [(x-new_df[col].min())/(new_df[col].max()-new_df[col].min()) for x in new_df[col]]
    elif normalizationtype == "zscore":
        for col in new_df.columns:
            if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
                #normalization[col] = ({"CLASS": ("zscore", 0, 1), "ID": ("zscore", 0, 1)})
                normalization[col] = ("zscore", new_df[col].mean(), new_df[col].std())
                new_df[col] = new_df[col].apply(lambda x:(x-new_df[col].mean())/new_df[col].std())

    return new_df, normalization

def apply_normalization(df, normalization):

    new_df = df.copy() #copy input dataframe
    
    for col in new_df.columns:
        if (new_df[col].dtype == "float" or new_df[col].dtype == "int") and col != "CLASS" and col !="ID":
            if normalization[col][0] == "minmax":
                new_df[col] = (new_df[col] - normalization[col][1])/(normalization[col][2] - normalization[col][1])
            elif normalization[col][0] == "zscore":
                new_df[col] = (new_df[col] - normalization[col][1])/normalization[col][2]

    return new_df

def one_hot(df):

    new_df = df.copy() #copy input dataframe

    one_hot = {} #a mapping (dictionary) from column name to a set of categories (possible values for the feature)

    for col in new_df.columns:
        if (new_df[col].dtype == "object" or new_df[col].dtype == "category") and col != "CLASS" and col !="ID":
            one_hot[col] = set(new_df[col])
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df, one_hot

def apply_one_hot(df, one_hot):
    
    new_df = df.copy() #copy input dataframe

    for col in new_df.columns:
        if new_df[col].dtype == "category" and col != "CLASS" and col !="ID":
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df

def accuracy(df, correctlabels):
    df_temp = df.copy()
    count = 0
    outputlabels = df_temp.idxmax(axis = 1)
    for i in range(outputlabels.size):
        if correctlabels[i] == outputlabels[i]:
            count += 1
    accuracy = count/outputlabels.size
        
    return accuracy

def brier_score(df, correctlabels):
    df_temp = df.copy()
    brier_score = 0
    mean = 0
    df_correct = pd.DataFrame(np.zeros((len(df), len(np.unique(correctlabels)))), columns=np.unique(correctlabels))
    for i in range(len(correctlabels)):
        df_correct.loc[i, correctlabels[i]] = 1
    #print(df_correct)
    for column in df_correct:
        columnSeriesObj = df_correct[column]
        for i in range(columnSeriesObj.size):
            brier_score += (df_correct.loc[i, column] - df_temp.loc[i, column])**2
    brier_score = brier_score/len(df)
    return brier_score

def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def create_bins(df,nobins = 10,bintype = 'equal-width'): # defining the function
    df_temp = df.copy() # creating a copy of the input dataFrame
    binning = {} 
    for column in df_temp: # running trough the colums in the copy of the input dataFrame
        # selecting only the requested columns
        if df_temp[column].dtype == 'int64' or 'float64' or 'int32' or 'float32':
            if column != "CLASS" and column != "ID":
                # differentiating for the two cases equal-width and equal-size, using pd.cut and pd.qcut functions respectively 
                if bintype == 'equal-width':
                    cat, bins = pd.cut(df_temp[column],bins=nobins,retbins=True,labels=False)
                elif bintype == 'equal-size':
                    cat, bins = pd.qcut(df_temp[column],q=nobins,retbins=True,labels=False,duplicates='drop')
                # replacing the extremes of the bins with the required -np.inf and np.inf
                bins[0] = -np.inf
                bins[-1] = np.inf
                # filling the dictionary and the dataFrame
                binning[column] = bins
                df_temp[column] = cat.astype('category')
    return df_temp,binning

def apply_bins(df,binning):
    df_temp = df.copy()
    for column in df_temp:
        if df_temp[column].dtype == 'int64' or 'float64' or 'int32' or 'float32':
            if column != "CLASS" and column != "ID":
                cat, bins = pd.cut(df_temp[column],bins=binning[column],retbins=True,labels=False)
                df_temp[column] = cat.astype('category')
    return df_temp

def getfrequencies(classlabels):
    totalcount = classlabels.value_counts().to_dict()
    totalindex = sum(totalcount.values())
    #total = sum(my_dict.values())
    result = {key: value / totalindex for key, value in totalcount.items()}
    return result

def transform(df, column_filter, imputation, normalization, one_hot): #this function applies all the transformations to the input dataframe
    
    new_df = df.copy() #copy input dataframe

    #apply the column_filter, imputation, normalization, and one_hot transformations to the input dataframe
    new_df = column_filter(new_df)
    new_df = imputation(new_df)
    new_df = normalization(new_df)
    new_df = one_hot(new_df)

    return new_df

def get_nearest_neighbor_predictions(row, k):
    print("test get_nearest_neighbor_predictions")
    return row, k



def one_hot(df):

    new_df = df.copy() #copy input dataframe

    one_hot = {} #a mapping (dictionary) from column name to a set of categories (possible values for the feature)

    for col in new_df.columns:
        if new_df[col].dtype == "category" and col != "CLASS" and col !="ID":
            one_hot[col] = set(new_df[col])
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df, one_hot



