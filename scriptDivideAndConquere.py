#algorithm code for the divide and conquere algorithm

import numpy as np
import sys
from IPython.display import display
import skylearn 
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def divideAndConquer(i, f): #function that divides the array in two parts, i are the instances and f the features
    
    #Input: instances I = {(X_1,y_1), ..., (X_n,y_n)},
    #   features F
    #Output: a decision tree T

    t = 0 #initialization of the tree

    if Terminate(i, f):
        return (t = LeafNode(i))

    


    return t #return the decision tree

#Assuming that "glass_train.csv" and "glass_test.csv" are in the same directory as this script

glass_train_df = np.loadtxt('../../glass_train.csv')
y = glass_train_df["CLASS"].values

glass_train_df.drop(["CLASS", "ID"], axis="columns", inplace=True)
X = glass_train_df.values

dt = tree.DecisionTreeClassifier(max_depth = 3)

dt.fit(X, y)

#decision tree in Scikil-learn.
tree_desc = tree.export_text(dt, feature_names=list(glass_train_df.columns))
print(tree_desc)

#another decision tree in Scikil-learn.
glass_test_df = np.loadtxt('../../glass_test.csv')
test_y = glass_test_df["CLASS"].values
test_x = glass_test_df.drop(["CLASS", "ID"], axis="columns", inplace=True).values

predictions = dt.predict(test_x)
display(predictions)

print("Accuracy: {:.4}" .format(np.sum(test_y==predictions)/len(predictions)), dt.score(test_x, test_y))

predictions = dt.predict_proba(test_x)
df = pd.DataFrame(predictions, columns=dt.classes_)
display(df)

