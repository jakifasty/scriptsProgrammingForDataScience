import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


#generate random data
a = np.random.multivariate_normal([0,5],[[2,1],[1,3]], size=10)
b = np.random.multivariate_normal([5,0],[[3,1],[1,4]], size=10)
X = np.concatenate((a,b))
plt.scatter(X[:,0], X[:,1])
for i in range(X.shape[0]):
    plt.text(X[i,0], X[i,0], str(i))
plt.show()

Z = linkage(X, "ward")
print(Z)

plt.title("Agglomerative Ward Cluster")
plt.xlabel("Instance")
plt.ylabel("Distance")
d = dendrogram(Z, leaf_rotation=90)
plt.show()

#frequent itemset mining using MLxtend
df = pd.DataFrame("tic-tac-toe.txt")
transactions = [[col+"="+str(row[col]) for col in df.columns] for _,row in df.iterrows()]
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
database = pd.DataFrame(te_ary, columns=te.columns_)
database