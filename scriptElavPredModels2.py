#plotting a ROC curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pos = [1,1,1,1,0,1,0,0] #10 possible scores position of the positive class
neg = [0,0,1,0,1,0,2,1] #10 example position of the negative class

tpr = [cs/sum(pos) for cs in np.cumsum(pos)] #true positive rate
fpr = [cs/sum(neg) for cs in np.cumsum(neg)] #false positive rate. cumsum stands for cumulative summation of elements in a list

plt.plot([0.0]+fpr+[1.0], [0.0]+tpr+[1.0], "-", label="1")
plt.plot([0.0, 1.0], [0.0, 1.0], "--", label="Baseline") #structure is [x1,x2] and [y1,y2]
plt.xlabel("False Positive Rate (fpr)")
plt.ylabel("True Positive Rate (tpr)")
plt.legend() #this shows the legend of the plot
plt.show()
#plt.savefig("ROC") #to save the ROC

#calculating the area under the ROC curve (AUC)

df = pd.DataFrame({"B": [0.1,0.1,0.4,0.45],
                 "tp": [0, 1, 1, 0],
                 "fp": [1, 0, 0, 1]
                })
tot_tp = np.sum(df["tp"])
tot_fp = np.sum(df["fp"])

predictions = pd.DataFrame({"A":[0.9,0.9,0.6,0.55],"B":[0.1,0.1,0.4,0.45]})

correctlabels = ["A","B","B","A"]


def areaUnderROC(df, correctlabels, tot_tp, tot_fp):
    #Input: sorted triples of scores, number of true and false positives with the scores wrt of some class c,
    #total sum of true positives, total sum of false positives
    #s_1: score of the positive class
    #tp_1: true positive rate
    #fp_1: false positive rate
    #s_2: score of the negative class
    #tp_2: true positive rate
    #fp_2: false positive rate

    #Output: area under the ROC curve (AUC)
    #returns the area under the ROC curve
    AUC = 0
    cov_tp = 0
    for idx in df.index:
        if df["fp"][idx] == 0:
            cov_tp += df["tp"][idx]
        elif df["tp"][idx] == 0:
            AUC += (cov_tp/tot_tp)*(df["fp"][idx]/tot_fp)
        else:
            AUC += (cov_tp/tot_tp)*(df["fp"][idx]/tot_fp) + (df["tp"][idx]/tot_tp)*(df["fp"][idx]/tot_fp)/2
            cov_tp += df["tp"][idx]
    return AUC

#print("AUC: {}".format(areaUnderROC(predictions,correctlabels)))
print(areaUnderROC(predictions, correctlabels, tot_tp, tot_fp))