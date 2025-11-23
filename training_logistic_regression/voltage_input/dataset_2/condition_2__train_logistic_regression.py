#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:30:53 2024

@author: kenny
"""



import numpy as np
import pandas as pd
import glob
import os
from collections import Counter, defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns



df_data = pd.read_csv('condition_2__data/df_combined_condition_2__per_channel_mean_time.csv',index_col=0)

df_data.head()

df_data.shape  ## (23013, 31)

df_data.columns

# Index(['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
#        'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
#        'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz',
#        'Fp2', 'AF8', 'AF4', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6',
#        'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
#        'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'trial',
#        'color', 'participant'],
#       dtype='object')


dir_y_label = 'try3__ss2_get_threshold_for_training/df_MEDIAN_y_label_reject_subject_N_trial.csv'


df_y_label = pd.read_csv(dir_y_label,index_col=0)


# df_y['participant'] = [string.replace('part_','P') for string in df_y['participant']]

# dict_threshold = dict(zip(df_y.participant, df_y.threshold))

# df_data['y_data'] =df_data['participant'].map(dict_threshold)

# df_y_label = pd.read_csv('attempt_1__df_y_label.csv',index_col=0)

# df_data.color = df_data.color.astype(str)

np.all(df_data.trial == df_y_label.trial) ### True
np.all(df_data.event == df_y_label.color)   ### True
np.all(df_data.participant == df_y_label.participant)  ### True


df_y_label.shape   ## (23013, 4)




########################
# split data between training and testing data
#########################


clean_df = df_data[df_data.columns[~df_data.columns.isin(['trial','event', 'participant'])]]

seed = 1488

X_train, X_test, y_train, y_test = train_test_split(clean_df.values, 
                                                    df_y_label.y_data.values, 
                                                    test_size=0.25, random_state=seed)

## X_train   17259, 28
## X_test   5754, 28
## y_test   5754
## y_train   17259

# X_train = X_train.reshape(-1, 1)

# X_test = X_test.reshape(-1,1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)


Counter(y_train[:,0].tolist())   ### Counter({0: 8643, 1: 8616})
 
Counter(y_test[:,0].tolist())   ### Counter({1: 2911, 0: 2843})







########################
#   training and predicting
#########################



# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)   ### for imbalance labels

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test) 


Counter(y_pred)  ### {0: 5754}     #### does not look good here


# instantiate the model (using the default parameters)
logreg_balance = LogisticRegression(random_state=16,class_weight='balanced')   ### for imbalance labels

# fit the model with data
logreg_balance.fit(X_train, y_train)

y_pred_balance = logreg_balance.predict(X_test) 


Counter(y_pred_balance)  ### {0: 5754}      #### does not look good here




########################
#  https://www.datacamp.com/tutorial/understanding-logistic-regression-python?dc_referrer=https%3A%2F%2Fwww.google.com%2F
#########################


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix



class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('condition_2__plots/condition_2_confusion_matrix.pdf', bbox_inches='tight')
# plt.show()

# Text(0.5,257.44,'Predicted label');






#########################
#
##########################


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC curve')
plt.savefig('condition_2__plots/condition_2_ROC_curve.pdf', bbox_inches='tight')
# plt.show()







##########################
#
#########################




from sklearn.metrics import accuracy_score

# # Example: True labels and predicted labels
# y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
# y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

####  0.51


# Write accuracy to a file
with open("condition_2__accuracy.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.5f}\n")



#########################
#
#########################

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = logreg.predict_proba(X_test)[::,1]

import pickle

with open('condition_2__stuff_plots/cnf_matrix.pickle', 'wb') as handle:
    pickle.dump(cnf_matrix, handle)


with open('condition_2__stuff_plots/y_pred_proba.pickle', 'wb') as handle:
    pickle.dump(y_pred_proba, handle)


with open('condition_2__stuff_plots/y_test.pickle', 'wb') as handle:
    pickle.dump(y_test, handle)


































