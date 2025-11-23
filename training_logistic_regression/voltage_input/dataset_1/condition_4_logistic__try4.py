#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:25:03 2024

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
import pickle


dir_data='/home/kenny/Documents/courses/Individual_project/classification_models/Jasna_data/try3_downsample_threshold'


df_data = pd.read_csv(f'{dir_data}/condition_4__data/df_combined_condition_4__all_channel_all_times.csv',index_col=0)

df_data.head()

df_data.shape  ## (59271, 3267)


# df_y = pd.read_csv('GFP_average_per_participant.csv',index_col=0)

# df_y['participant'] = [string.replace('part_','P') for string in df_y['participant']]

# dict_threshold = dict(zip(df_y.participant, df_y.threshold))

# df_data['y_data'] =df_data['participant'].map(dict_threshold)

df_y_label = pd.read_csv('try_4_df_MEDIAN_y_label.csv',index_col=0)


df_data.color = df_data.color.astype(str)

np.all(df_data.trial == df_y_label.trial) ### True
np.all(df_data.color == df_y_label.color)   ### True
np.all(df_data.participant == df_y_label.participant)  ### True


df_y_label.shape   ## (59271, 4)



with open ('all_event_list.pickle', 'rb') as fp:
    all_event_list = pickle.load(fp)

all_event_list = [str(i) for i in all_event_list]


subset_index = df_y_label.color.isin(all_event_list)   ### Counter({True: 51219, False: 8052})


df_y_label = df_y_label[subset_index]    ### (51219, 4)

df_data = df_data[subset_index]   ### (51219, 4)

np.all(df_data.trial == df_y_label.trial) ### True
np.all(df_data.color == df_y_label.color)   ### True
np.all(df_data.participant == df_y_label.participant)  ### True


########################
# split data between training and testing data
#########################

clean_df = df_data[df_data.columns[~df_data.columns.isin(['trial','color', 'participant'])]]

seed = 1488

X_train, X_test, y_train, y_test = train_test_split(clean_df.values, 
                                                    df_y_label.y_data.values, 
                                                    test_size=0.25, random_state=seed)

## X_train   44453, 3264
## X_test   14818, 3264
## y_test   14818
## y_train   44453

# X_train = X_train.reshape(-1, 1)

# X_test = X_test.reshape(-1,1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)


Counter(y_train[:,0].tolist())   ### Counter({0: 20466, 1: 17948})
 
Counter(y_test[:,0].tolist())   ### Counter({0: 6797, 1: 6008})





########################
#   training and predicting
#########################



# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)   ### for imbalance labels

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test) 


Counter(y_pred)  ### {0: 12805}     #### does not look good here


# instantiate the model (using the default parameters)
logreg_balance = LogisticRegression(random_state=16,class_weight='balanced')   ### for imbalance labels

# fit the model with data
logreg_balance.fit(X_train, y_train)

y_pred_balance = logreg_balance.predict(X_test) 


Counter(y_pred_balance)  ### {0: 12805}      #### does not look good here






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
plt.savefig('condition_4__plots/condition_4_confusion_matrix_try4.pdf', bbox_inches='tight')
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
plt.savefig('condition_4__plots/condition_4_ROC_curve_try4.pdf', bbox_inches='tight')
# plt.show()









#########################
#
##########################



from sklearn.metrics import accuracy_score

# # Example: True labels and predicted labels
# y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
# y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

####  0.53



# Write accuracy to a file
with open("condition_4__accuracy_try4.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.5f}\n")







#########################
#
#########################

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = logreg.predict_proba(X_test)[::,1]

import pickle

with open('condition_4__stuff_plots/cnf_matrix_try4.pickle', 'wb') as handle:
    pickle.dump(cnf_matrix, handle)


with open('condition_4__stuff_plots/y_pred_proba_try4.pickle', 'wb') as handle:
    pickle.dump(y_pred_proba, handle)


with open('condition_4__stuff_plots/y_test_try4.pickle', 'wb') as handle:
    pickle.dump(y_test, handle)



