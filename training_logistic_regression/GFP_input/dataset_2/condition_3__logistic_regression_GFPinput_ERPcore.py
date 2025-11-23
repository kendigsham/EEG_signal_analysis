#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:22:13 2024

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



dir_x_data = '/home/kenny/Documents/courses/Individual_project/linear_models/ERP_core__GFP_input/condition_3__GFP_data'

df_data = pd.read_csv(f'{dir_x_data}/df_combined_condition_3__GFP_all_times.csv',index_col=0)

df_data.head()

df_data.shape  ## (23013, 54)

df_data.columns

# Index(['time_0', 'time_1', 'time_2', 'time_3', 'time_4', 'time_5', 'time_6',
#        'time_7', 'time_8', 'time_9', 'time_10', 'time_11', 'time_12',
#        'time_13', 'time_14', 'time_15', 'time_16', 'time_17', 'time_18',
#        'time_19', 'time_20', 'time_21', 'time_22', 'time_23', 'time_24',
#        'time_25', 'time_26', 'time_27', 'time_28', 'time_29', 'time_30',
#        'time_31', 'time_32', 'time_33', 'time_34', 'time_35', 'time_36',
#        'time_37', 'time_38', 'time_39', 'time_40', 'time_41', 'time_42',
#        'time_43', 'time_44', 'time_45', 'time_46', 'time_47', 'time_48',
#        'time_49', 'time_50', 'trial', 'event', 'participant'],
#       dtype='object')



dir_base_data_y='/home/kenny/Documents/courses/Individual_project/classification_models/ERP_core/try3__ss2_get_threshold_for_training'

dir_y_label = f'{dir_base_data_y}/df_MEDIAN_y_label_reject_subject_N_trial.csv'


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

## X_train   17259, 51
## X_test   5754, 51
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
plt.savefig('condition_3__GFP_plots/condition_3_confusion_matrix_ERPcore_GFPinput.pdf', bbox_inches='tight')
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
plt.savefig('condition_3__GFP_plots/condition_3_ROC_curve_ERPcore_GFPinput.pdf', bbox_inches='tight')
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

####  0.49



# Write accuracy to a file
with open("condition_3__accuracy_ERPcore_GFPinput.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.5f}\n")







#########################
#
#########################

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

y_pred_proba = logreg.predict_proba(X_test)[::,1]

import pickle

with open('condition_3__GFP_stuff_plots/cnf_matrix_ERPcore_GFPinput.pickle', 'wb') as handle:
    pickle.dump(cnf_matrix, handle)


with open('condition_3__GFP_stuff_plots/y_pred_proba_ERPcore_GFPinput.pickle', 'wb') as handle:
    pickle.dump(y_pred_proba, handle)


with open('condition_3__GFP_stuff_plots/y_test_ERPcore_GFPinput.pickle', 'wb') as handle:
    pickle.dump(y_test, handle)












































