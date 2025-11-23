#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:51:53 2024

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


dir_data='/home/kenny/Documents/courses/Individual_project/linear_models/Jasna_dat_try2_GFP_input'

df_data = pd.read_csv(f'{dir_data}/condition_3__GFP_data/df_combined_condition_3__GFP_all_times.csv',index_col=0)

df_data.head()

df_data.shape  ## (59271, 54)


data_dir = '/home/kenny/Documents/courses/Individual_project/linear_models/Jasna_data'

dir_y_values = f'{data_dir}/df__y_data__mean_GFP.csv'



df_y_values = pd.read_csv(dir_y_values,index_col=0)



df_data.color = df_data.color.astype(str)

np.all(df_data.trial == df_y_values.trial) ### True
np.all(df_data.color == df_y_values.color)   ### True
np.all(df_data.participant == df_y_values.participant)  ### True


df_y_values.shape   ## (59271, 4)





y_label_dir='/home/kenny/Documents/courses/Individual_project/classification_models/Jasna_data/try4_correct_number_of_event_type'

with open (f'{y_label_dir}/all_event_list.pickle', 'rb') as fp:
    all_event_list = pickle.load(fp)

all_event_list = [str(i) for i in all_event_list]


subset_index = df_y_values.color.isin(all_event_list)   ### Counter({True: 51219, False: 8052})


df_y_values = df_y_values[subset_index]    ### (51219, 4)

df_data = df_data[subset_index]   ### (51219, 4)

np.all(df_data.trial == df_y_values.trial) ### True
np.all(df_data.color == df_y_values.color)   ### True
np.all(df_data.participant == df_y_values.participant)  ### True





########################
# split data between training and testing data
#########################

clean_df = df_data[df_data.columns[~df_data.columns.isin(['trial','color', 'participant'])]]

seed = 1488

X_train, X_test, y_train, y_test = train_test_split(clean_df.values, 
                                                    df_y_values.y_data.values, 
                                                    test_size=0.25, random_state=seed)

## X_train   38414,64
## X_test   12805,64
## y_test   12805
## y_train   38414

# X_train = X_train.reshape(-1, 1)

# X_test = X_test.reshape(-1,1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)





############################
#
############################



from sklearn.metrics import mean_squared_error, r2_score


# mean_squared_error(y_test, y_pred)

# r2_score(y_test, y_pred)


import statsmodels.api as sm

X_train_with_const = sm.add_constant(X_train)
X_test_with_const = sm.add_constant(X_test)

# Fit the linear regression model on the training data
model = sm.OLS(y_train, X_train_with_const).fit()

# Make predictions on the test data
y_pred = model.predict(X_test_with_const)

# Calculate Mean Squared Error (MSE) on the test data
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared on the test data
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')



######################################
#
#######################################

dict_for_results = {'MSE':[mse],'R_squared':[r_squared]}

df_result = pd.DataFrame(dict_for_results)

df_result.to_csv('condition_3__GFP_results_restricted_events/df_MSE_N_r_squared.csv')




import pickle

with open('condition_3__GFP_results_restricted_events/linear_model__condition_3.pickle', 'wb') as handle:
    pickle.dump(model, handle)












































