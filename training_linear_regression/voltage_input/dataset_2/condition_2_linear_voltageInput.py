#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:08:50 2024

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



dir_base_X='/home/kenny/Documents/courses/Individual_project/classification_models/ERP_core'


df_data = pd.read_csv(f'{dir_base_X}/condition_2__data/df_combined_condition_2__per_channel_mean_time.csv',index_col=0)

df_data.head()

df_data.shape  ## (23013, 4)


dir_base_y='/home/kenny/Documents/courses/Individual_project/linear_models/ERP_core__GFP_input'

dir_y_values = f'{dir_base_y}/df__y_values_reject_subject_N_trial.csv'


df_y_values = pd.read_csv(dir_y_values,index_col=0)

# df_y['participant'] = [string.replace('part_','P') for string in df_y['participant']]

# dict_threshold = dict(zip(df_y.participant, df_y.threshold))

# df_data['y_data'] =df_data['participant'].map(dict_threshold)

# df_y_label = pd.read_csv('attempt_1__df_y_label.csv',index_col=0)


np.all(df_data.trial == df_y_values.trial) ### True
np.all(df_data.event == df_y_values.color)   ### True
np.all(df_data.participant == df_y_values.participant)  ### True


df_y_values.shape   ## (23013, 4)



########################
# split data between training and testing data
#########################

clean_df = df_data[df_data.columns[~df_data.columns.isin(['trial','event', 'participant'])]]

seed = 1488

X_train, X_test, y_train, y_test = train_test_split(clean_df.values, 
                                                    df_y_values.y_data.values, 
                                                    test_size=0.25, random_state=seed)

## X_train   483252
## X_test   161112
## y_test   5754
## y_train   17259

# X_train = X_train.reshape(-1, 1)

# X_test = X_test.reshape(-1,1)

y_train = y_train.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)




######################
#
######################



# instantiate the model (using the default parameters)
# lin_regr = linear_model.LinearRegression() 

# fit the model with data
# lin_regr.fit(X_train, y_train)

# X_with_const = sm.add_constant(X)

# y_pred = lin_regr.predict(X_test) 



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

df_result.to_csv('condition_2__results/ERP_core_df_MSE_N_r_squared.csv')




import pickle

with open('condition_2__results/ERP_core_linear_model__condition_2.pickle', 'wb') as handle:
    pickle.dump(model, handle)



