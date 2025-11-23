#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:29:21 2024

@author: kenny
"""


import numpy as np
import pandas as pd
import glob
import os
from collections import Counter, defaultdict

# from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns



df_data = pd.read_csv('condition_1__GFP_data/df_combined_condition_1__GFP_mean_time.csv',index_col=0)

df_data.head()

df_data.shape  ## (23013, 4)



dir_y_values = 'df__y_values_reject_subject_N_trial.csv'


df_y_values = pd.read_csv(dir_y_values, index_col=0)




np.all(df_data.trial == df_y_values.trial) ### True
np.all(df_data.event == df_y_values.color)   ### True
np.all(df_data.participant == df_y_values.participant)  ### True


df_y_values.shape   ## (23013, 4)




########################
# split data between training and testing data
#########################

seed = 1488

X_train, X_test, y_train, y_test = train_test_split(df_data.mean_GFP_time.values, 
                                                    df_y_values.y_data.values, 
                                                    test_size=0.25, random_state=seed)

## X_train   17259
## X_test   5754
## y_test   5754
## y_train   17259

X_train = X_train.reshape(-1, 1)

X_test = X_test.reshape(-1,1)

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

df_result.to_csv('condition_1__GFP_results/ERP_core_df_MSE_N_r_squared.csv')




import pickle

with open('condition_1__GFP_results/ERP_core_linear_model__condition_1.pickle', 'wb') as handle:
    pickle.dump(model, handle)


