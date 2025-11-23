#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:37:37 2024

@author: kenny
"""

#### from classification_models/Jasna_data/try3_downsample_threshold/using_MEDIAN/get_downsample_MEDIAN_GFP_threshold.py




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


from mne.io import read_raw_edf, read_raw_bdf
import mne

from numpy import random
import pickle





edited_file_list = glob.glob('/home/kenny/OneDrive/Thierry_Data/dataedited/*_ALL70_interpol.set')

len(edited_file_list)

edited_file_list[:5]


deviants = [12,22,32,42,52,62,72,82]   ### these are not index for the MNE epoch data

with open ('Non_Deviant_event_list.pickle', 'rb') as fp:
    non_deviant_list = pickle.load(fp)



i=0

dict_for_dataframe = defaultdict(list)

for temp_file in edited_file_list:


    # temp_file = edited_file_list[0]
    
    
    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)
    
    
    ######################
    #            deviant
    #######################
    
    actual_index = []       ###getting the actual index
    for code in deviants:
        temp_index = event_id[str(code)]
        actual_index.append(temp_index)
    
    
    deviant_index = np.isin(data_participant.events[:,2], actual_index)
    
    
    #######################
    #         non deviant
    #######################
    
    non_deviant_actual = []
    
    for code in non_deviant_list:
        temp_index = event_id[str(code)]
        non_deviant_actual.append(temp_index)
    
    non_deviant_index = np.isin(data_participant.events[:,2], non_deviant_actual)
    
    
    ##################################################
        
    epoch_deviant = data_participant[deviant_index]
    
    data_deviant = epoch_deviant.get_data(copy=False)
    
    
    epoch_non_deviant = data_participant[non_deviant_index]
    
    data_non_deviant = epoch_non_deviant.get_data(copy=False)
    
    ##############################################
    

    deviant_number = data_deviant.shape[0]
    
    non_deviant_number = data_non_deviant.shape[0]
    
    print('number of deviant', deviant_number)

    print('number of non deviant', non_deviant_number)
    
    
    random_sample_index = random.randint(non_deviant_number, size=(deviant_number))
    
    
    
    #####################################################
    
    downsample_non_deviant = data_non_deviant[random_sample_index,:64,:]
    
    
    no_eye_channels = data_deviant[:,:64,:] 
    
    
    combined_data = np.concatenate((no_eye_channels,downsample_non_deviant),axis=0)  ### (trials,  channels,  time)
    
    
    print('combined shape', combined_data.shape)
    
    # median_trials = np.median(combined_data,axis=0)   ## trying MEDIAN here
    
    GFP_all_trials = np.std(combined_data, axis=1)    ## (trials, time)   GFP here
    
    # median_std_trials = np.median(std_trials, axis=0)  ## (256,)
    
    
    times = epoch_deviant.times
    
    time_index = np.where((times > 0.25) & (times < 0.35))[0]
    
    GFP_trials_time_mean = np.mean(GFP_all_trials[:,time_index], axis =1)    ### (trials,)
    
    
    
    median_std_trials = np.median(GFP_trials_time_mean)  ## one value
    


    dict_for_dataframe['participant'].append(participant)
    
    dict_for_dataframe['threshold'].append(median_std_trials)
    

    
    
df_per_participant = pd.DataFrame(dict_for_dataframe)  ### (17,2)


df_per_participant.to_csv('downSample_GFP_MEDIAN_per_participant_restricted_events.csv')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    