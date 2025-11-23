#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:40:20 2024

@author: kenny
"""


import mne
import glob
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

import seaborn as sns
import scipy.io
import os

from numpy import random
import matplotlib.pyplot as plt


dir_data = '/path/to/data/Analysis_ERP_core'


file_list = glob.glob('/path/to/data/MMN_All_Data_and_Scripts/**/*_MMN_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set')

### from their codes, the paper is using only 39 participant for MMN
kept_subjects = ['1', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']


for i in range(40):
    string = str(i+1)
    
    if string not in kept_subjects:
        print('this participant will be decarded: ',string)
        
#### subject 7 was not used

kept_file_list=[]

for temp_file in file_list:
    basename = os.path.basename(temp_file)
    
    subject_number = basename.split('_')[0]
    
    if subject_number in kept_subjects:
        kept_file_list.append(temp_file)
    
    
    
df_GFP = pd.read_csv('reject_trials_reject_subject7_get_threshold_per_participant_MEDIAN_downsample.csv',index_col=0)

dict_for_threshold = dict(zip(df_GFP['participant'],df_GFP['threshold']))




##################################
#
##################################

deviants = 'B1(70)/80'

standard = 'B2(80)/B2(80)'


event_types_list = [deviants,standard]


dataframe_list=[]

i=0

dict_for_dataframe = defaultdict(list)

for temp_file in kept_file_list:

    # temp_file = edited_file_list[0]
        
    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    
    event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)
    
    part_threshold = dict_for_threshold[participant]
    
    #################################
    ##
    #################################
    reject_file_name = f'{dir_data}/rejected_trials/P_{num_participant}.txt'
    with open(reject_file_name, 'r') as file:
        content = file.read().strip()
        
    # print(content)
        
    if not content:
        # print(content)
        print("The file contains only whitespace. Please provide a valid file.")
        print(num_participant)

    else:

        rejected_trials = pd.read_csv(reject_file_name,header=None)
    
        new_index = list(rejected_trials[0]-1)   ### make them python index
    
        
        actual_index =[]
        
        for i in range(data_participant.get_data(copy=False).shape[0]):
            if i not in new_index:
                actual_index.append(i)
            
    

        data_participant = data_participant[actual_index]
        
    #========================================


    times = data_participant.times
    
    
    dict_for_y = defaultdict(list)
    
    
    for event_type in event_types_list:
        
        actual_deviant_index = event_id[event_type]       ###getting the actual index

        temp_event_index = np.isin(data_participant.events[:,2], actual_deviant_index)
        
        print(event_type)
        
        print(Counter(temp_event_index))
        
        epoch_events = data_participant[temp_event_index]
        
        data_events = epoch_events.get_data(copy=False)  #### take out the useless channels
        
        data_events = data_events[:,:28,:]   ### take out the channels that are not on the head
        
        print(data_events.shape)
        
        std_per_trials = np.std(data_events,axis=1)    ### GFP  here

        print(std_per_trials.shape)  ### (trials, time=256)
        
        
        index = np.where((times > 0.25) & (times < 0.35))[0]  ### this is the peak for GFP
        
        channel_times_data = std_per_trials[:,index]
        
        shape_time = channel_times_data.shape
        
        print(shape_time)    ## (trials, 25)
        
        for trial_index in range(channel_times_data.shape[0]):
            one_trial_data = channel_times_data[trial_index,:]
            
            
            mean_std_data = np.mean(one_trial_data, axis=0)
            
            y=0
            if mean_std_data > part_threshold:
                y=1
                
            dict_for_y['y_data'].append(y)
    
            dict_for_y['trial'].append(trial_index)
    
            dict_for_y['color'].append(event_type)
    
            dict_for_y['participant'].append(participant)
            
            # dict_for_y['std'].extend(std_per_trials[index,].tolist())
            # dict_for_y['time'].extend(times.tolist())
            # dict_for_y['event_type'].extend([event_type]*times.shape[0])
            # dict_for_y['number'].extend([index]*times.shape[0])
            # dict_for_y['participant'].extend([participant]*times.shape[0])
    
    
    # i+=1
    # if i ==3:
    #     break
        
    df_combine_temp = pd.DataFrame(dict_for_y)
    
    dataframe_list.append(df_combine_temp)
    




df_combine = pd.concat(dataframe_list)
    
df_combine.to_csv('df_MEDIAN_y_label_reject_subject_N_trial.csv')
        

df_label_numbers = pd.crosstab(df_combine.participant, df_combine.y_data).reset_index()
            
df_label_numbers.to_csv('df_y_label_numbers_per_class_per_participant_reject_subject_N_trial.csv')













































