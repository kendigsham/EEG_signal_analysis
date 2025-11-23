#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:22:11 2024

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


from mne.io import read_raw_edf, read_raw_bdf
import mne




edited_file_list = glob.glob('/path/to/dataedited/*_ALL70_interpol.set')

len(edited_file_list)

edited_file_list[:5]





i=0

dataframe_list = []


for temp_file in edited_file_list:

# temp_file = edited_file_list[0]

    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)
    
    
    
    
    index_for_trials = 0
    
    dict_for_info = defaultdict(list)
    
    dict_for_channel_mean = {}
    
    for color_name, color_index in event_id.items():
        print(color_name)
        
        temp_index = np.where(data_participant.events[:,2] == color_index)[0]
    
        sub_epoch_temp = data_participant[temp_index]
        
        temp_data = sub_epoch_temp.get_data(copy=False)
        
        print(temp_data.shape)
        
        no_eye_channels = temp_data[:,:64,:]   ## already take out the eye channel
    
        print(no_eye_channels.shape)   ## numpy  (trials, 64, 256)
    
        
        times = sub_epoch_temp.times
        
        index = np.where(times <0)[0]  ### getting the time before the event at 0
        
        channel_times_data = no_eye_channels[:,:,index]
        
        shape_time = channel_times_data.shape
        
        print(shape_time)
    
    
        for trial_index in range(shape_time[0]):
            # print(trial_index)
            
            one_trial_data = channel_times_data[trial_index,:,:]   ### shape = (64, 51)
            
            np_arr_flat = one_trial_data.flatten()
    
            dict_for_channel_mean[f'{index_for_trials}'] = np_arr_flat
            
            index_for_trials+=1
            
            dict_for_info['trial'].append(trial_index)
            
            dict_for_info['color'].append(color_name)
            
            dict_for_info['participant'].append(participant)
            
        
            
            
    df_combine_times = pd.DataFrame(dict_for_channel_mean)
        
    df_combine_info = pd.DataFrame(dict_for_info)
    
    t_df_times = df_combine_times.T
    
    df_all = pd.concat([df_combine_info.reset_index(drop=True), t_df_times.reset_index(drop=True)], axis=1)
        
    dataframe_list.append(df_all)
    
    # i+=1
    # if i ==2:
    #     break
    
    # t_df_times = df_combine_times.T
    
    # df_all = pd.concat([t_df_times, df_combine_info],ignore_index=True, axis =1)
    
    # df_all = pd.concat([df_combine_info.reset_index(drop=True), t_df_times.reset_index(drop=True)], axis=1)
    
    
    
    
    
    
###########################
#
###########################
    
save_dir = 'condition_4__data'


df_combine = pd.concat(dataframe_list)

df_combine.to_csv(f'{save_dir}/df_combined_condition_4__all_channel_all_times.csv')











