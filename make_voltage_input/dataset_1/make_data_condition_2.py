#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:08:17 2024

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




edited_file_list = glob.glob('/home/kenny/OneDrive/Thierry_Data/dataedited/*_ALL70_interpol.set')

len(edited_file_list)

edited_file_list[:5]





dataframe_list=[]

for temp_file in edited_file_list:
    # print(temp_file)
    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)


    dict_for_channel_mean = defaultdict(list)
    
    for color_name, color_index in event_id.items():
        # print(color_index)
        
        temp_index = np.where(data_participant.events[:,2] == color_index)[0]
    
        sub_epoch_temp = data_participant[temp_index]
        
        temp_data = sub_epoch_temp.get_data(copy=False)
        
        # print(temp_data.shape)
        
        no_eye_channels = temp_data[:,:64,:]   ## already take out the eye channel
    
        # print(no_eye_channels.shape)   ## numpy  (trials, 64, 256)
    
        
        times = sub_epoch_temp.times
        
        index = np.where(times <0)[0]  ### getting the time before the event at 0
        
        channel_times_data = no_eye_channels[:,:,index]
        
        shape_time = channel_times_data.shape
        
        # print(shape_time)    ## (trials, 64, 51)
        
    
        all_channel_names = sub_epoch_temp.ch_names
    
        for trial_index in range(shape_time[0]):
            # print(trial_index)
            
            one_trial_data = channel_times_data[trial_index,:,:]   ### shape = (64, 51)  ## we are doing all channels
            
            mean_time_per_channel = np.mean(one_trial_data, axis=1) ### shape = (64,)
            
            
            
            
            for channel_index in range(mean_time_per_channel.shape[0]):    
                
                dict_for_channel_mean[all_channel_names[channel_index]].append(mean_time_per_channel[channel_index])
                    
            dict_for_channel_mean['trial'].append(trial_index)
            
            dict_for_channel_mean['color'].append(color_name)
            
            dict_for_channel_mean['participant'].append(participant)
                
                
    df_temp_channels = pd.DataFrame(dict_for_channel_mean)
            
    dataframe_list.append(df_temp_channels)
            
        
        
# #################
# #
# #  #################       
        
        
        
save_dir = 'condition_2__data'

df_combine = pd.concat(dataframe_list)
    
df_combine.to_csv(f'{save_dir}/df_combined_condition_2__per_channel_mean_time.csv')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        