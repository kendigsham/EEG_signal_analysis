#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:14:34 2024

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


dir_data = '/home/kenny/Documents/courses/Individual_project/Analysis_ERP_core'


file_list = glob.glob('/home/kenny/Documents/courses/Individual_project/Emily_Kappenman/MMN_All_Data_and_Scripts/**/*_MMN_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set')


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
    
    

##################################
#
##################################

deviants = 'B1(70)/80'

standard = 'B2(80)/B2(80)'


event_types_list = [deviants,standard]


dataframe_list=[]



# dict_for_dataframe = defaultdict(list)
i=0
for temp_file in kept_file_list:

    # temp_file = edited_file_list[0]
        
    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    
    event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)
    
    
    
    #################################
    ##
    #################################
    reject_file_name = f'{dir_data}/try3_rejected_trials/P_{num_participant}.txt'
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
    
    
       
       
    channel_names = data_participant.ch_names
    
    # auditory_channels = ['FCz', 'Cz', 'CPz', 'Fz']  ## this is from their codes
    
    
    # exist_auditory = []
    # index_auditory = []
    
    # for index, channel in enumerate(channel_names):
    #     for temp_channel in auditory_channels:
    #         if temp_channel == channel:
    #             exist_auditory.append(temp_channel)
    #             index_auditory.append(index)

       

    times = data_participant.times
    
    index_for_trials = 0
    
    dict_for_info = defaultdict(list)
    
    dict_for_channel_mean = {}
    
    for event_type in event_types_list:
        
        actual_deviant_index = event_id[event_type]       ###getting the actual index

        temp_event_index = np.isin(data_participant.events[:,2], actual_deviant_index)
        
        print(event_type)
        
        print(Counter(temp_event_index))
        
        epoch_events = data_participant[temp_event_index]
        
        data_events = epoch_events.get_data(copy=False)  #### take out the useless channels
        
        data_events = data_events[:,:28,:]   ### using auditory channels
        
        print(data_events.shape)
        

        all_channel_names = epoch_events.ch_names[:28] ### no useless channels
        
        
        index = np.where(times <0)[0] ### this is the peak for GFP
        
        channel_times_data = data_events[:,:,index]
        
        shape_time = channel_times_data.shape
        
        print(shape_time)    ## (trials, 51)
            
    

    
        for trial_index in range(channel_times_data.shape[0]):
            
            one_trial_data = channel_times_data[trial_index,:,:] ## shape  (28,51)
            
            np_arr_flat = one_trial_data.flatten()  ### looks like it is 1 row after an other

            dict_for_channel_mean[f'{index_for_trials}'] = np_arr_flat
            
            index_for_trials+=1
            
            dict_for_info['trial'].append(trial_index)
            
            dict_for_info['event'].append(event_type)
            
            dict_for_info['participant'].append(participant)

    
    df_combine_times = pd.DataFrame(dict_for_channel_mean)
        
    df_combine_info = pd.DataFrame(dict_for_info)
    
    t_df_times = df_combine_times.T
    
    df_all = pd.concat([df_combine_info.reset_index(drop=True), t_df_times.reset_index(drop=True)], axis=1)
        
    dataframe_list.append(df_all)
            
            
            
            
            
            
            
            
            

    
###########################
#
###########################
    
save_dir = 'condition_4__data'


df_combine = pd.concat(dataframe_list)  ## shape   23013 , 1431

df_combine.to_csv(f'{save_dir}/df_combined_condition_4__all_channel_all_times.csv')




            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            