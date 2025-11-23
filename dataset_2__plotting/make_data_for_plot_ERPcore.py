#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:52:23 2024

@author: kenny
"""

#### /home/kenny/Documents/courses/Individual_project/classification_models/ERP_core/make_data_condition_3.py

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



dict_for_all_participant = defaultdict(list)

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
    
    auditory_channels = ['FCz', 'Cz', 'CPz', 'Fz']  ## this is from their codes
    
    
    exist_auditory = []
    index_auditory = []
    
    for index, channel in enumerate(channel_names):
        for temp_channel in auditory_channels:
            if temp_channel == channel:
                exist_auditory.append(temp_channel)
                index_auditory.append(index)

       

    times = data_participant.times
    
    event_id = data_participant.event_id
    
    dict_for_channel_mean = defaultdict(list)
    
    for event_type in event_types_list:
        
        actual_deviant_index = event_id[event_type]       ###getting the actual index

        temp_event_index = np.isin(data_participant.events[:,2], actual_deviant_index)
        
        print(event_type)
        
        print(Counter(temp_event_index))
        
        epoch_events = data_participant[temp_event_index]
        
        data_events = epoch_events.get_data(copy=False)  #### take out the useless channels
        
        data_events = data_events[:,index_auditory,:]   ### using auditory channels
        
        print(data_events.shape)
        
        mean_per_trials = np.mean(data_events,axis=1)    ###  getting the mean here

        print(mean_per_trials.shape)  ### (trials, time=256)
        
        average_trial_mean = np.mean(mean_per_trials, axis=0)  ### shape (256,)
        
        length_ = average_trial_mean.shape[0]
        
        temp_mean = average_trial_mean.tolist()
        
    # i+=1
    # if i ==2:
    #     break

        dict_for_all_participant['mean_auditory'].extend(temp_mean)
        dict_for_all_participant['participant'].extend([participant]*length_)
        dict_for_all_participant['event_type'].extend([event_type]*length_)
        dict_for_all_participant['time'].extend(times)
        
        
        
        
#         #

        
df_results = pd.DataFrame(dict_for_all_participant)
 
        


import pickle


with open('ERPcore_mean_for_plots.pickle', 'wb') as handle:
    pickle.dump(df_results, handle)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        