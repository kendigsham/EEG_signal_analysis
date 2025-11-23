#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:10:53 2024

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

import warnings
warnings.filterwarnings("ignore")


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
    
    
##################################
#
##################################

deviants = 'B1(70)/80'

standard = 'B2(80)/B2(80)'


i=0

dict_for_dataframe = defaultdict(list)



for temp_file in kept_file_list:

    # temp_file = edited_file_list[0]
        
    data_participant = mne.io.read_epochs_eeglab(temp_file, verbose=False)
    
    # event_id = data_participant.event_id
    
    num_participant = os.path.basename(temp_file).split('_')[0]
            
    participant = f'P{num_participant}'        
    print(participant)
    
    # print(data_participant.get_data(copy=False).shape)
    
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

    # print(data_participant.get_data(copy=False).shape)
    
    #==========================================
    
    event_id = data_participant.event_id
    
    actual_deviant_index = event_id[deviants]       ###getting the actual index

    deviant_index = np.isin(data_participant.events[:,2], actual_deviant_index)
    
    
    dict_true_deviant = dict(Counter(deviant_index))
    
    deviant_number = dict_true_deviant[True]
    
    #==========================================
    
    actual_standard_index = event_id[standard] 
    
    standard_index = np.isin(data_participant.events[:,2], actual_standard_index)
    
    dict_true_standard = dict(Counter(standard_index))
    
    standard_number = dict_true_standard[True]
    
    #========================================
    
    epoch_deviant = data_participant[deviant_index]
    
    data_deviant = epoch_deviant.get_data(copy=False)
    
    data_deviant = data_deviant[:,:28,:]   ### take out the channels that are not on the head
    
    #========================================
    
    epoch_standard = data_participant[standard_index]
    
    data_standard = epoch_standard.get_data(copy=False)
    
    data_standard = data_standard[:,:28,:]   ### take out the channels that are not on the head
    
    print('deviant shape', data_deviant.shape)
    
    print('standard shape', data_standard.shape)
    
    print('number of deviant', deviant_number)
    
    print('number of non deviant', standard_number)
    
    
    
    random_sample_index = random.randint(standard_number, size=(deviant_number))
    
    
    downsample_standard = data_standard[random_sample_index,:,:]
    
    
    print('downsampled_number',downsample_standard.shape)
    
    
    combined_data = np.concatenate((data_deviant, downsample_standard),axis=0)  ### (trials,  channels,  time)
    
    # median_trials = np.median(combined_data,axis=0)   ## trying MEDIAN here
    
    std_trials = np.std(combined_data, axis=1)    ## (trials, time)   GFP here
    
    print(std_trials.shape)
    
    # median_std_trials = np.median(std_trials, axis=0)  ## (256,)
    
    
    times = epoch_deviant.times
    
    time_index = np.where((times > 0.25) & (times < 0.35))[0]
    
    std_trials_time_mean = np.mean(std_trials[:,time_index], axis =1)    ### (trials,)
    
    print('average of time', std_trials_time_mean.shape)
    
    median_std_trials = np.median(std_trials_time_mean)  ## one value
    
    print('final median value', median_std_trials) ## one value
    
    # i+=1
    # if i ==3:
    #     break
    
    ### only the first 28 channels are useful and located on the brain 
    
    dict_for_dataframe['participant'].append(participant)
    
    dict_for_dataframe['threshold'].append(median_std_trials)
    




df_per_participant = pd.DataFrame(dict_for_dataframe)  ### (39,2)


df_per_participant.to_csv('reject_trials_reject_subject7_get_threshold_per_participant_MEDIAN_downsample.csv')

























