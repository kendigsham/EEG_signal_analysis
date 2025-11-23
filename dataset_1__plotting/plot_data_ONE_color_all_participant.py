#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:49:58 2024

@author: kenny
"""

# /home/kenny/Documents/courses/Individual_project/Analysis/try4_plots_all_participant_per_event_part3.py



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
import pickle


dir_base = '/path/to/dataset_1__preprocessed_data'


with open(f'{dir_base}/each_participant_per_event_average_color_GFP_csv.pickle', 'rb') as handle:
    df_mean = pickle.load(handle)



######################
#   averaging the participant
######################


df_all_participant = df_mean.groupby(['time', 'events'])[['GFP']].mean()


df_all_participant = df_all_participant.reset_index()


df_all_participant.shape   ##  shape  (2048, 3)


# 2048*17 = 34816



#######################
#
#########################


fig, ax = plt.subplots(1, 1, figsize=(12, 8))

fig.suptitle('Average GFP for all participants', fontsize=20, y =1) 
 
sns.lineplot(data=df_all_participant, x='time',y='GFP', ax=ax, hue='events', style='events',
         dashes=[ (2, 2),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0)] )

# axes[i].set_title(participant)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

ax.set_ylabel('GFP (Voltage)')

ax.set_xlabel('time (s)')
    
fig.tight_layout()

plt.savefig('plots/GFP_all_participants.pdf')
# plt.show()
























