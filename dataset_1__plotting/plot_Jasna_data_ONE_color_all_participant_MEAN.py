#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:59:40 2024

@author: kenny
"""



from mne.io import read_raw_edf, read_raw_bdf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import mne
import seaborn
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

# from my_functions import *

import pickle
import os


dir_base = '/home/kenny/Documents/courses/Individual_project/Analysis'



df_mean_std = pd.read_csv(f'{dir_base}/try3_average_participant.csv',index_col=0)

df_mean_std.shape


#################################################################

dict_keys = {'std_2':[11,21,31,41,51,61,71,81],
             'std_3':[111,221,331,441,551,661,771,881],
             'std_4':[1111,2221,3331,4441,5551,6661,7771,8881],
             'std_5':[11111,22221,33331,44441,55551,66661,77771,88881],
             'std_1_nonTargetDev':[9211,9221,9231,9241,9251,9261,9271,9281],
             'std_1_stdColTarget':[9311,9321,9331,9341,9351,9361,9371,9381],
             'std_1_devColTarget':[9411,9421,9431,9441,9451,9461,9471,9481],
             'deviant':[12,22,32,42,52,62,72,82]}



df_dict_keys = pd.DataFrame(dict_keys)

df_melt_dict_keys = pd.melt(df_dict_keys)

df_melt_dict_keys['value'] = df_melt_dict_keys['value'].astype(str)


dictionary_keys = dict(zip(df_melt_dict_keys['value'],df_melt_dict_keys['variable']))



################################################################

df_mean_std.type = df_mean_std.type.astype(str)


df_mean_std['events'] = df_mean_std['type'].map(dictionary_keys)











##############################################################


df_all_participant = df_mean_std.groupby(['time', 'events'])[['mean']].mean()


df_all_participant = df_all_participant.reset_index()


df_all_participant.shape   ##  shape  (2048, 3)

############################################################






fig, ax = plt.subplots(1, 1, figsize=(12, 8))

fig.suptitle('Average voltage for all participants', fontsize=20, y =1) 
 
sns.lineplot(data=df_all_participant, x='time',y='mean', ax=ax, hue='events', style='events',
         dashes=[ (2, 2),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0)] )

# axes[i].set_title(participant)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
ax.set_ylabel('Mean (Voltage)')

ax.set_xlabel('time (s)')
    

fig.tight_layout()

plt.savefig('Jasna_plots/MEAN_voltage_all_participants.pdf', bbox_inches='tight')
# plt.show()









































