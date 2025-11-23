#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:11:53 2024

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
import pickle



with open('ERPcore_mean_for_plots.pickle', 'rb') as handle:
    df_mean = pickle.load(handle)


#################################################

dict_key = {'B1(70)/80':'deviants' , 'B2(80)/B2(80)':'standard'}


df_mean['events'] = df_mean['event_type'].map(dict_key)


###############################################################################

df_all_participant = df_mean.groupby(['time', 'events'])[['mean_auditory']].mean()


df_all_participant = df_all_participant.reset_index()


df_all_participant.shape   ##  shape  (512, 3)





##########################################################################################


fig, ax = plt.subplots(1, 1, figsize=(12, 8))

fig.suptitle('Average voltage for all participants', fontsize=20, y =1) 
 
sns.lineplot(data=df_all_participant, x='time',y='mean_auditory', ax=ax, hue='events', style='events',
         dashes=[ (2, 2),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0)] )

# axes[i].set_title(participant)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    
ax.set_ylabel('Mean (Voltage)')

ax.set_xlabel('time (s)')
    

fig.tight_layout()

plt.savefig('ERPcore_plots/MEAN_all_participant.pdf', bbox_inches='tight')
# plt.show()



























    