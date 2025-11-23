#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:26:35 2024

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



dict_color_keys = {'color_1':[11,111,1111,11111,9211,9311,9411,12],
                   'color_2':[21,221,2221,22221,9221,9321,9421,22],
                   'color_3':[31,331,3331,33331,9231,9331,9431,32],
                   'color_4':[41,441,4441,44441,9241,9341,9441,42],
                   'color_5':[51,551,5551,55551,9251,9351,9451,52],
                   'color_6':[61,661,6661,66661,9261,9361,9461,62],
                   'color_7':[71,771,7771,77771,9271,9371,9471,72],
                   'color_8':[81,881,8881,88881,9281,9381,9481,82]}






def make_dataframe_plot(color_first, color_second):

    type_list_1 = dict_color_keys[color_first]
    type_list_2 = dict_color_keys[ color_second]
    
    dataframe_list=[]
    
    for index in range(len(type_list_1)):
        # print(index)
        
        type_first = type_list_1[index]
        
        type_second = type_list_2[index]
    
        stand_2__color1 = df_mean_std[df_mean_std.type == type_first]
        stand_2__color2 = df_mean_std[df_mean_std.type == type_second]
        
        #################################################
        stacked = np.stack((stand_2__color1['mean'].values, stand_2__color2['mean'].values),axis=1) ### changing to mean here
        
        mean_numpy = np.mean(stacked,axis=1)
        
        new_dataframe = stand_2__color1.copy()
        
        new_dataframe['color2_mean'] = stand_2__color2['mean'].values
        
        new_dataframe['mean_voltage'] = mean_numpy ## this is actuall
        
        new_dataframe['types_2'] = [f'{type_first}_{type_second}']*mean_numpy.shape[0]
    
        # print(new_dataframe.head())
        dataframe_list.append(new_dataframe)
        
    combined_df = pd.concat(dataframe_list)
    
    return combined_df


df_1_2 = make_dataframe_plot('color_1', 'color_2')

df_3_4 = make_dataframe_plot('color_3', 'color_4')

df_5_6 = make_dataframe_plot('color_5', 'color_6')

df_7_8 = make_dataframe_plot('color_7', 'color_8')



###################################
#
####################################



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



# df_1_2['events'] = df_1_2['types_2'].map(dictionary_keys)

# df_3_4['events'] = df_3_4['types_2'].map(dictionary_keys)

# df_5_6['events'] = df_5_6['types_2'].map(dictionary_keys)

# df_7_8['events'] = df_7_8['types_2'].map(dictionary_keys)

new_1_2 = []
for index, row in df_1_2.iterrows():
    # print(index)
    temp_list = row['types_2'].split('_')
    
    new_value = dictionary_keys.get(temp_list[0])

    new_1_2.append(new_value)
df_1_2['events'] = new_1_2



new_3_4 = []
for index, row in df_3_4.iterrows():
    # print(index)
    temp_list = row['types_2'].split('_')
    
    new_value = dictionary_keys.get(temp_list[0])

    new_3_4.append(new_value)
df_3_4['events'] = new_3_4



new_5_6 = []
for index, row in df_5_6.iterrows():
    # print(index)
    temp_list = row['types_2'].split('_')
    
    new_value = dictionary_keys.get(temp_list[0])

    new_5_6.append(new_value)
df_5_6['events'] = new_5_6



new_7_8 = []
for index, row in df_7_8.iterrows():
    # print(index)
    temp_list = row['types_2'].split('_')
    
    new_value = dictionary_keys.get(temp_list[0])

    new_7_8.append(new_value)
df_7_8['events'] = new_7_8












fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(14, 10),sharex=True, sharey=True)

axes = axes.flatten()


# blue_palette = sns.color_palette("Blues",n_colors=8)
# blue_palette.reverse()

sns.lineplot(data=df_1_2, x='time',y='mean_voltage', ax=axes[0], hue='events', palette='Blues' , style='events',
             dashes=[(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0), (2, 2)])

handles1, labels1 = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles1, labels=labels1)

sns.lineplot(data=df_3_4, x='time',y='mean_voltage', ax=axes[1], hue='events', palette='Greens', style='events',
             dashes=[(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0), (2, 2)] )

handles2, labels2 = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles2, labels=labels2)

sns.lineplot(data=df_5_6, x='time',y='mean_voltage', ax=axes[2], hue='events', palette='Reds', style='events',
             dashes=[(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0), (2, 2)] )

handles3, labels3 = axes[2].get_legend_handles_labels()
axes[2].legend(handles=handles2, labels=labels2)

sns.lineplot(data=df_7_8, x='time',y='mean_voltage', ax=axes[3], hue='events', palette='YlOrBr', style='events',
             dashes=[(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0), (2, 2)] )

handles4, labels4 = axes[3].get_legend_handles_labels()
axes[3].legend(handles=handles4, labels=labels4)


axes[0].set_title('blue')
axes[0].set_ylabel('Mean (Voltage)')
axes[0].axvline(x=0, linewidth=1, color='black',  linestyle = '--')

axes[1].set_title('green')
axes[1].axvline(x=0, linewidth=1, color='black',  linestyle = '--')

axes[2].set_title('red/pink')
axes[2].set_ylabel('Mean (Voltage)')
axes[2].axvline(x=0, linewidth=1, color='black',  linestyle = '--')
axes[2].set_xlabel('time (s)')

axes[3].set_title('brown/yellow')
axes[3].axvline(x=0, linewidth=1, color='black',  linestyle = '--')
axes[3].set_xlabel('time (s)')

fig.suptitle('The Mean of all colors for Dataset 1', fontsize=15)

fig.tight_layout()

plt.savefig('Jasna_plots/MEAN_4_colors_Jasna_data.pdf')
# plt.show()


































