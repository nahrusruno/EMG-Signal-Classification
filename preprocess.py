#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:57:03 2021

@author: onursurhan
"""


from scipy.signal import butter, lfilter, sosfilt
import matplotlib.pyplot as plt
import pandas as pd
from generateDataset import get_csv



def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def filter_EMGs() :
    
    datam = get_csv() 
    filtered_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
    
    for i in range(len(datam)):
        fs = datam.loc[i,'EMGFreq']
        for j in filtered_df.columns:
            filtered_signal = butter_bandpass_filter(datam.loc[i,'{}'.format(j)],10,200,fs)
            filtered_df.loc[i,'{}'.format(j)] = filtered_signal 
   
    return filtered_df

def ewma_filtered_EMGs():
    
    filtered_data = filter_EMGs() 
    filtered_windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
    for i in range(len(datam)):
        for j in filtered_windowed_df.columns:
            onur_filtered = filtered_data.loc[i,'{}'.format(j)]
            onur_filtered_data = pd.DataFrame(data=onur_filtered)
            filtered_windowed_df.loc[i,'{}'.format(j)]  = onur_filtered_data.ewm(span=5, adjust=True).mean().values      
    return filtered_windowed_df

def ewma_raw_EMGs():    
    datam = get_csv() 
    windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
    for i in range(len(datam)):
        for j in windowed_df.columns:
            onur = datam.loc[i,'{}'.format(j)]
            onur_data = pd.DataFrame(data=onur)
            windowed_df.loc[i,'{}'.format(j)]  = onur_data.ewm(span=5, adjust=True).mean().values      
    return windowed_df

def concatingDataframe(filtered_df):
    
    datam = get_csv() 
    united_df = pd.concat([filtered_df,datam.iloc[:,0:9]],axis=1)
    
    return united_df


# kelebek1 = onur_data.ewm(span=5, adjust=True).mean()
# kelebek2 = onur_data.ewm(span=10, adjust=True).mean()
# kelebek3 = onur_data.ewm(span=15, adjust=True).mean()
# kelebek4 = onur_data.ewm(span=20, adjust=True).mean()
# plt.plot(onur_data)
# plt.plot(kelebek1.values)
# plt.plot(kelebek2.values)
# plt.plot(kelebek3.values)
# plt.plot(kelebek4.values)
# plt.legend(['original','1','2','3','4'])
# plt.show()

datam = get_csv()
filtered_df = filter_EMGs()
raw_windowed = ewma_raw_EMGs()
filtered_windowed = ewma_filtered_EMGs()

onur = datam.loc[1,'TA']
onur_data = pd.DataFrame(data=onur)


surhan = filtered_df.loc[1,'TA']
surhan_data = pd.DataFrame(data=surhan)


raw_window_unique = raw_windowed.loc[1,'TA']
raw_window_unique_data = pd.DataFrame(data=raw_window_unique)

filtered_window_unique = filtered_windowed.loc[1,'TA']
filtered_window_unique_data = pd.DataFrame(data=filtered_window_unique)


plt.plot(onur)
plt.plot(surhan)
plt.plot(raw_window_unique)
plt.plot(filtered_window_unique)
plt.legend(['original','filtered','raw_window_unique','filtered_window_unique'])
plt.show()






# kelebek_filtered = surhan_data.ewm(span=5, adjust=True).mean()
# kelebek_final = onur_data.ewm(span=5, adjust=True).mean()
# plt.plot(onur_data)
# plt.plot(kelebek_filtered.values)
# plt.plot(kelebek_final.values)
# plt.legend(['original','filtered','final'])
# plt.show()

# datam = get_csv() 
# filtered_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])

# for i in range(len(datam)):
#     fs = datam.loc[i,'EMGFreq']
#     for j in filtered_df.columns:
#         filtered_signal = butter_bandpass_filter(datam.loc[i,'{}'.format(j)],10,200,fs)
#         filtered_df.loc[i,'{}'.format(j)] = abs(filtered_signal) 

# filtered_df = pd.concat([filtered_df,datam.iloc[:,0:9]],axis=1)

        # if j != 'Task':
        #     filtered_signal = butter_bandpass_filter(datam.loc[i,'{}'.format(j)],10,200,fs)
            
        #     filtered_df.loc[i,'{}'.format(j)] = filtered_signal 
            
        #     df_data = pd.DataFrame(data=filtered_signal)
        #     df_data.ewm(span=10, adjust=True).mean()
        #     ewma_df.loc[i,'{}'.format(j)] = df_data.values


# onur = datam.loc[1,'TA']
# onur_data = pd.DataFrame(data=onur)
# windowed_data = onur_data.ewm(span=10, adjust=True).mean().values


#surhan = filtered_df.loc[1,'TA']
#surhan_data = pd.DataFrame(data=surhan)
# windowed_data = surhan_data.ewm(span=10, adjust=True).mean().values

