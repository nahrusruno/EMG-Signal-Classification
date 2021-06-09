#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:41:19 2021

@author: onursurhan
"""

from scipy.signal import butter, lfilter, sosfilt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from generateDataset import get_csv

class preprocessor:
    
    def __init__(self, dataset):
        self.datam = dataset
        self.span = 5
        self.lowcut = 10.0
        self.highcut = 200.0
        self.new_sample_size = 1000
        self.butter_order = 2
        
    def butter_bandpass(self, lowcut, highcut, fs, order):
        #order = self.butter_order
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos
    
    def butter_bandpass_filter(self, data, fs):
        order = self.butter_order
        lowcut = self.lowcut
        highcut = self.highcut
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        #data = self.datam
        y = sosfilt(sos, data)
        return y

    def interpolator(self, data, new_sample_size):
        
        x = np.arange(0, np.shape(data)[0])
        y = data

        f = interpolate.interp1d(x, y,axis=0, kind='cubic', bounds_error=False, fill_value= "extrapolate")
        increment = np.shape(data)[0]/new_sample_size
        xnew = np.arange(0, np.shape(data)[0], increment)
        ynew = f(xnew)   # use interpolation function returned by `interp1d`
        
        if np.shape(ynew)[0]==new_sample_size:
            return ynew
        else :
            #return self.interpolator(data=ynew, new_sample_size= self.new_sample_size)
            return ynew[:-1]  
            
    def filter_EMGs(self) :
        
        datam = self.datam
        filtered_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
        
        for i in range(len(datam)):
            fs = datam.loc[i,'EMGFreq']
            for j in filtered_df.columns:
                filtered_signal = self.butter_bandpass_filter(datam.loc[i,'{}'.format(j)], fs)
                filtered_df.loc[i,'{}'.format(j)] = filtered_signal 
       
        return filtered_df

   
    def filter_EMGs_with_interpolation(self) :   
        datam = self.datam
        filtered_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])    
        for i in range(len(datam)):
            fs = datam.loc[i,'EMGFreq']
            for j in filtered_df.columns:
     
                filtered_signal = self.butter_bandpass_filter(datam.loc[i,'{}'.format(j)],fs)
                filtered_df.loc[i,'{}'.format(j)] = self.interpolator(data=filtered_signal, new_sample_size= self.new_sample_size)
    
        return filtered_df

    
    def ewma_raw_EMGs(self):    
        datam = self.datam
        windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
        for i in range(len(datam)):
            for j in windowed_df.columns:
                onur_data = pd.DataFrame(data=datam.loc[i,'{}'.format(j)])
                windowed_df.loc[i,'{}'.format(j)]  = onur_data.ewm(span=self.span, adjust=True).mean().values      
        return windowed_df

    def ewma_filtered_EMGs(self):
        datam = self.datam
        filtered_data = self.filter_EMGs() 
        filtered_windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
        for i in range(len(datam)):
            for j in filtered_windowed_df.columns:
                onur_filtered_data = pd.DataFrame(data=filtered_data.loc[i,'{}'.format(j)])
                filtered_windowed_df.loc[i,'{}'.format(j)]  = onur_filtered_data.ewm(span=self.span, adjust=True).mean().values      
        return filtered_windowed_df
   
    def ewma_raw_EMGs_with_interpolation(self):    
        datam = self.datam
        windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
        for i in range(len(datam)):
            for j in windowed_df.columns:
    
                onur_data = pd.DataFrame(data=datam.loc[i,'{}'.format(j)])
                windowed_df.loc[i,'{}'.format(j)]  = self.interpolator(data=onur_data.ewm(span=self.span, adjust=True).mean().values, new_sample_size = self.new_sample_size)    
                
        return windowed_df
    
    def ewma_filtered_EMGs_with_interpolation(self):    
        datam = self.datam
        filtered_data = self.filter_EMGs() 
        filtered_windowed_df = pd.DataFrame(columns=['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM'])
        for i in range(len(datam)):
            for j in filtered_windowed_df.columns:
    
                onur_data = pd.DataFrame(data=datam.loc[i,'{}'.format(j)])
                filtered_windowed_df.loc[i,'{}'.format(j)]  = self.interpolator(data=onur_data.ewm(span=self.span, adjust=True).mean().values, new_sample_size = self.new_sample_size)    
                
        return filtered_windowed_df
    
    
    def concatingDataframe(self, EMG_concat_preference='interpolated_filtered'):
        
        if EMG_concat_preference == 'interpolated_filtered':
            chosen_df_to_be_transformed = self.filter_EMGs_with_interpolation()
        elif EMG_concat_preference == 'non-interpolated_filtered':
            chosen_df_to_be_transformed = self.filter_EMGs()
        elif EMG_concat_preference == 'non-interpolated_non-filtered':
            return self.datam           
        elif EMG_concat_preference == 'interpolated_filtered_windowed':
            chosen_df_to_be_transformed = self.ewma_filtered_EMGs_with_interpolation()
        elif EMG_concat_preference == 'non-interpolated_filtered_windowed':
            chosen_df_to_be_transformed = self.ewma_filtered_EMGs()  
        elif EMG_concat_preference == 'interpolated_non-filtered_windowed':
            chosen_df_to_be_transformed = self.ewma_raw_EMGs_with_interpolation()
        elif EMG_concat_preference == 'non-interpolated_non-filtered_windowed':
            chosen_df_to_be_transformed = self.ewma_raw_EMGs()

        else :
            print("please type parameter as one of the 7 options")
        
        datam = self.datam
        united_df = pd.concat([chosen_df_to_be_transformed,datam.iloc[:,0:9]],axis=1)
        
        return united_df


the_dataset = get_csv() # Construct raw dataframe from mat file to dataframe
data_class = preprocessor(dataset = the_dataset) # Contstruct preprocessor class with fixed parameters given inside __init__
resultant_df = data_class.concatingDataframe() # Preprocessed dataset with default parameters (desired configurations)








