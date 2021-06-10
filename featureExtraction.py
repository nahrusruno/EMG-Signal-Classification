#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:34:24 2021

@author: onursurhan
"""
import numpy as np
from preprocess_class import *

class extractFeatures:
    
    def __init__(self, preprocessed_EMG_data, window_size):
        
        self.data = preprocessed_EMG_data   #It contains many trials for several muscles
        self.window_size = window_size

    def rolling_window(self, data, step_size=1):
        window = self.window_size
        shape = data.shape[:-1] + (data.shape[-1] - window + 1 - step_size + 1, window)
        strides = data.strides + (data.strides[-1] * step_size,)
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

       
    def WAMP(self, windowed_data, threshold):        
        
        """ Willison Amplitude
        "This feature is defined as the amount of times that the
        change in EMG signal amplitude exceeds a threshold; it is
        an indicator of the firing of motor unit action potentials
        and is thus a surrogate metric for the level of muscle contraction." (Tkach et. al 4)
        wamp = sum of f(abs(data_input[iSample] - data_input[iSample + 1])) in an analysis time window with n samples
        where f(x) = 1 if data_input[iSample] - data_input[iSample + 1]) > wamp_thresh and f(x) = 0 else
        :param windowed_data: input samples to compute feature
        :return: scalar feature value
        """
        
       # wamp=0    
        
       # for i in range(len(windowed_data)):         
       #     if abs(windowed_data[i]-windowed_data[i+1])>threshold:
        #        wamp=wamp+1;    
        fs = 200
        n = windowed_data.shape[0]

        wamp = np.sum(((abs(windowed_data[1:n - 1] - windowed_data[0:n - 2])) > threshold), axis=0) * fs / n
        return wamp  
    
    def RMS(self, windowed_data):

        """ Root Mean Squared
        Compute rms across all samples (axis=0)
        :param windowed_data: input samples to compute feature
        :return: scalar feature value
        """
        
        return np.sqrt(np.mean(windowed_data**2))
    
    def MAV(self, windowed_data):
        
        """ Mean Absolute Value
        Compute mav across all samples (axis=0)
        :param data_input: input samples to compute feature
        :return: scalar feature value
        """
        
        return np.sum(abs(windowed_data))/len(windowed_data)
    
    def ZC(self, windowed_data):
        
        """ Zero-crossings
        Criteria for crossing zero
        zeroCross=(y[iSample] - t > 0 and y[iSample + 1] - t < 0) or (y[iSample] - t < 0 and y[iSample + 1] - t > 0)

        :param windowed_data: input samples to compute feature
        :return: scalar feature value
        """
        
        return ((windowed_data[:-1] * windowed_data[1:]) < 0).sum()
    
    def v_order(self, windowed_data):
        
        """ V-Order Feature
        "This metric yields an estimation of the exerted muscle force...
        characterized by the absolute value of EMG signal
        to the vth power. The applied smoothing filter is the moving
        average window. Therefore, this feature is defined as
        , where E is the expectation operator
        applied on the samples in one analysis window. One study
        indicates that the best value for v is 2, which leads to
        the definition of the EMG v-Order feature as the same as
        the square root of the var feature." (Tkach et. al 4)
        vorder = sqrt(sum of signal x squared in an analysis time window with n samples, over (n-1))
        :param windowed_data: input samples to compute feature
        :return: scalar feature value
        """
        return np.sqrt(np.sum(np.square(windowed_data), axis=0) / (windowed_data.shape[0]-1))
        
 
    def variance(self, windowed_data):
        
        """ Variance
        "This feature is the measure of the EMG signal's power." (Tkach et. al 4)
        var = sum of signal x squared in an analysis time window with n samples all over (n-1)
        :param data_input: input samples to compute feature
        :return: scalar feature value
        """
        return np.sum(np.square(data_input), axis=0) / (windowed_data.shape[0]-1)


    def log_detector(self, windowed_data):
        
        """ V-Order
        "This metric yields an estimation of the exerted muscle force" (Tkach et. al 4)
        logdetect = e raised to (the mean of the log of the absolute value of the signal input)
        :param data_input: input samples to compute feature
        :return: scalar feature value
        """

        # TODO: log detect function needs to protect against log(0) (-INF) occuring
        return math.e**(np.mean(np.log(abs(windowed_data)), axis=0))
    
    def createSelectedFeatures(self, muscles=['all'], features=['all']):
        
        if muscles[0]=='all':
            muscle_names = ['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM']
        else:
            muscle_names = muscles
            
        if features[0]=='all':
            feature_names = ['WAMP','RMS','MAV','ZC','V_ORDER']
        else:
            feature_names = features
        
        new_column_names=[]
        for feature_name in feature_names:
            for column_name in muscle_names:
                #column_names = new_column_names.append(column_name+'_'+feature_name)
                new_column_names.append(column_name+'_'+feature_name)

        print(new_column_names)
        
        extracted_feature_df = pd.DataFrame(columns=new_column_names)        
        
        for i in range(len(self.data)):
            for j in muscle_names:
                #temporary_data = np.lib.stride_tricks.sliding_window_view(self.data.loc[i,j], self.window_size) 
                temporary_data = self.rolling_window(self.data.loc[i,j])                          
                for l in feature_names:
                    temporary_list = []
                    if l=='WAMP':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.WAMP(temporary_data[k,:],0.005))
                    elif l=='RMS':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.RMS(temporary_data[k,:]))
                    elif l=='MAV':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.MAV(temporary_data[k,:]))  
                    elif l=='ZC':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.ZC(temporary_data[k,:])) 
                    elif l=='V_ORDER':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.v_order(temporary_data[k,:])) 
                    elif l=='VAR':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.variance(temporary_data[k,:]))                            
                    elif l=='LOG':
                        for k in range(len(temporary_data)):
                            temporary_list.append(self.log_detector(temporary_data[k,:])) 
                            
                    extracted_feature_df.loc[i,j+'_'+l] = temporary_list
    def reshaper(self, df):
        X = []
        for i in range(np.shape(df)[0]):
            for j in range(np.shape(df)[1]):
                df.loc[i,j]           
 
data_class = preprocessor(get_csv())
resultant_df = data_class.concatingDataframe('interpolated_filtered')

extractor = extractFeatures(preprocessed_EMG_data=resultant_df.loc[:,['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM']], window_size=50)
extracted_feature_EMGs = extractor.createSelectedFeatures()       
                        
        
        
