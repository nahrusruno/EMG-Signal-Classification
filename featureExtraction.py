#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: onursurhan
"""
import numpy as np
from preprocess_class import *
from sklearn.preprocessing import LabelEncoder

class extractFeatures:
    
    def __init__(self, preprocessed_EMG_data, window_size):
        
        self.data = preprocessed_EMG_data   #It contains many trials for several muscles
        self.window_size = window_size

    def rolling_window(self, data, step_size=1):
        """ A.K.A. Moving average window. It takes a specified portion from the data provided for making desired calculations.
        Rolling window is beneficial in extracting features from a time series data. Because the rapid changes occur in these kinds
        of data, windowing smoothes these kinds of data and it results in more comprehendible data. 
        
        The data entered in this function, loose some portion of its length as much as (window size-1) ## No padding with zeros ##.
        As can be understood from the notation below, windowing resembles with stride=1 in deep learning. 
        
        The algorithm is vectorized so that the complexity is reduced.
        """
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
        average window. 
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
        
        """ Constructing dataset with extracted features in a desired configuration
        
        param - muscles: desired muscles to be included in the input samples
                        default: 'all' which stands for ['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM']
        param - features: desired features to be calculated for each of the selected muscles to use in model
                        default: 'all' which stands for ['WAMP','RMS','MAV','ZC','V_ORDER']  
                        
        :return: extracted_feature_df: Pandas dataframe which consist of combination of desired muscles and features.
                        Examples for the column names of the extracted_feature_df dataframe: 'TA_WAMP' or 'GAM_RMS'
        
        """
        
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
                    
        return extracted_feature_df
                    
    def reshapeDataset(self, df):
        """ This function takes the dataset with extracted features and makes the dataset
        ready for feeding model by reshaping it. The shape of return is 
        X.shape(number of samples, number of time steps, number of features)
        """
        X = []
        for i in range(np.shape(df)[0]):
            X_per_sample = []
            for j in range(np.shape(df)[1]):
                X_per_sample.append(df.iloc[i,j])
            X.append(np.transpose(np.asarray(X_per_sample)))
        X = np.stack(X, axis=0)
        
        return X
                
    def labelEncoding(self, dataset):
        
        """ The labels which are aimed to be classifed are transformed into """
        self.label_encoder = LabelEncoder()
        dataset['Task']= self.label_encoder.fit_transform(dataset['Task'])
        
        self.labels_array = dataset['Task'].values
        
        return dataset['Task']
    
    def labelDecoder(self):
        
        """ This function can be used when the actual labels are wanted to be seen """
        actual_labels = list(self.label_encoder.inverse_transform(self.labels_array))
    
        return actual_labels
        
data_class = preprocessor(get_csv())
resultant_df = data_class.concatingDataframe('interpolated_filtered')
extractor = extractFeatures(preprocessed_EMG_data=resultant_df.loc[:,['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM']], window_size=50)
extracted_feature_EMGs = extractor.createSelectedFeatures(muscles=['TA','SO','GAM'], features = ['WAMP','RMS'])       
X = extractor.reshapeDataset(extracted_feature_EMGs)
y = extractor.labelEncoding(resultant_df).values
