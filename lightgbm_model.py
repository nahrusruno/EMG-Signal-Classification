#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:42:47 2021

@author: onursurhan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:32:04 2021

@author: onursurhan
"""
from multiprocessing import Queue
import scipy.io as sio
from mat4py import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from scipy import interpolate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 100)



def get_csv():
    main_df = pd.DataFrame(columns = ['Task','Speed','Stride Length','Subject','EMGFreq','Age','Gender','Height','Mass','TA','SO','GAM','PL','RF','VM','BF','GM'])
    for i in range(1,51):
        data = loadmat('data/Subject{}.mat'.format(i))
        RecordedDataAll = pd.DataFrame(columns = ['Task','Speed','Stride Length','Subject','EMGFreq','Age','Gender','Height','Mass','TA','SO','GAM','PL','RF','VM','BF','GM'])
        RecordedData1 = pd.DataFrame(pd.DataFrame(data).loc['Data','s'])[['Task','EMG','speed','strideLength']]
        RecordedDataAll['Task'] = RecordedData1['Task']
        RecordedDataAll['Speed'] = RecordedData1['speed']
        RecordedDataAll['Stride Length'] = RecordedData1['strideLength']
        RecordedDataAll['Subject'] = str(pd.DataFrame(data).loc['name','s'])
        RecordedDataAll['EMGFreq'] = pd.DataFrame(data).loc['EMGFreq','s']
        RecordedDataAll['Age'] = pd.DataFrame(data).loc['Age','s']
        RecordedDataAll['Gender'] = pd.DataFrame(data).loc['Gender','s']
        RecordedDataAll['Height'] = pd.DataFrame(data).loc['BH','s']
        RecordedDataAll['Mass'] = pd.DataFrame(data).loc['BM','s']
        RecordedDataAll['TA'] = RecordedData1['EMG'].apply(lambda x: x[0][:])
        RecordedDataAll['SO'] = RecordedData1['EMG'].apply(lambda x: x[1][:])
        RecordedDataAll['GAM'] = RecordedData1['EMG'].apply(lambda x: x[2][:])
        RecordedDataAll['PL'] = RecordedData1['EMG'].apply(lambda x: x[3][:])
        RecordedDataAll['RF'] = RecordedData1['EMG'].apply(lambda x: x[4][:])
        RecordedDataAll['VM'] = RecordedData1['EMG'].apply(lambda x: x[5][:])
        RecordedDataAll['BF'] = RecordedData1['EMG'].apply(lambda x: x[6][:])
        RecordedDataAll['GM'] = RecordedData1['EMG'].apply(lambda x: x[7][:])

        main_df = main_df.append(RecordedDataAll, ignore_index=True)


    main_df['Task']=main_df['Task'].apply(lambda x: 'StepUp' if x=='StepUp     ' else x)
    main_df['Task']=main_df['Task'].apply(lambda x: 'StepDown' if x=='StepDown   ' else x)
    main_df['Task']=main_df['Task'].apply(lambda x: 'ToeWalking' if x=='ToeWalking ' else x)
    main_df['Task']=main_df['Task'].apply(lambda x: 'HeelWalking' if x=='HeelWalking' else x)
    main_df['Task']=main_df['Task'].apply(lambda x: 'Walking' if x=='Walking' else x)

    print('Total trail number: {}'.format(main_df.shape[0]))
    print('Walking trail number: {}'.format(main_df[main_df['Task']=='Walking'].shape[0]))
    print('Step Up trail number: {}'.format(main_df[main_df['Task']=='StepUp'].shape[0]))
    print('Step Down trail number: {}'.format(main_df[main_df['Task']=='StepDown'].shape[0]))
    print('Toe Walking trail number: {}'.format(main_df[main_df['Task']=='ToeWalking'].shape[0]))
    print('Heel Walking trail number: {}'.format(main_df[main_df['Task']=='HeelWalking'].shape[0]))    
    
    #main_df.to_csv('data/all_data.csv', index=False)
    #pd.read_csv('data/all_data.csv')
    return main_df




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



class extractFeatures:
    
    def __init__(self, preprocessed_EMG_data):
        
        self.data = preprocessed_EMG_data   #It contains many trials for several muscles
        #self.window_size = window_size

    def rolling_window(self, data, window_size, step_size=1):
        window = window_size
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
                temporary_data = self.rolling_window(self.data.loc[i,j], window_size = self.data.loc[i,j].shape[0])                          
                for l in feature_names:
                    #temporary_array = np.array([])
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
        
        label_encoder = LabelEncoder()
        dataset['Task']= label_encoder.fit_transform(dataset['Task'])
        
        return dataset['Task']
    

class LSTM_Classifier:
    
    def __init__(self, X, y, first_layer_neurons, second_layer_neurons, nb_epoch = 50, batch_size = 128):
        self.X = X
        self.y = y
        self.first_layer_neurons = first_layer_neurons
        self.second_layer_neurons = second_layer_neurons
        self.input_dimension = X.shape[1]
        self.number_of_classes = len(np.unique(y))
        self.number_of_epochs = nb_epoch
        self.batch_size = batch_size
        

        
    def build_model(self):
        model = Sequential()
#        model.add(LSTM(self.first_layer_neurons, input_dim=self.input_dimension , dropout_U=0.3)) ## I don't understand why droupout_U is used
        model.add(LSTM(self.first_layer_neurons, input_dim=self.input_dimension))
        model.add(Dense(self.second_layer_neurons))
        model.add(Dropout(0.2))
        model.add(Dense(self.number_of_classes, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        return model


    def RandomSearchPipeline(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        
        clf=KerasClassifier(build_fn=self.build_model(), 
                            epochs=self.number_of_epochs,
                            batch_size = self.batch_size,
                            verbose=2)

        # early_stop = EarlyStopping(
        #     monitor = 'val_loss', min_delta = 0.01, patience = 5, verbose = 0,restore_best_weights=True)

        # callbacks = [early_stop]
        # keras_fit_params = {   
        #     'callbacks': callbacks,
        #     'epochs': 50,
        #     'batch_size': 128,
        #     'validation_data': (X_test, y_test),
        #     'verbose': 0
        #                     }
        
        
        param_grid = {
                'clf__batch_size': [32,64,128], 
                'clf__optimizer': ['Adam', 'Adadelta']}
        
        #pipe = Pipeline([
        #            ('clf', clf)
        #                ])
        
       # my_cv =  StratifiedKFold(n_splits=5, shuffle = True, random_state=21).split()

        rs_lstm = RandomizedSearchCV(clf, param_grid, cv=5, scoring='accuracy', 
                               verbose=3,n_jobs=1, random_state=21)
        rs_lstm.fit(X_train, y_train)

        return y_test, rs_lstm.predict(X_test)


    def train_test_splitter(self, test_size=0.25):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test= y_test

    
    def cross_validation(self):
        
        X_train = self.X_train
        y_train = self.y_train
        
        cv_model = KerasClassifier(
            build_fn=self.build_model,
            nb_epoch = self.number_of_epochs,
            batch_size = self.batch_size,
            verbose=2
        )
        param_grid = {
            "first_layer_neurons":[10, 50, 100, 150],
            "second_layer_neurons":[10, 50, 100, 150]
        }
        grid = GridSearchCV(estimator=cv_model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
    
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        return grid_result
    
    def fit(self):
        
        the_model = self.build_model().fit(self.X_train, self.y_train, self.number_of_epochs, self.batch_size, verbose = 2)
        scores = the_model.evaluate(self.X_train, self.y_train)
        print("%s: %.2f" % (the_model.metrics_names[1], scores[1]))
        return the_model
    
    
    def predict(self):
        model = self.fit()
        X_test = self.X_test
        y_test = self.y_test
        predictions = model.predict(X_test)
        get_class = lambda classes_probabilities: np.argmax(classes_probabilities) + 1
        y_pred = np.array(map(get_class, predictions))
        if y_test is not None:
            y_true = np.array(map(get_class, y_test))
            print (accuracy_score(y_true, y_pred))
        return y_pred

from multiprocessing import Queue
import scipy.io as sio
from mat4py import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from scipy import interpolate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score
import lightgbm as lgb

X = np.load('X_nonwindowed_nonnormalized_step_up&down.npy')
Y = np.load('y_nonwindowed_nonnormalized_step_up&down.npy')
Y = Y - 1 
#Y = to_categorical(Y)

### NO WINDOW NO INTERPOLATION SCRIPT

# data_class = preprocessor(get_csv())
# resultant_df = data_class.concatingDataframe('non-interpolated_filtered')
# extractor = extractFeatures(preprocessed_EMG_data=resultant_df.loc[:,['TA', 'SO', 'GAM', 'PL', 'RF', 'VM', 'BF', 'GM']])
# extracted_feature_EMGs = extractor.createSelectedFeatures()       
# X = extractor.reshapeDataset(extracted_feature_EMGs)
# y = extractor.labelEncoding(resultant_df).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params =  {'learning_rate': 0.4,
                'max_depth': 15,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'subsample': 0.2,
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance':True,
                'bagging_fraction': 0.8,
                'bagging_freq':5,
                'boosting_type':'dart'}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=30)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval

y_predicted = np.zeros(len(X_test))
for i in range(len(y_pred)):
    if y_pred[i]>0.5:
        y_predicted[i]=1
accuracy_test = accuracy_score(y_test, y_predicted)
print(f'The accuracy of prediction is: {accuracy_test}')

lgb.plot_importance(booster = gbm)



