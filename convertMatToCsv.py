!pip install mat4py
import scipy.io as sio
from mat4py import loadmat
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)

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

