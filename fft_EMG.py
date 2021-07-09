#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:51:12 2021

@author: onursurhan
"""

import numpy as np

from scipy.fftpack import fft
 

def plot_fft_signal(four_trials_list, df):

    fig, axs = plt.subplots(2, 2, figsize=(24,24))
    plt.subplots_adjust(top=0.915)
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    plt.suptitle('FFT Analysis', fontsize=48)

    i = 0
    j = 0

    for trial in four_trials_list:

        y_value = np.array(df.loc[trial,'TA']) 
        f_s =  df.loc[trial,'EMGFreq']
        N = np.shape(df.loc[trial,'TA'])[0]
        T = 1/f_s
        
        freq_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values = 2.0/N * np.abs(fft(y_value)[0:N//2])

        axs[i, j].plot(freq_values, fft_values, linestyle='-', color='blue')
        axs[i, j].set_xlabel('Frequency [Hz]', fontsize=28)
        axs[i, j].set_ylabel('Amplitude', fontsize=28)
        axs[i, j].set_title("Frequency domain of the signal of {}".format(str(np.array(df.loc[trial,'Subject']))), fontsize=32)

        if i==0 and j==0:
            i=0
            j=1
        elif i==0 and j==1:
            i=1
            j=0
        elif i==1 and j==0:
            i=1
            j=1

    plt.savefig('EDA_fft.jpg', bbox_inches='tight')

    return plt.show()

plot_fft_signal([15,49,195,255], df)



