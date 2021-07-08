#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:59:11 2021

@author: onursurhan
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from generateDataset import get_csv

def Age_plot(df):
    
    age_group_list = []
    
    for i in range(len(df)):
        if df['Age'].iloc[i]<18:
            age_group_list.append(' < 18 ')
        elif df['Age'].iloc[i]<55:
            age_group_list.append(' 18 > and < 55 ')
        else :
            age_group_list.append(' > 55 ')
            
    age_df = pd.DataFrame(age_group_list, columns=['Age Group'])
    
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="Age Group", data=age_df, 
                       order = age_df['Age Group'].value_counts().index, 
                       palette="Set3").set_title('Distribution of age groups',
                                                 fontdict = { 'fontsize': 14, 
                                                             'fontweight':'bold'})
    ax.figure.savefig('EDA_age.jpg')
    
    return plt.show()


def Gender_plot(df):
    

    df['Gender'] = df['Gender'].apply(lambda x: 'F '  if x=='f ' else x)
    gender_df = pd.DataFrame(df.loc[:,'Gender'].values, columns=['Gender'])

    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="Gender", data=gender_df, 
                       order = gender_df['Gender'].value_counts().index, 
                       palette="Set3").set_title('Distribution of genders',
                                                 fontdict = { 'fontsize': 14, 
                                                             'fontweight':'bold'})
    ax.figure.savefig('EDA_gender.jpg')
    
    return plt.show()


def EMGFreq_plot(df):
    

    EMGFreq_df = pd.DataFrame(df.loc[:,'EMGFreq'].values, columns=['EMGFreq'])

    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="EMGFreq", data=EMGFreq_df, 
                       order = EMGFreq_df['EMGFreq'].value_counts().index, 
                       palette="Set3").set_title('Distribution of EMG Frequencies',
                                                 fontdict = { 'fontsize': 14, 
                                                             'fontweight':'bold'})
    ax.figure.savefig('EDA_EMGFreq.jpg')
    
    return plt.show()


def Speed_plot(df):
    

    speed_df = pd.DataFrame(df.loc[:,'Speed'].values, columns=['Speed'])

    sns.set_theme(style="darkgrid")
    ax = sns.displot(x="Speed", data=speed_df, kde=True,
                       palette="Set3")
    plt.title('Distribution of Speed', fontdict = { 'fontsize': 14, 
                                                             'fontweight':'bold'})
    plt.savefig('EDA_Speed.jpg')
    
    return plt.show()


def Task_plot(df):
    

    Task_df = pd.DataFrame(df.loc[:,'Task'].values, columns=['Task'])

    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="Task", data=Task_df, 
                       order = Task_df['Task'].value_counts().index, 
                       palette="Set3").set_title('Distribution of Tasks',
                                                 fontdict = { 'fontsize': 14, 
                                                             'fontweight':'bold'})
    ax.figure.savefig('EDA_Task.jpg')
    
    return plt.show()


df = get_csv()
Task_plot(df)
Speed_plot(df)
EMGFreq_plot(df)
Gender_plot(df)
Age_plot(df)