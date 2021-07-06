#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:24:13 2021

@author: onursurhan
"""

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
from keras.utils import to_categorical 
from tensorflow.keras import regularizers

X = np.load('X_step_up&down.npy')
Y = np.load('y_step_up&down.npy')
Y = Y - 1 

Y = to_categorical(Y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Sequential()
#        model.add(LSTM(self.first_layer_neurons, input_dim=self.input_dimension , dropout_U=0.3)) ## I don't understand why droupout_U is used
model.add(LSTM(150, input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(100, kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2),
    bias_regularizer=regularizers.l2(1e-2),
    activity_regularizer=regularizers.l2(1e-2)))
#model.add(Dropout(0.2))
model.add(Dense(50, kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2),
    bias_regularizer=regularizers.l2(1e-2),
    activity_regularizer=regularizers.l2(1e-2)))
#model.add(Dropout(0.4))
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


# clf=KerasClassifier(build_fn=model, 
#                             epochs=100,
#                             batch_size = 32,
#                             verbose=3)

# clf.fit(X_train, y_train)
# y_predicted = clf.predict(X_test)
# model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=3)
# y_predicted = (model.predict(X_test) > 0.5).astype("int32")
# y_actual = y_test

history = model.fit(X, Y, validation_split=0.25, epochs=150)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
# cvscores = []
# for train, test in kfold.split(X, Y):
#   # create model
# 	model = Sequential()
# 	model.add(LSTM(20, input_shape=(X.shape[1],X.shape[2])))
# 	model.add(Dense(30))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# Fit the model
# 	model.fit(X[train], Y[train], epochs=100, batch_size=16, verbose=3)
# 	# evaluate the model
# 	scores = model.evaluate(X[test], Y[test], verbose=2)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


predictions = model.predict(X_test)
y_predicted = np.zeros(len(X_test))
for i in range(len(X_test)):
    if predictions[i,0]<predictions[i,1]:
        y_predicted[i]=1
        






