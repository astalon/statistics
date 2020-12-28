#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine learning and statistics module, to allow for user to not specify nitty gritty details
Regression, SVM, neural nets, factor models. Time series?
Expects training and testing data to be of numpy array forms
@author: Erik Lagerström
"""

import numpy as np
import sklearn
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import initializers

import matplotlib.pyplot as plt


# Support vector classifier, used for binary predictions
class svc:
    # Labels used supposed to be of format 1 and -1
    def __init__(self, training_input, training_output, kernel='rbf', degree=3, C=1, scale_range=(0, 1), scale=True):
        self.output_data = training_output
        self.feature_range = scale_range
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.scale = scale
        if scale:
            self.scaler = MinMaxScaler(
                feature_range=scale_range).fit(training_input)
            self.input_data = self.scaler.transform(training_input)
        else:
            self.input_data = training_input

    # Does a pseudo split with training data for hint of accuracy
    # Then fits model to all data
    def fit(self, shuffle=True):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            self.input_data, self.output_data, test_size=0.2, shuffle=shuffle)
        
        clf = sklearn.svm.SVC(C=self.C, kernel=self.kernel, degree=self.degree)
        clf.fit(x_train, y_train)

        print("Artificial R^2: ", clf.score(x_test, y_test))
        print("Fitting model to all of input data")

        self.clf = sklearn.svm.SVC(
            C=self.C, kernel=self.kernel, degree=self.degree)
        self.clf.fit(self.input_data, self.output_data)
    
    # Predicts data using the model currently fitted
    def predict(self, data):
        if self.scale:
            scaled = self.scaler.transform(data)[0]
            calced = self.clf.predict(scaled.reshape(1, -1))
            return calced
        else:
            return self.clf.predict(data)

class linreg():
    def __init__(self, training_input, training_output, scale_range=(-1, 1), epochs=1):
        self.output_data = training_output
        self.data = training_input
        for i in range(epochs-1):
            self.data = np.append(self.data, training_input, axis=0)
            self.output_data = np.append(
                self.output_data, training_output, axis=0)

        self.feature_range = scale_range
        self.scaler = MinMaxScaler(feature_range=scale_range).fit(self.data)
        self.input_data = self.scaler.transform(self.data)

    def fit(self, shuffle=True, njobs=1):
        x_train, x_test, y_train, y_test = train_test_split(
            self.input_data, self.output_data, test_size=0.2, shuffle=shuffle)
        clf = sklearn.linear_model.LinearRegression(n_jobs=njobs)
        clf.fit(x_train, y_train)
        print("Artificial R^2: ", clf.score(x_test, y_test))
        self.clf = sklearn.linear_model.LinearRegression(n_jobs=njobs)
        self.clf.fit(self.input_data, self.output_data)

    def predict(self, data):
        scaled = self.scaler.transform(data)
        predicted = self.clf.predict(scaled)
        return predicted[0]



class nn_new():
    def __init__(self, input_data, output_data, hidden_layer_nodes, classification = False, layer_activation = 'tanh', output_activation='linear', epochs=10, batch_size = 3, scale_range = (-1, 1)):

        self.hidden_layer_nodes = hidden_layer_nodes

        activations = ['sigmoid', 'tanh', 'relu', 'elu', 'exponential', 'relu', 'linear', 'softmax', 'softplus', 'softsign']
        self.layer_activation = layer_activation if layer_activation in activations else 'relu'
        self.output_activation = output_activation if output_activation in activations else 'linear'

        self.classification = classification

        self.input_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=scale_range)
        self.input_scaler.fit(input_data)
        input_data = self.input_scaler.transform(input_data)

        self.output_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=scale_range)
        self.output_scaler.fit(output_data)
        
        if classification:
            self.metric = 'accuracy'

            if output_data.shape[1]>1:
                # ASSUMES labels are one hot encoded
                self.output_activation = "softmax"
                self.loss = "categorical_crossentropy"
            else:
                # ASSUMES labels of 0 or 1
                self.output_activation = "sigmoid"
                self.loss = 'binary_crossentropy'

        else:
            output_data = self.output_scaler.transform(output_data)
            self.metric = "mae"
            self.loss = "mse"

        self.model = Sequential()

        self.model.add(Dense(hidden_layer_nodes[0], input_dim = input_data.shape[1],  kernel_initializer = initializers.he_uniform, activation = self.layer_activation))

        for i in range(1,len(hidden_layer_nodes)):
            self.model.add(Dense(hidden_layer_nodes[i], activation = self.layer_activation, kernel_initializer = initializers.he_uniform))

        self.model.add(Dense(output_data.shape[1], activation=self.output_activation, kernel_initializer = initializers.he_uniform))

        self.model.compile(loss=self.loss, optimizer = 'adam', metrics=[self.metric])
        self.model.fit(input_data, output_data, epochs = epochs, batch_size = batch_size, verbose=2)

        # _, accuracy = self.model.evaluate(input_data, output_data)
        # print("Mean average error: ", accuracy)

    def predict(self, data, return_one_hot = True):
        data = self.input_scaler.transform(data)

        unscaled_return = self.model.predict(data)

        if self.classification:

            # Transform the returned values to one-hot encoded arrays, only done for classification problems
            if return_one_hot:
                if len(unscaled_return[0]) > 1:
                    for prediction in unscaled_return[:]:
                        max_element = max(prediction)
                        prediction[:] = [0 if j<max_element else 1 for j in prediction]

                    return unscaled_return

                else:
                    unscaled_return = unscaled_return.reshape(-1,)
                    unscaled_return[:] = [1 if i>0.5 else 0 for i in unscaled_return]
                    return unscaled_return.reshape(-1,1)

            else:
                return unscaled_return

        return self.output_scaler.inverse_transform(unscaled_return)




class cnn:
    None

class rnn:
    None

def transform_data(input_data, output_data, look_back=1):
    X, Y = [], []

    for i in range(input_data.shape[0]-look_back):
        X.append(input_data[i:(i+look_back), :])
        Y.append(output_data[i+look_back-1, :])

    return np.array(X), np.array(Y)

# Expects input data in 2D format independent of look-back period. Transforming done automatically
class lstm:

    def __init__(self, input_data, output_data, hidden_layer_nodes, classification = False, layer_activation = 'relu', output_activation='linear', dropout=0.0, epochs=30, batch_size = 3, scale_range = (-1, 1), look_back = 1):

        self.hidden_layer_nodes = hidden_layer_nodes
        self.look_back = look_back

        possible_activations = ['sigmoid', 'tanh', 'relu', 'elu', 'exponential', 'relu', 'linear', 'softmax', 'softplus', 'softsign']

        layer_activation = layer_activation if layer_activation in possible_activations else 'relu'
        output_activation = output_activation if output_activation in possible_activations else 'linear'

        self.classification = classification

        self.input_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=scale_range)
        self.input_scaler.fit(input_data)
        input_data = self.input_scaler.transform(input_data)

        if look_back >1:
            input_data, output_data = transform_data(input_data, output_data, look_back)  
            #output_data = np.delete(output_data, [i for i in range(look_back-1)], 0)
            
        print(input_data.shape)
            
        self.output_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=scale_range)
        self.output_scaler.fit(output_data)
        
        if classification:
            self.metric = 'accuracy'

            if output_data.shape[1]>1:
                # ASSUMES labels are one hot encoded
                self.output_activation = "softmax"
                self.loss = "categorical_crossentropy"
            else:
                # ASSUMES labels of 0 or 1
                self.output_activation = "sigmoid"
                self.loss = 'binary_crossentropy'

        else:
            output_data = self.output_scaler.transform(output_data)
            self.metric = "mae"
            self.loss = "mse"

        self.model = Sequential()

        self.model.add(LSTM(hidden_layer_nodes[0], input_shape = (input_data.shape[1], input_data.shape[2]), return_sequences = len(hidden_layer_nodes)>1,  kernel_initializer = initializers.he_uniform, activation = layer_activation))

        #Might want to fiddle with dropout and dropout_recurrent for LSTM layer, use say 0.1-0.2
        for i in range(1,len(hidden_layer_nodes)):
            self.model.add(LSTM(hidden_layer_nodes[i], return_sequences = (i != len(hidden_layer_nodes)-1), kernel_initializer = initializers.he_uniform, activation = layer_activation))

        self.model.add(Dense(output_data.shape[1], activation=output_activation, kernel_initializer = initializers.he_uniform))

        self.model.compile(loss=self.loss, optimizer = 'adam', metrics=[self.metric])
        self.model.fit(input_data, output_data, epochs = epochs, batch_size = batch_size, verbose=2)

        # _, accuracy = self.model.evaluate(input_data, output_data)
        # print("Mean average error: ", accuracy)

    def predict(self, data, return_one_hot = True):
        data = self.input_scaler.transform(data)
        
        data, _ = transform_data(data, data, self.look_back)
        #np.reshape(data, (data.shape[0], self.look_back, data.shape[1]))

        unscaled_return = self.model.predict(data)

        if self.classification:

            # Transform the returned values to one-hot encoded arrays, only done for classification problems
            if return_one_hot:
                if len(unscaled_return[0]) > 1:
                    for prediction in unscaled_return[:]:
                        max_element = max(prediction)
                        prediction[:] = [0 if j<max_element else 1 for j in prediction]

                    return unscaled_return

                else:
                    unscaled_return = unscaled_return.reshape(-1,)
                    unscaled_return[:] = [1 if i>0.5 else 0 for i in unscaled_return]
                    return unscaled_return.reshape(-1,1)

            else:
                return unscaled_return

        return self.output_scaler.inverse_transform(unscaled_return)

'''
Test whether we should try and use activation function in the last layer also? Linear?

'''