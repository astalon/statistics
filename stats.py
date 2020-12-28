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

# General purpose MLP, can be used for classification problems
class nn:
    def __init__(self, training_input, training_labels, nr_outputs, hidden_layer_nodes,activation_function = 'elu',learning_rate = 0.001, scale_output = True, epochs=5, optimization_runs=5, scale_range=(0, 1), scale=True, classification_classes=None, classifier = False):
        self.scale = scale
        self.scale_output = scale_output
        if scale:
            self.scaler = MinMaxScaler(feature_range=scale_range).fit(training_input)
            self.input_data = self.scaler.transform(training_input).astype(np.float32)

        else:
            self.input_data = training_input    


        if scale_output:
            self.output_scaler = MinMaxScaler(feature_range=scale_range).fit(training_labels)
            self.output_data = self.output_scaler.transform(training_labels).astype(np.float32)

        else:
            self.output_data = training_labels

        if classifier:
            self.output_data = output_data

        if activation_function == 'sigmoid':
            self.activation_function = tf.keras.activations.sigmoid
        elif activation_function == 'tanh':
            self.activation_function = tf.keras.activations.tanh
        elif activation_function == 'relu':
            self.activation_function = tf.keras.activations.relu
        elif activation_function == 'elu':
            self.activation_function = tf.keras.activations.elu
        elif activation_function == 'exp':
            self.activation_function = tf.keras.activations.exponential
        else:
            print("No known activation function was passed (", activation_function, "), using default of elu")
            self.activation_function = tf.keras.activations.elu

        self.lr = learning_rate
        self.nr_features = np.shape(self.input_data)[1]
        self.nr_training_samples = np.shape(self.input_data)[0]
        self.nr_outputs = nr_outputs
        self.nr_classes = classification_classes
        self.nr_hidden_layers = len(hidden_layer_nodes)

        self.classification = classifier
        
        self.hidden_layer_nodes = hidden_layer_nodes
        self.hidden_layers = [None]*(self.nr_hidden_layers+1)
        self.final_hidden_layers = [None]*(self.nr_hidden_layers+1)
        
        self.epochs = epochs
        self.optimization_tries = optimization_runs


    # Fits the data according to the data provided
    def fit(self):

        best = 10000000
        #Weights are initialized randomly, loop and see where optimizer performs the best
        for opt_iteration in range(self.optimization_tries):
            tf.compat.v1.disable_eager_execution()
            x = tf.compat.v1.placeholder(tf.float32, [1, self.nr_features])
            y = tf.compat.v1.placeholder(tf.float32, [None, self.nr_outputs])

            predictions = self.__predict_train(x) # nr_samples*nr_outputs sized matrix, produced through original non optimized network

            # Define the cost function 
            if self.classification:
                if self.nr_classes > 2:
                    cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=predictions, logits=y,))
                else:
                    cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=predictions, logits=y,))
            else:
                cost = tf.reduce_sum(tf.square(y-predictions))

            # Choose what optimizer should be used and how 
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.lr).minimize(cost)
            
            with tf.compat.v1.Session() as sess:
                
                sess.run(tf.compat.v1.global_variables_initializer())

                #Iterate several times over the training data
                for epoch in range(self.epochs):
                    epoch_loss = 0
                    for sample in range(self.nr_training_samples):
                        x_tmp = self.input_data[sample].reshape((1, self.nr_features))
                        y_tmp = self.output_data[sample].reshape((1, self.nr_outputs))

                        _, c = sess.run([optimizer, cost], feed_dict={x: x_tmp, y: y_tmp})
                        epoch_loss += c
                    print("Epoch", epoch, "out of", self.epochs, "epoch loss:", epoch_loss)

                print("Optimization run", opt_iteration+1,"out of", self.optimization_tries,"loss", epoch_loss)
                if epoch_loss < best:
                    best = epoch_loss
                    for i in range(self.nr_hidden_layers+1):
                        self.final_hidden_layers[i]= {'weights': sess.run(self.hidden_layers[i]['weights']),
                                                      'biases': sess.run(self.hidden_layers[i]['biases'])}

    def __predict_train(self, data):
        # Initialize the network, defined by its hidden layers and their number of nodes as well as weights
        # Special care taken with input and output layer

        # initialize all the weights for the network
        for hidden_layer in range(self.nr_hidden_layers+1):
            if(hidden_layer == 0):
                self.hidden_layers[hidden_layer] = {'weights': tf.Variable(tf.random.normal([self.nr_features, self.hidden_layer_nodes[hidden_layer]])*np.sqrt(2/(self.nr_features))),
                                                    'biases': tf.Variable(tf.zeros([1, self.hidden_layer_nodes[hidden_layer]]))}
            elif(hidden_layer == self.nr_hidden_layers):
                self.hidden_layers[hidden_layer] = {'weights': tf.Variable(tf.random.normal([self.hidden_layer_nodes[hidden_layer-1], self.nr_outputs])*np.sqrt(2/self.hidden_layer_nodes[hidden_layer-1])),
                                                    'biases': tf.Variable(tf.zeros([1, self.nr_outputs]))}
            else:
                self.hidden_layers[hidden_layer] = {'weights': tf.Variable(tf.random.normal([self.hidden_layer_nodes[hidden_layer-1], self.hidden_layer_nodes[hidden_layer]])*np.sqrt(2/self.hidden_layer_nodes[hidden_layer-1])),
                                                    'biases': tf.Variable(tf.zeros([1, self.hidden_layer_nodes[hidden_layer]]))}

        layers = [None]*(self.nr_hidden_layers)
        print(self.hidden_layers[0]['biases'].dtype)

        # Feed the data through the network, creating a nr_samples*nr_outputs sized matrix
        layers[0] = tf.add(tf.matmul(data, self.hidden_layers[0]['weights']), self.hidden_layers[0]['biases'])
        layers[0] = self.activation_function(layers[0])

        for layer in range(1,self.nr_hidden_layers):

            layers[layer] = tf.add(tf.matmul(layers[layer-1], self.hidden_layers[layer]['weights']), self.hidden_layers[layer]['biases'])
            layers[layer] = self.activation_function(layers[layer])

        ret = tf.add(tf.matmul(layers[-1], self.hidden_layers[-1]['weights']), self.hidden_layers[-1]['biases'])
        
        #Different activation functions depending on what kind of problem we are doing
        if self.classification:
            if self.nr_classes > 2:
                ret = tf.keras.activations.softmax(ret)
            else:
                ret = tf.keras.activations.sigmoid(ret)
        # else:
        #     ret = tf.keras.activations.linear(ret)   
        return ret

    def predict(self, input_data):
        if self.scale:
            data = self.scaler.transform(input_data).astype(np.float32)
        else:
            data = input_data.astype(np.float32)
        layers = [None]*(self.nr_hidden_layers)
        
        #Feed forward
        for layer in range(self.nr_hidden_layers):
            if(layer == 0):
                layers[layer] = tf.add(tf.matmul(
                    data, self.final_hidden_layers[layer]['weights']), self.final_hidden_layers[layer]['biases'])
            else:
                layers[layer] = tf.add(tf.matmul(
                    layers[layer-1], self.final_hidden_layers[layer]['weights']), self.final_hidden_layers[layer]['biases'])

            layers[layer] = self.activation_function(layers[layer])

        output = tf.add(tf.matmul(
            layers[-1], self.final_hidden_layers[-1]['weights']), self.final_hidden_layers[-1]['biases'])
        
        #Let er rip
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            ret = sess.run(output)
            if self.classification:
                if self.nr_classes > 2:
                    ret = tf.keras.activations.softmax(ret)
                else:
                    ret = tf.convert_to_tensor(ret, np.float32)
                    ret = tf.keras.activations.sigmoid(ret)
            # else:
            #     ret = tf.keras.activations.linear(ret)    
            ret = tf.convert_to_tensor(ret, np.float32)
            ret = sess.run(ret)

        if self.scale_output:
            return self.output_scaler.inverse_transform(ret).astype(np.float32)
        return ret


class lstm():
    def __init__(self, training_input, training_labels, nr_outputs, hidden_layer_nodes, epochs=5,
                optimization_runs=1, scale_range=(0, 1), activation_function='softmax', history=3, dropout=0.2):
        self.output_data = training_labels
        #self.scaler = MinMaxScaler(feature_range=scale_range).fit(training_input)
        #self.input_data = self.scaler.transform(training_input).astype(np.float32)
        self.input_data = training_input
        self.nr_features = np.shape(self.input_data)[-1]
        self.nr_training_samples = np.shape(self.input_data)[0]
        self.nr_classes = nr_outputs
        self.nr_hidden_layers = len(hidden_layer_nodes)
        self.epochs = epochs
        self.hidden_layer_nodes = hidden_layer_nodes
        self.optimization_tries = optimization_runs
        self.activation_function = activation_function
        self.history=history
        self.dropout = dropout

    def fit(self):
        best = 10000000
        for i in range(self.optimization_tries):
            self.model = Sequential()

            for i in range(self.nr_hidden_layers):
                if i==0:
                    self.model.add(LSTM(units=self.hidden_layer_nodes[i],activation = self.activation_function, kernel_initializer='random_normal',
                        return_sequences=True,input_shape=(self.history, self.nr_features)))
                    self.model.add(Dropout(self.dropout))
                elif i==(self.nr_hidden_layers-1):
                    self.model.add(LSTM(units=self.hidden_layer_nodes[i], activation=self.activation_function,
                        kernel_initializer='random_normal'))
                    self.model.add(Dropout(self.dropout))                
                else:
                    self.model.add(LSTM(units=self.hidden_layer_nodes[i], activation=self.activation_function,
                        kernel_initializer='random_normal', return_sequences=True))
                    self.model.add(Dropout(self.dropout))

            self.model.add(Dense(units=self.nr_classes))
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            self.model.fit(self.input_data, self.output_data, epochs=self.epochs)
            model_loss = self.model.evaluate(self.input_data, self.output_data)[0]
            if model_loss < best:
                best = model_loss
                self.final_model = self.model


    def predict(self, data):
        #data = self.scaler.transform(data) #Kanske denna behöver castas till np float 32?
        prediction = self.model.predict(data, batch_size = 1)
        return prediction

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
    def __init__(self, input_data, output_data, hidden_layer_nodes, classification = False, layer_activation = 'relu ', output_activation='linear', epochs=60, batch_size = 10, scale_range = (-1, 1)):

        self.hidden_layer_nodes = hidden_layer_nodes
        self.layer_activation = layer_activation if layer_activation in ['sigmoid', 'tanh', 'relu', 'elu', 'exponential', 'relu', 'linear', 'softmax', 'softplus', 'softsign'] else 'relu'
        self.output_activation = output_activation if output_activation in ['sigmoid', 'tanh', 'relu', 'elu', 'exponential', 'relu', 'linear', 'softmax', 'softplus', 'softsign'] else 'linear'

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

        for i in range(1,len(hidden_layer_nodes)-1):
            self.model.add(Dense(hidden_layer_nodes[i], activation = self.layer_activation, kernel_initializer = initializers.he_uniform))

        self.model.add(Dense(output_data.shape[1], activation=self.output_activation, kernel_initializer = initializers.he_uniform))

        self.model.compile(loss=self.loss, optimizer = 'adam', metrics=[self.metric])
        self.model.fit(input_data, output_data, epochs = epochs, batch_size = batch_size, verbose=2)

        # _, accuracy = self.model.evaluate(input_data, output_data)
        # print("Mean average error: ", accuracy)

    def predict(self, data):
        data = self.input_scaler.transform(data)

        unscaled_return = self.model.predict(data)

        return unscaled_return if self.classification else self.output_scaler.inverse_transform(unscaled_return)




class cnn:
    None

class rnn:
    None

class lstm:
    None

'''
Test whether we should try and use activation function in the last layer also? Linear?

'''