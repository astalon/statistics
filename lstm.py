#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:21:30 2020
x
@author: Erik Lagerström
"""


import pandas as pd
import numpy as np
import stats_new as stats
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_excel("eurodollar_new.xls", header=0)
dates = matplotlib.dates.date2num(df.iloc[:, 0])
df.set_index("Date", inplace=True)

df.loc[:, df.columns != 'EURUSD'] = df.loc[:, df.columns != 'EURUSD']/100

df['3mspread'] = df['US3M'] - df['EURO3M']
df = df.drop(['US3M', 'EURO3M'], axis=1)

target = df['EURUSD'].shift(-1).dropna()
target.drop(df.index[0], inplace = True)

#df = df.join(df.diff(), rsuffix='_1mchange').dropna()
df.drop(df.index[len(df)-1], inplace = True)
df.drop(df.index[0], inplace = True)

data_points = len(df)

df = df.to_numpy()
target = target.to_numpy()

#df = np.random.rand(data_points,1)*10 - 1
#target = (np.sin(df))/2.5 + (np.random.randn(data_points,1)/20)

training_part = 0.7
end_index = np.int(np.floor(training_part*data_points))

features = df.shape[1]

train_features = df[:end_index, :]
train_labels = target[:end_index, ].reshape(-1,1)

test_features = df[end_index:, :]
test_labels = target[end_index:, ].reshape(-1, 1)

lookback = 2

test_features_nn = df[(end_index-lookback):, :]

#test_labels = target[end_index:, ].reshape(-1,1)
x = np.linspace(1, data_points-end_index, data_points-end_index)

#dates_plot = dates[150:, ]

# nn_predictions = nn.predict(test_features)

nn = stats.lstm(train_features, train_labels, [128,128], epochs = 30, layer_activation = "sigmoid", look_back = lookback, scale_range = (0, 1))
nn_predictions = nn.predict(test_features_nn)

errors_nn = np.subtract(test_labels,nn_predictions)
errors_nn = np.square(errors_nn)

print("Mean squared error ANN: ", np.mean(errors_nn))


lm = stats.linreg(train_features, train_labels)
lm.fit(shuffle=False)
lm_predictions = []
dates_plot = []

for i in range(len(test_features)):
	inp = test_features[i,:].reshape(-1, features)
	predicted = lm.predict(inp)
	lm_predictions.append(predicted[0])
	#dates_plot.append(dates[training_index+i])
	

lm_predictions = np.array(lm_predictions).reshape(-1, 1)
#lm_predictions = np.asarray(lm_predictions, dtype=np.float32).reshape(-1, 1)

errors = np.subtract(test_labels,lm_predictions)
errors = np.square(errors)
print("Mean squared error LM: ", np.mean(errors))

#plt.plot(x, test_labels, 'r', linewidth=1, label='sin')
plt.plot(x, errors_nn, 'b', linewidth=1, label='nn')
plt.plot(x, errors, 'g', linewidth=1, label='lm')
plt.legend()
plt.show()

# Borde bara vara en rad som försvinner när vi sätter look_back = 2? Gör om så att den shiftar för look_back = 1
# Ändra test_labels likadant
