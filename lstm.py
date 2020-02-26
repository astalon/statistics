#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:21:30 2020
x
@author: astalon
"""


import pandas as pd
import numpy as np
import stats
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_excel("eurodollar_new.xls", header=0)
dates = matplotlib.dates.date2num(df.iloc[:, 0])
df.set_index("Date", inplace=True)

df.loc[:, df.columns != 'EURUSD'] = df.loc[:, df.columns != 'EURUSD']/100

#df['3mspread'] = df['US3M'] - df['EURO3M']
#df = df.drop(['US3M', 'EURO3M'], axis=1)

#df = df.join(df.diff(), rsuffix='_1mchange').dropna()
target = df['EURUSD'].shift(-1).dropna()
df = df[:-1]
dates = dates[:-1]

features = df.shape[1]

training_index = 150
look_back = 5

train_features = df.iloc[:150, ]
train_labels = target.iloc[:150, ]

test_features = df.iloc[150:, ]
test_labels = target.iloc[150:, ].to_numpy()

lm = stats.linreg(train_features, train_labels)
lm.fit(shuffle=False)
lm_predictions = []
dates_plot = []

# nn = stats.nn(train_features, train_labels, 1, [6], epochs=15, optimization_runs=10)
# nn.train_network()
# nn_predictions = nn.predict(test_features)

for i in range(len(test_features)):
	data = test_features.iloc[i, :].to_numpy().reshape((1, features))
	predicted = lm.predict(data)
	lm_predictions.append(predicted)
	dates_plot.append(dates[training_index+i])
	
	
errors = (test_labels - lm_predictions)
errors = np.square(errors)
fig, ax = plt.subplots(2,1)
ax[0].plot_date(dates_plot, lm_predictions, label='model predictions', linestyle='solid', marker=None)
ax[0].plot_date(dates_plot, test_labels, label='EUR/USD', linestyle='solid', marker=None)
ax[0].set_title('Out of sample EUR/USD 1month forecasts')
ax[0].legend()
ax[1].plot_date(dates_plot, errors, label='squared errors', linestyle='solid', marker=None)
ax[1].legend()
plt.show()

# features = []
# labels= []

# features_test = []
# labels_test = []

# for j in range(look_back, training_index):
#     tmp = df.iloc[j-look_back:j,].to_numpy()
#     features.append(tmp)
#     labels.append(target.iloc[j,])
 	
# for i in range(training_index+look_back, len(df)):
#     tmp = df.iloc[i-look_back:i,].to_numpy()
#     features_test.append(tmp)
#     labels_test.append(target.iloc[i,])


# #features = pd.DataFrame(features).to_numpy()

# features = np.array(features)
# labels = np.array(labels)

# features_test = np.array(features_test)
# labels_test = np.array(labels_test)

# lstm = stats.lstm(features, labels, 1, [10, 5], optimization_runs=10, history=look_back, dropout=0.4)
# lstm.fit()
# lstm_predictions = lstm.predict(features_test)





# # Kolla validation
# Make LSTM in TF
# Make minimum variance
# SVC köp/sälj eurusd
