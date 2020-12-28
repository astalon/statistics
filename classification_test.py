
import pandas as pd
import numpy as np
import stats_new as stats

from numpy import loadtxt

dataset = loadtxt('classifi.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

data_points = X.shape[0]

training_part = 0.8
end_index = np.int(np.floor(training_part*data_points))

features = 1

train_features = X[:end_index, ]
train_labels = Y[:end_index, ].reshape(-1, 1)

test_features = X[end_index:, ]

test_labels = Y[end_index:, ].reshape(-1, 1)
test_data_points = len(test_labels)

nn = stats.nn_new(train_features, train_labels, [50,50,50], epochs=30, classification = True)
nn_predictions = nn.predict(test_features)

test_labels = test_labels.reshape(-1,)
correct = 0

for i in range(test_data_points):
	correct = correct + 1 if nn_predictions[i] == test_labels[i] else correct

test_accuracy = 100*correct/test_data_points

print("%.2f"% test_accuracy)