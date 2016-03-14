#!/usr/bin/python -tt

import pandas as pd
import numpy as np
from sknn.mlp  import Classifier, Layer

#Load Data
testdata = pd.read_csv('mnist-test-labeled.csv')
traindata = pd.read_csv('train.csv')

target = traindata[[0]].values.ravel()
target_t = testdata[[0]].values.ravel()
y_train = target.astype(np.uint8)
y_test = target_t.astype(np.uint8)

train = traindata.iloc[:,1:].values
test = testdata.iloc[:,1:].values
X_train = np.array(train).astype(np.uint8)
X_test = np.array(test).astype(np.uint8)

# Fit a 2-Layer Neural Network
nn = Classifier(layers=[Layer("Sigmoid", units=392),Layer("Softmax")],learning_rate=0.001, n_iter=25)
nn.fit(X_train, y_train)

#Predict using the fitted model
pred_train = nn.predict(X_train)
pred_test = nn.predict(X_test)
cErrTest = metrics.accuracy_score(y_test, pred_test)
print cErrTest

pred_test.to_csv('ks_test.csv')