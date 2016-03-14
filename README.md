# Kaggle
This is my submission to the Kaggle's Digit Recognizer Competition, which asks for a machine learning algorithm to classify images of
handwritten single digits. More info here: https://www.kaggle.com/c/digit-recognizer

This repository includes the submission, code, as well as the code used to test other models. 2 and 3 layer neural network models were tested. The 3-layer models have the better accuracy than the 2-layer ones on the training dataset. However, they have lower accuracy scores on the test set, which suggests ]over-fitting and thus, an inferior generalization performance. This is confirmed by comparing the 5-fold cross-validation accuracy scores—the 2-layer models have better scores.

The final model submitted had an accuracy score of 93.7% on the publicly available dataset. The model was a two layer NN, with 392 hidden layer units, regularization of 0.01, and Sigmoid, then Softmax activation functions.
