# Kaggle
This is my submission to the Kaggle's Digit Recognizer Competition, which asks for a machine learning algorithm to classify images of
handwritten single digits. More info here: https://www.kaggle.com/c/digit-recognizer

This repository includes Includes the submission, code, as well as the code used to test different models. 2 and 3 layer neural network models were tested. The 3-layer model has the better accuracy than the 2-layer on the training dataset. However, it has smaller accuracy score on the test set, which suggests it is over-fitting and thus, has the inferior generalization performance. This is confirmed by comparing the 5-fold cross-validation accuracy scores—the 2-layer model has better scores.
