#!/usr/bin/python -tt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

testdata = pd.read_csv('mnist-test-labeled.csv')
traindata = pd.read_csv('train.csv')

#Print the average digit images for the two datasets
def extract_digit(n,dset):
    dic = {}
    for index, row in dset.iterrows():
        key = row['label']
        if key not in dic:
            dic[key] = []
        dic[key].append(row)
    img = pd.DataFrame(dic[n])
    mean = img.describe().loc['mean'].values
    return mean

def transfer(digit_pixel):
    img_matrix = np.zeros((28,28))
    for i in range(0,27):
        for j in range (0,27):
            index = i * 28 + j
            img_matrix[i][j] =digit_pixel[index+1]
    return img_matrix


def displayTrain(digit,dset):
    mean = extract_digit(digit,dset)
    img = transfer(mean)
    plt.imshow(img,cmap=cm.binary)
    fname = 'trainAvg' + str(digit)+'.png'
    plt.savefig(fname)

def displayTest(digit,dset):
    mean = extract_digit(digit,dset)
    img = transfer(mean)
    plt.imshow(img,cmap=cm.binary)
    fname = 'testAvg' + str(digit)+'.png'
    plt.savefig(fname)
    
for i in range(10):
  displayTrain(i,traindata)
  displayTest(i,testdata)