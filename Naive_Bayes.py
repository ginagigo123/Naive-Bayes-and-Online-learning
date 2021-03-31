# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:20:44 2021

@author: ginag
content:
    Naive bayes classifier and online learning.
    Input : 
        1. Training image data from MNIST
        
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import math

def training_images():
    with gzip.open("train-images-idx3-ubyte.gz", 'r') as f:
        """
            int.from_bytes: bytes -> integer
            byteorder = 'big' -> big endian
        """
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_image = int.from_bytes(f.read(4), 'big')
        # thrid 4 bytes = row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes = col count
        col_count = int.from_bytes(f.read(4), 'big')
        
        # rest = image pixel data, each pixel is stored as unsigned byte
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_image, row_count, col_count)
        return images
    
def training_label():
    with gzip.open("train-labels-idx1-ubyte.gz", 'r') as f:
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_label = int.from_bytes(f.read(4), 'big')
        
        # rest = label data, each label is stored as unsigned byte
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8).reshape(num_label, 1)
        return labels

def test_images():
    with gzip.open("t10k-images-idx3-ubyte.gz", 'r') as f:
        """
            int.from_bytes: bytes -> integer
            byteorder = 'big' -> big endian
        """
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_image = int.from_bytes(f.read(4), 'big')
        # thrid 4 bytes = row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes = col count
        col_count = int.from_bytes(f.read(4), 'big')
        
        # rest = image pixel data, each pixel is stored as unsigned byte
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_image, row_count, col_count)
        return images
    
def test_label():
    with gzip.open("t10k-labels-idx1-ubyte.gz", 'r') as f:
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_label = int.from_bytes(f.read(4), 'big')
        
        # rest = label data, each label is stored as unsigned byte
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8).reshape(num_label, 1)
        return labels
    
    
X_train = training_images()
y_train = training_label()

X_test = test_images()
y_test = test_label()

# Discrete mode
# to make the X_train writeable 
X_train_bin = X_train.copy()
X_train_bin.flags
    
X_train_bin = X_train_bin // 8
        
# check bin
for i in range(28):
    for j in range(28):
        print(X_train_bin[0][i][j], end=" ")
    print()
            

# probablity
# instead of np.zeros(), use np.ones() to avoid empty bin => give a peudocont
pro = np.ones(shape=(10, 32, 28, 28))
prior = np.zeros(10)

# fill in the likelihood
for k in trange( len(X_train_bin) ):
    prior[ y_train[k][0] ] += 1.0
    for i in range(28):
        for j in range(28):
            #print(y_train[k][0], X_train_bin[k][i][j], i, j)
            pro[ y_train[k][0] ][ X_train_bin[k][i][j] ][i][j] += 1.0

# test the probability
for i in range(10):
    pro[i] = pro[i] / prior[i]
prior = prior / 60000.0

"""
tmp = 0
for i in range(28):
    for j in range(28):
        tmp += pro[0][0][i][j]
        print(pro[0][0][i][j], end=' ')
    print()
"""

# to make X_test writeable
X_test_bin = X_test.copy()
X_test_bin = X_test_bin // 8

posterior_log = np.zeros(shape=(len(X_test_bin), 10))
y_predict = np.array( [0] * len(X_test_bin) )

for k in trange(len(X_test_bin)):
    #print("\nPosterior (in log scale):")
    max_post_type = 0
    marginal = 0.0
    for num in range(10):
        # prior addition
        posterior_log[k][num] += math.log(prior[num])
        # point addition
        for i in range(28):
            for j in range(28):
                bins = X_test_bin[k][i][j]
                #print(num, bins, i, j, "probability : ", pro[num][bins][i][j])
                posterior_log[k][num] += math.log( pro[num][bins][i][j] )
    
        if (posterior_log[k][num] > posterior_log[k][max_post_type] ):
            max_post_type = num
        # minus marginal
        marginal += posterior_log[k][num]
    
    # normalize the posterior
    for num in range(10):
        posterior_log[k][num] /= marginal
        #print(num, ":", posterior_log[k][num])
    y_predict[k] = max_post_type
    #print("Prediction:", max_post_type, "Ans:", y_test[k][0])


def error_rate(predict, true):
    error = 0
    for i in range(len(predict)):
        if predict[i] != true[i]:
            error += 1
    return error / len(predict)

print("error rate: ", error_rate(y_predict, y_test))
    
# show image of the first word
"""
image = np.asarray(X_train[0]).squeeze()
plt.imshow(image)
plt.show()
"""