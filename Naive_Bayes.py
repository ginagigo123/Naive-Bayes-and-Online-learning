# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:20:44 2021

@author: ginag
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt

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
    
X_train = training_images()
y_train = training_label()

# show image of the first word
image = np.asarray(X_train[0]).squeeze()
plt.imshow(image)
plt.show()
