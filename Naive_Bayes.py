# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:20:44 2021

@author: ginag
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt

train_image = gzip.open("train-images-idx3-ubyte.gz", 'r')

image_size = 28
num_images = 8
train_image.read(16)

buf = train_image.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

# print image
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()