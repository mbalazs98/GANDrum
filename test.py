import os
import pypianoroll as pp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
				
				
model=tf.keras.Sequential()
			
model.add(layers.Conv2D(filters=128,kernel_size=1,strides=1,input_shape=[8,12,128]))
model.add(layers.LeakyReLU())

model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
model.add(layers.LeakyReLU())
			
model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
model.add(layers.LeakyReLU())
			
model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
model.add(layers.LeakyReLU())

model.add(layers.Flatten())
model.add(layers.Dense(1))
model.summary()