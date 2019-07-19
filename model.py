import pypianoroll as pp
import os
import numpy as np
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers

class GANDrum(object):
	def __init__(self,buffer_size=60000,batch_size=72):
		self.train_files=[]
		self.buffer_size=buffer_size
		self.batch_size=batch_size
		
	def load_data():
		entries= s.listdir(r'D:/midi_d/')
		for entry in entries:
			x=pp.parse(entry)
			pp.save(r'D:\midi_d\your_training_data',x,compressed=False)
			data=np.load(r'D:\midi_d\your_training_data.npz')
			mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr']), shape=(1536, 128)).todense()
			self.train_files.append(mtx)
		
		self.train_labels=np.load(r'D:\midi_d_label\labels.npy')
		
		train_dataset=tf.data.Dataset.from_tensor_slices(self.train_files).shuffle(self.buffer_size).batch(self.batch_size)
		
	def make_generator_model():
		model=tf.keras.Sequential()
		model.add(layers.Dense(7*7*256,use_bias=False,input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Reshape((7,7,256)))
		assert model.output_shape==(None,7,7,256)

		model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))
		assert model.output_shape==(None,7,7,128)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
		assert model.output_shape==(None,14,14,64)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))
		assert model.output_shape==(None,28,28,1)

		return model
	
		
	
	
		