import pypianoroll as pp
import os
import numpy as np
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers
import time

class GANDrum(object):
	def __init__(self,buffer_size=60000,batch_size=72,epochs=20,noise_dim=100,num_examples_to_generate=16):
		self.train_files=[]
		self.buffer_size=buffer_size
		self.batch_size=batch_size
		cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
		generator_optimizer=tf.keras.optimizers.Adam(1e-4)
		discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
		checkpoint_dir='./training_checkpoints'
		checkpoint_prefix=os.path.join(checkpoint_dir,"ckpt")
		checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
		self.epochs=epochs
		self.noise_dim=noise_dim
		self.num_examples_to_generate=num_examples_to_generate
		self.seed=tf.random.normal([num_examples_to_generate, noise_dim])
		
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
		
	def make_generator_model(self):
		model=tf.keras.Sequential()
			
			
		model.add(layers.Dense(units=512,input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
			
		model.add(layers.Dense(units=256))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Reshape((2,1,128)))


		model.add(layers.Conv2DTranspose(filters=1,kernel_size=2,strides=2))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
			
		model.add(layers.Conv2DTranspose(filters=1,kernel_size=2,strides=2))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
			
		model.add(layers.Conv2DTranspose(filters=1,kernel_size=2,strides=2))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(filters=128,kernel_size=1,strides=1,activation='tanh'))
		model.add(layers.BatchNormalization())
		model.add(layers.Softmax())

		return model
	
	def make_discriminator_model(self):
		model=tf.keras.Sequential()
		
		
		model.add(layers.Conv2D(filters=128,kernel_size=1,strides=1,input_shape=[16,8,128]))
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
		model.add(layers.LeakyReLU())
		
		model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
		model.add(layers.LeakyReLU())
		
		model.add(layers.Conv2D(filters=1,kernel_size=2,strides=2))
		model.add(layers.LeakyReLU())

		model.add(layers.Flatten())
		model.add(layers.Dense(1))

		return model
		
	def discriminator_loss(real_output,fake_output):
		real_loss=cross_entropy(tf.ones_like(real_output),real_output)
		fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
		total_loss=real_loss+fake_loss
		return total_loss
		
	def generator_loss(fake_output):
		return cross_entropy(tf.ones_like(fake_output),fake_output)
		
	def train_step(images):
    noise = tf.random.normal([self.batch_size,self.noise_dim])

	with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
      generated_images=generator(noise,training=True)

      real_output=discriminator(images,training=True)
      fake_output=discriminator(generated_images,training=True)

      gen_loss=generator_loss(fake_output)
      disc_loss=discriminator_loss(real_output,fake_output)

	gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)
	gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
	
	def train(dataset, epochs):
		for self.epoch in range(epochs):
			start=time.time()

		for image_batch in dataset:
			train_step(image_batch)

		display.clear_output(wait=True)
		generate_and_save_images(generator,epoch+1,seed)

		if (epoch + 1) % 15 == 0:
		  checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

		