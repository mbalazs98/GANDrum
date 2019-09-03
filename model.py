import pypianoroll as pp

import os

import numpy as np

from scipy import sparse

import tensorflow as tf

from tensorflow.keras import layers

import time

import sys



class GANDrum(object):

	def __init__(self,buffer_size=60000,batch_size=72,epochs=20,noise_dim=100,num_examples_to_generate=1):

		self.train_files=[]

		self.buffer_size=buffer_size

		self.batch_size=batch_size

		self.load_data()


		self.generator_optimizer=tf.keras.optimizers.Adam(1e-4)

		self.discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)

		

		self.generator=self.make_generator_model()

		self.discriminator=self.make_discriminator_model()

		

		self.checkpoint_prefix=os.path.join('./training_checkpoints',"ckpt")

		self.checkpoint=tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,discriminator_optimizer=self.discriminator_optimizer,generator=self.generator,discriminator=self.discriminator)

		

		self.epochs=epochs

		self.noise_dim=noise_dim

		self.seed=tf.random.normal([num_examples_to_generate, noise_dim])

		

		

	def load_data(self):

		i=0

		entries=os.listdir(r'D:\GANDrum\midi_d_processed')

		for entry in entries:

			try:

				data=(pp.parse(r'D:/GANDrum/midi_d_processed/'+entry))

				mtx=tf.reshape(tensor=tf.convert_to_tensor(data.tracks[0].pianoroll),shape=(8,12,128))
				
				mtx=tf.cast(mtx, tf.float32)

				self.train_files.append(mtx)

			except:

				i=i+1

		print(i,'files were not in the correct format')

		self.train_dataset=tf.data.Dataset.from_tensor_slices(self.train_files).shuffle(self.buffer_size).batch(self.batch_size)



	def make_generator_model(self):

		try:

			model=tf.keras.Sequential()

				

				

			model.add(layers.Dense(units=1536,input_shape=(100,)))

			model.add(layers.BatchNormalization())

			model.add(layers.LeakyReLU())

						

			model.add(layers.Dense(units=768))

			model.add(layers.BatchNormalization())

			model.add(layers.LeakyReLU())





			model.add(layers.Reshape((2,3,128)))





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

		except:

			print('make_generator_model')
			
			
			
	def make_discriminator_model(self):

		try:

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



			return model

		except:

			print('make_discriminator_model')

		

		

	def discriminator_loss(self,real_output,fake_output):

		cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
		
		real_loss=cross_entropy(tf.ones_like(real_output),real_output)

		fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)

		total_loss=real_loss+fake_loss

		return total_loss

	

	

	def generator_loss(self,fake_output):

		cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
		
		return cross_entropy(tf.ones_like(fake_output),fake_output)

	

	

	def train_step(self,images):

		noise = tf.random.normal([self.batch_size,self.noise_dim])



		with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

			generated_midi=self.generator(noise,training=True)



			real_output=self.discriminator(images,training=True)

			fake_output=self.discriminator(generated_midi,training=True)



			gen_loss=self.generator_loss(fake_output)

			disc_loss=self.discriminator_loss(real_output,fake_output)



		gradients_of_generator=gen_tape.gradient(gen_loss,self.generator.trainable_variables)

		gradients_of_discriminator=disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)



		self.generator_optimizer.apply_gradients(zip(gradients_of_generator,self.generator.trainable_variables))

		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,self.discriminator.trainable_variables))

	

	

	def train(self,dataset,epochs):

		for self.epoch in range(epochs):

			start=time.time()



		for midi_batch in self.train_dataset:

			self.train_step(midi_batch)



		self.generate_and_save_midi(self.generator,self.epoch+1,self.seed)



		if(self.epoch+1)%15==0:

			checkpoint.save(file_prefix=self.checkpoint_prefix)



		print ('Time for epoch {} is {} sec'.format(self.epoch+1,time.time()-start))





	def generate_and_save_midi(self,model,epoch,test_input):

			predictions=model(test_input,training=False)
			
			shaped_predictions=np.reshape(a=predictions.numpy(),newshape=(96,128))
			
			track=pp.Track(pianoroll=shaped_predictions,is_drum=True)
			
			multitrack=pp.Multitrack(tracks=[track])
			
			multitrack.write('./generated_midi')





	def run(self):

		self.train(self.train_dataset,self.epochs)