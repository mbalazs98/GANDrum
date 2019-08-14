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
		self.load_data()
		
		self.cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
		
		self.generator_optimizer=tf.keras.optimizers.Adam(1e-4)
		self.discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
		
		self.generator=make_generator_model()
		self.discriminator=make_discriminator_model()
		
		self.checkpoint_prefix=os.path.join('./training_checkpoints',"ckpt")
		self.checkpoint=tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,discriminator_optimizer=self.discriminator_optimizer,generator=self.generator,discriminator=self.discriminator)
		
		self.epochs=epochs
		self.seed=tf.random.normal([num_examples_to_generate, noise_dim])
		print('init')
		
		
	def load_data(self):
		i=0
		entries=os.listdir(r'D:\GANDrum\midi_d_processed')
		for entry in entries:
			try:
				data=pp.parse(r'D:/GANDrum/midi_d_processed/'+entry)
				mtx=np.reshape(a=data.tracks[0].pianoroll,newshape=(8,12,128))
				self.train_files.append(mtx)
			except:
				i=i+1
		#self.train_labels=np.load(r'D:\midi_d_label\labels.npy')
		self.train_labels=np.ones(i-1)
		
		self.train_dataset=self.train_files#tf.compat.v1.data.get_output_classes([self.train_files,self.train_labels])
		for i in range(0,len(self.train_dataset)):
			self.train_dataset[i].shuffle(self.buffer_size)
		for i in range(0,len(self.train_dataset)):
			self.train_dataset[i].batch(self.batch_size)


	def make_generator_model(self):
		model=tf.keras.Sequential()
			
			
		model.add(layers.Dense(units=6144,input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
					
		model.add(layers.Dense(units=3072))
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


	def make_discriminator_model(self):
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
		
		
	def discriminator_loss(real_output,fake_output):
		real_loss=cross_entropy(tf.ones_like(real_output),real_output)
		fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
		total_loss=real_loss+fake_loss
		return total_loss
		
		
	def generator_loss(fake_output):
		return cross_entropy(tf.ones_like(fake_output),fake_output)
		
		
	def train_step(images):
		print('train_step')
		noise = tf.random.normal([self.batch_size,self.noise_dim])

		with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
		  generated_midi=self.generator(noise,training=True)

		  real_output=self.discriminator(images,training=True)
		  fake_output=self.discriminator(generated_midi,training=True)

		  gen_loss=generator_loss(fake_output)
		  disc_loss=discriminator_loss(real_output,fake_output)

		gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)
		gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))	
	
	
	def train(dataset,epochs):
		print('train')
		for self.epoch in range(epochs):
			start=time.time()

		for midi_batch in self.dataset:
			train_step(midi_batch)

		display.clear_output(wait=True)
		generate_and_save_midi(self.generator,self.epoch+1,self.seed)

		if(self.epoch+1)%15==0:
		  checkpoint.save(file_prefix=self.checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(self.epoch+1,time.time()-start))


	def generate_and_save_midi(model,epoch,test_input):
		predictions=model(test_input,training=False)
		pp.write('./generated_midi',predictions)
	
	
	def run(self):
		try:
			train(self.train_dataset,self.epochs)
		except:
			sys.exit("Error message")