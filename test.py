import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def make_generator_model():
	model=tf.keras.Sequential()
		
		
	model.add(layers.Dense(units=1024,input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
		
	model.add(layers.Dense(units=512))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())


	model.add(layers.Reshape((1,2)))


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
	model.add(layers.softmax())

	return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

