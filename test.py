import matplotlib.pyplot as plt
from model import GANDrum
import tensorflow as tf

m=GANDrum()

generator = m.make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

for item in generated_image:
	print(item)


