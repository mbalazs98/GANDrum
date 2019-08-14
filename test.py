import tensorflow as tf
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(train_images)