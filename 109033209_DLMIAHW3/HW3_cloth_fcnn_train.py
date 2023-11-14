# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:25:59 2021

@author: User
"""


from tensorflow import keras
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

fcnn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])


fcnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

fcnn_model.summary()

fcnn_model.fit(x_train, y_train, epochs=10)

fcnn_model.save("Model/cloth_fcnn.h5")

probability_model = tf.keras.Sequential([fcnn_model,tf.keras.layers.Softmax()])

test_loss, test_acc = fcnn_model.evaluate(x_test,  y_test, verbose=2)

print("\nTest loss: ",test_loss)
print('Test accuracy:', test_acc)