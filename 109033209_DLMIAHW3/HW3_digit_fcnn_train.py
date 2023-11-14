# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:12:13 2021

@author: User
"""

import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

fcnn_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])


fcnn_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(
                       from_logits=True),
                   metrics=['accuracy'])

fcnn_model.summary()

fcnn_model.fit(x_train, y_train, epochs=10)

fcnn_model.save("Model/digit_fcnn.h5")

probability_model = tf.keras.Sequential(
    [fcnn_model, tf.keras.layers.Softmax()])

test_loss, test_acc = fcnn_model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
