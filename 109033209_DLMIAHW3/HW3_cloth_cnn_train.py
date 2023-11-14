# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:34:01 2021

@author: User
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

num_classes = 10
x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_test = np.expand_dims(x_test, -1)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (28, 28, 1)

cnn_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

cnn_model.summary()

cnn_model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

cnn_model.fit(x_train, y_train, batch_size=128,
              epochs=10, validation_split=0.1)

cnn_model.save("Model/cloth_cnn.h5")

score = cnn_model.evaluate(x_test, y_test, verbose=0)

print("\nTest loss:", score[0])
print("Test accuracy:", score[1])
