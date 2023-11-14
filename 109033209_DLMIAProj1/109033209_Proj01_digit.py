# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:28:52 2021

@author: User
"""

import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train = x_train.astype("float32") / 255.0

img_file = []
for roots, dirs, files in os.walk("processed_number/"):
    for file in files:
        img_file.append("processed_number/"+file)

x_test = np.empty(shape=[0, 28, 28])

for i in range(10):
    image = cv2.imread(img_file[i], 0)
    img = image/255.0
    img = np.expand_dims(img, 0)
    x_test = np.vstack((x_test, img))

test_label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    # %% FCNN


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

probability_model = tf.keras.Sequential(
    [fcnn_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)

num_rows = 5
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_label, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_label)
plt.tight_layout()
# plt.savefig("digit_fcnn.png")
plt.show()
# %% CNN
"""
## Build the model
"""
num_classes = 10
x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_test = np.expand_dims(x_test, -1)

input_shape = (28, 28, 1)

cnn_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

cnn_model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 10

cnn_model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])


cnn_model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)

# test

pre_array = cnn_model.predict(x_test)

num_rows = 5
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pre_array[i], test_label, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pre_array[i], test_label)
plt.tight_layout()
# plt.savefig("digit_cnn.png")
plt.show()
