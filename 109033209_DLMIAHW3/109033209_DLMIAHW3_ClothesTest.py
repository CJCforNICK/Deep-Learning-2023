# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:27:42 2021

@author: User
"""


import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


img_file = []
for roots, dirs, files in os.walk("processed_cloth/"):
    for file in files:
        img_file.append("processed_cloth/"+file)
# print(img_file)
# print(len(img_file))

test_img_added = np.empty(shape=[0, 28, 28])

for i in range(len(img_file)):
    image = cv2.imread(img_file[i], 0)
    img = image/255.
    img = np.expand_dims(img, 0)
    test_img_added = np.vstack((test_img_added, img))

test_label = np.array([4, 7, 5, 4, 6, 0, 0, 0, 0, 1,
                      1, 8, 4, 7, 0, 0, 8, 8, 6, 0])


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

# %%


fcnn_model = tf.keras.models.load_model("Model/cloth_fcnn.h5")

x_test = test_img_added

probability_model = tf.keras.Sequential(
    [fcnn_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)

num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_label, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_label)
plt.tight_layout()
plt.savefig("results/cloth_fcnn1.png")
plt.show()
# %%

cnn_model = tf.keras.models.load_model("Model/cloth_cnn.h5")
x_test = np.expand_dims(test_img_added, -1)
pre_array = cnn_model.predict(x_test)


num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pre_array[i], test_label, test_img_added)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pre_array[i], test_label)
plt.tight_layout()
plt.savefig("results/cloth_cnn1.png")
plt.show()
