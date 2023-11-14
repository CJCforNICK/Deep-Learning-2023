# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:49:59 2021

@author: User
"""

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf


class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

img_file = []
for roots, dirs, files in os.walk("p_digits/"):
    for file in files:
        img_file.append("p_digits/" + file)

test_img = np.empty(shape=[0, 28, 28])

for i in range(10):
    image = cv2.imread(img_file[i], 0)
    img = image / 255.0
    img = np.expand_dims(img, 0)
    test_img = np.vstack((test_img, img))

test_label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# %%


fcnn_model = keras.models.load_model("Model/digit_fcnn.h5")

x_test = test_img

probability_model = tf.keras.Sequential([fcnn_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)

# num_rows = 5
# num_cols = 2
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], test_label, x_test)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], test_label)
# plt.tight_layout()
# plt.savefig("results/digit_fcnn1.png")
# # plt.show()
# %%

cnn_model = keras.models.load_model("Model/digit_cnn.h5")
x_test = np.expand_dims(test_img, -1)
pre_array = cnn_model.predict(x_test)


num_rows = 5
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, pre_array[i], test_label, test_img)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, pre_array[i], test_label)
plt.tight_layout()
plt.savefig("results/digit_cnn1.png")
plt.show()
