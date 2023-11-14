import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255.0

# Reshape images to have size larger than 200x200
train_images = tf.image.resize(train_images, [224, 224])
test_images = tf.image.resize(test_images, [224, 224])

# Define CNN model
model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten output and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(train_images, train_labels, batch_size=16, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print test accuracy
print('Test accuracy:', test_acc)
