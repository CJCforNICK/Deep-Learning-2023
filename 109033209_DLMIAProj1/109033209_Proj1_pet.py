import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers, models

dataset, info = tfds.load(
    'oxford_iiit_pet', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Preprocess data


def preprocess_data(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize images to 224x224
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    label = tf.one_hot(label, depth=37)  # One-hot encode labels
    return image, label


train_dataset = train_dataset.map(preprocess_data).shuffle(1000).batch(32)
test_dataset = test_dataset.map(preprocess_data).batch(32)

# Define CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(37, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

print('Test accuracy:', test_acc)
