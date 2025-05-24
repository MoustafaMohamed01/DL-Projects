# Fruit and Vegetable Image Classification Project

# Importing Necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Loading Image Dataset for Training
data_train = tf.keras.utils.image_dataset_from_directory(
    "train",
    shuffle=True,
    image_size=(180, 180),
    batch_size=32,
    validation_split=False
)

data_cat = data_train.class_names

# Loading Validation Dataset
data_val = tf.keras.utils.image_dataset_from_directory(
    "validation",
    image_size=(180, 180),
    batch_size=32,
    shuffle=False,
    validation_split=False
)

# Loading Test Dataset
data_test = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=(180, 180),
    batch_size=32,
    validation_split=False
)

# Convolutional Neural Network (CNN) Model
model = Sequential()
model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(16, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128))
model.add(layers.Dense(len(data_cat)))
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Training the CNN Model
epochs=25
training_results = model.fit(data_train, validation_data=data_val, epochs=epochs, batch_size=32)

# Visualizing Model Performance
epochs_range = range(epochs)
plt.style.use("dark_background")
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, training_results.history["accuracy"], label="Training Accuracy")
plt.plot(epochs_range, training_results.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, training_results.history["loss"], label="Training Loss")
plt.plot(epochs_range, training_results.history["val_loss"], label="Validation Loss")
plt.title("Loss")

plt.tight_layout()
plt.legend()
plt.show()
plt.savefig("training_accuracy&loss.png")

# Image Prediction and Class Probability
image = "apple.jpg"
image = tf.keras.utils.load_img(image, target_size=(180, 180))
image = tf.keras.utils.array_to_img(image)
image = tf.expand_dims(image, 0)
predict = model.predict(image)
score = tf.nn.softmax(predict)
print("Image: {} | Accuracy: {:0.2f}%".format(data_cat[np.argmax(score)], np.max(score)*100))

model.save("Fruit& Vegetables Image Classifier.keras")
