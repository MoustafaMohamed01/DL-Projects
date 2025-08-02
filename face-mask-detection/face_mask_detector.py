import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

with_mask = os.listdir('data/with_mask')
without_mask = os.listdir('data/without_mask')

print(f"Images with mask: {len(with_mask)}")
print(f"Images without mask: {len(without_mask)}")

with_mask_labels = [1]*3725
without_mask_labels = [0]*3828

labels = with_mask_labels + without_mask_labels

img = mpimg.imread('data/with_mask/with_mask_1.jpg')
imgplot = plt.imshow(img)
plt.show()

img = mpimg.imread('data/without_mask/without_mask_2925.jpg')
imgplot = plt.imshow(img)
plt.show()

with_mask_path = 'data/with_mask'
data = []
labels = []

for img_name in os.listdir(with_mask_path):
    img_path = os.path.join(with_mask_path, img_name)
    try:
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        data.append(np.array(img))
        labels.append(1)
    except:
        pass

without_mask_path = 'data/without_mask'

for img_name in os.listdir(without_mask_path):
    img_path = os.path.join(without_mask_path, img_name)
    try:
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        data.append(np.array(img))
        labels.append(0)
    except:
        pass

X = np.array(data) / 255.0
y = np.array(labels)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train_scaled.shape)
print(y_train.shape)

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=5)

loss, accuracy = model.evaluate(X_test_scaled, y_test)

h = history

plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(h.history['accuracy'], label='train acc')
plt.plot(h.history['val_accuracy'], label='validation acc')
plt.legend()
plt.show()

input_image_path = input('Path of image: ')
input_image = cv2.imread(input_image_path)

cv2.imshow("Input Image", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255.0
input_image_reshaped = np.reshape(input_image_scaled, (1, 128, 128, 3))

input_prediction = model.predict(input_image_reshaped)
print("Raw prediction:", input_prediction)

if input_prediction[0][0] > 0.5:
    print("With Mask")
else:
    print("Without Mask")

model.save("models/face_mask_model.h5")
