import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for img_name in os.listdir(folder):
        try:
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path).convert('RGB').resize((128, 128))
            images.append(np.array(img))
            labels.append(label)
        except:
            continue
    return images, labels

with_mask_images, with_mask_labels = load_images_from_folder('data/with_mask', 1)
without_mask_images, without_mask_labels = load_images_from_folder('data/without_mask', 0)

data = with_mask_images + without_mask_images
labels = with_mask_labels + without_mask_labels

print(f"Loaded {len(data)} images.")

X = np.array(data) / 255.0
y = np.array(labels)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.1, 
    random_state=42
)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=15,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

plt.style.use('dark_background')
train_color = '#00ffe7'
val_color = '#ff4c98'

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color=train_color, linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', color=val_color, linewidth=2)
plt.title('Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=300)
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc', color=train_color, linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', color=val_color, linewidth=2)
plt.title('Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)
plt.show()


input_image_path = input("Path of image: ").strip()
input_image = cv2.imread(input_image_path)

if input_image is None:
    print("Error: Image not found.")
else:
    cv2.imshow("Input Image", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    input_resized = cv2.resize(input_image, (128, 128))
    input_scaled = input_resized / 255.0
    input_reshaped = np.reshape(input_scaled, (1, 128, 128, 3))

    prediction = model.predict(input_reshaped)
    label = int(prediction[0][0] > 0.5)

    print(f"Raw Prediction: {prediction[0][0]:.4f}")
    if label == 1:
        print("Prediction: With Mask")
    else:
        print("Prediction: Without Mask")


os.makedirs("models", exist_ok=True)
model.save("models/face_mask_model.h5")
