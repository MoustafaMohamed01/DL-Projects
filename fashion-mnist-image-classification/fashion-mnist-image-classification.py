import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    return train_images, train_labels, test_images, test_labels

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

def plot_training_history(history):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background') 

    plt.figure(figsize=(12, 5))
    color_train = '#00FFAB' 
    color_val = '#FF6EC7'    

    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy', color=color_train, linewidth=2)
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy', color=color_val, linestyle='--', linewidth=2)
    plt.title('📈 Accuracy Over Epochs', fontsize=14, color='#00FFFF')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color=color_train, linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color=color_val, linestyle='--', linewidth=2)
    plt.title('📉 Loss Over Epochs', fontsize=14, color='#FF00FF')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Training Progress Overview 🚀', fontsize=16, color='cyan')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
    plt.title(f"Sample Image - Label: {train_labels[0]}")
    plt.axis('off')
    plt.show()

    model = build_model()
    model.summary()

    early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

    history = model.fit(
        train_images, train_labels,
        epochs=20,
        batch_size=64,
        validation_data=(test_images, test_labels),
        callbacks=[early_stop, checkpoint]
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    plot_training_history(history)

if __name__ == '__main__':
    main()
