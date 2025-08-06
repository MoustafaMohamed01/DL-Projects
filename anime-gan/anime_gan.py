import os
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras import layers

DATA_DIR = "/kaggle/input/animefacedataset/images"
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 100
SAVE_FREQ = 50

OUTPUT_DIR = "./anime_gan_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_images(data_dir, img_size):
    paths = glob(os.path.join(data_dir, "*.jpg"))
    dataset = []
    for path in tqdm(paths, desc="Loading images"):
        try:
            img = Image.open(path).resize((img_size, img_size)).convert("RGB")
            img = np.asarray(img) / 127.5 - 1.0
            dataset.append(img)
        except:
            continue
    return np.array(dataset)


images = load_images(DATA_DIR, IMG_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(1000).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh'),
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=[IMG_SIZE, IMG_SIZE, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                    cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

seed = tf.random.normal([16, LATENT_DIM])

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig(os.path.join(OUTPUT_DIR, f"anime_epoch_{epoch:03d}.png"))
    plt.close()


def train(dataset, epochs):
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        start = time.time()
        gen_losses = []
        disc_losses = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        print(f"Epoch {epoch:03d} | Gen Loss: {np.mean(gen_losses):.4f} | Disc Loss: {np.mean(disc_losses):.4f} | Time: {time.time() - start:.2f}s")

        if epoch % SAVE_FREQ == 0 or epoch == 1:
            generate_and_save_images(generator, epoch, seed)

train(train_dataset, EPOCHS)

from IPython.display import display, Image as IPImage

def show_generated_images():
    first_image_path = os.path.join(OUTPUT_DIR, f"anime_epoch_001.png")
    last_image_path = os.path.join(OUTPUT_DIR, f"anime_epoch_{EPOCHS:03d}.png")

    print(f"First Epoch Output:")
    display(IPImage(first_image_path))

    print(f"\nLast Epoch Output:")
    display(IPImage(last_image_path))

show_generated_images()
