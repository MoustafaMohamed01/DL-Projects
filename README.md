# Deep Learning Projects Portfolio

Welcome to my portfolio of deep learning projects, A curated collection of deep learning projects implemented using **TensorFlow**, **Keras**, and **PyTorch**. This repository demonstrates practical applications of neural networks in domains such as image classification, generative modeling, and medical diagnostics, emphasizing clean code, reproducibility, and performance evaluation.

---

## Repository Structure

Each subfolder within this repository contains an independent deep learning project, complete with source code, dataset details, training instructions, and results visualization.

| Project                                                                                             | Framework        | Domain                | Key Topics                              |
| --------------------------------------------------------------------------------------------------- | ---------------- | --------------------- | --------------------------------------- |
| [Fruit & Vegetable Image Classification](#1-fruit--vegetable-image-classification)                  | TensorFlow/Keras | Image Classification  | CNNs, Image Processing                  |
| [Breast Cancer Prediction using Neural Networks](#2-breast-cancer-prediction-using-neural-networks) | PyTorch          | Binary Classification | Feedforward Neural Networks, Medical AI |
| [Fashion MNIST Image Classification](#3-fashion-mnist-image-classification)                         | TensorFlow/Keras | Image Classification  | CNNs, Regularization                    |
| [Face Mask Detection](#4-face-mask-detection)                                                       | TensorFlow/Keras | Binary Classification | Real-Time Detection, CNN, OpenCV        |
| [MNIST GAN – Digit Generation](#5-mnist-gan--digit-generation)                                      | TensorFlow       | Generative Modeling   | GANs, Image Synthesis                   |

---

## 1. Fruit & Vegetable Image Classification

A convolutional neural network (CNN) designed to classify images of fruits and vegetables.

**Technologies:** TensorFlow, Keras, Python
**Dataset:** [Kaggle - Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

### Highlights

* Data preprocessing and augmentation
* CNN architecture with Dropout and MaxPooling
* Real-time prediction support with confidence scores
* Training and validation performance visualization

📁 Folder: [Fruits-and-Vegetables-Image-Recognition-Dataset](./Fruits-and-Vegetables-Image-Recognition-Dataset)

---

## 2. Breast Cancer Prediction using Neural Networks

A binary classification model developed using PyTorch to predict tumor malignancy from the Breast Cancer Wisconsin dataset.

**Technologies:** PyTorch, Scikit-learn, Matplotlib
**Dataset:** [sklearn.datasets.load\_breast\_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

### Highlights

* FFNN architecture
* Binary cross-entropy loss with Adam optimizer
* Training curves and evaluation metrics
* Available in Jupyter and standalone script formats

📁 Folder: [breast-cancer-prediction](./breast-cancer-prediction)

---

## 3. Fashion MNIST Image Classification

A CNN-based image classifier trained on Fashion MNIST dataset, classifying clothing items into 10 categories.

**Technologies:** TensorFlow, Keras
**Dataset:** [`tensorflow.keras.datasets.fashion_mnist`](https://www.tensorflow.org/datasets/catalog/fashion_mnist)

### Highlights

* Batch Normalization and Dropout layers
* EarlyStopping and ModelCheckpoint callbacks
* Accuracy up to 91–93% with tuning
* Stylish metric visualizations

📁 Folder: [Fashion-MNIST-Image-Classification](./Fashion-MNIST-Image-Classification)

---

## 4. Face Mask Detection

A real-time CNN classifier that detects whether a person is wearing a face mask or not.

**Technologies:** TensorFlow, Keras, OpenCV, Python
**Dataset:** [Kaggle - Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

### Highlights

* Binary classification: With Mask 😷 vs Without Mask 😐
* Data augmentation and preprocessing
* Real-time prediction from user-provided images
* Model saved and reloadable (`.h5`)
* GPU support and training visualizations

📁 Folder: [face-mask-detection](./face-mask-detection)

---

## 5. MNIST GAN – Digit Generation

A Generative Adversarial Network (GAN) that synthesizes realistic handwritten digits from the MNIST dataset.

**Technologies:** TensorFlow, Python
**Dataset:** [`tensorflow.keras.datasets.mnist`](https://www.tensorflow.org/datasets/catalog/mnist)

### Highlights

* Fully functional GAN (Generator + Discriminator)
* Saves generated digit images every epoch
* GAN-stabilization tricks: label smoothing, custom beta values
* Available in both `.py` and `.ipynb` formats

📁 Folder: [mnist-gan](./mnist-gan)

---

## Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/MoustafaMohamed01/DL-Projects.git
cd DL-Projects
```

2. **Install Dependencies**
   Each project includes a `requirements.txt`. To install dependencies:

```bash
pip install -r requirements.txt
```

3. **Run Projects**
   Navigate to the relevant folder and follow its README to train or run inference.

---

## Contributing

Contributions are welcome! If you’d like to improve a project or add a new one:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

Ideas, feedback, and improvements are always appreciated.

---

## Connect With Me

* **LinkedIn:** [Moustafa Mohamed](https://www.linkedin.com/in/moustafamohamed01/)
* **GitHub:** [MoustafaMohamed01](https://github.com/MoustafaMohamed01)
* **Kaggle:** [moustafamohamed01](https://www.kaggle.com/moustafamohamed01)
* **Portfolio:** [Moustafa Mohamed](https://moustafamohamed.netlify.app/)

---
