# Deep Learning Projects Portfolio

A curated collection of deep learning projects implemented using **TensorFlow**, **Keras**, and **PyTorch**. This repository demonstrates practical applications of neural networks in domains such as image classification and medical diagnostics, emphasizing clean code, reproducibility, and performance evaluation.

---

## Repository Structure

Each subfolder within this repository contains an independent deep learning project, complete with source code, dataset details, and instructions for execution.

| Project                                   | Framework          | Domain              | Key Topics                        |
|-------------------------------------------|--------------------|---------------------|----------------------------------|
| [Fruit & Vegetable Image Classification](#1-fruit--vegetable-image-classification) | TensorFlow/Keras   | Image Classification | CNNs, Image Processing            |
| [Breast Cancer Prediction using Neural Networks](#2-breast-cancer-prediction-using-neural-networks) | PyTorch            | Binary Classification | Feedforward Neural Networks, Medical AI |
| [Fashion MNIST Image Classification](#3-fashion-mnist-image-classification)         | TensorFlow/Keras   | Image Classification | CNNs, Batch Normalization, Dropout |

---

## 1. Fruit & Vegetable Image Classification

A convolutional neural network (CNN) designed to classify images of fruits and vegetables.

**Technologies:** TensorFlow, Keras, Python  
**Dataset:** [Kaggle - Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

### Highlights
- Data preprocessing and augmentation techniques
- CNN architecture with dropout and max pooling layers
- Training and validation performance visualization
- Real-time image prediction with confidence scores

### Folder: [Fruits-and-Vegetables-Image-Recognition-Dataset](./Fruits-and-Vegetables-Image-Recognition-Dataset)

---

## 2. Breast Cancer Prediction using Neural Networks

A binary classification model developed using PyTorch to predict tumor malignancy from the Breast Cancer Wisconsin dataset.

**Technologies:** PyTorch, Scikit-learn, Matplotlib  
**Dataset:** [sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

### Highlights
- Feedforward neural network (FFNN) architecture
- Training using binary cross-entropy loss and Adam optimizer
- Visualizations of accuracy and loss curves
- Implementations in both Jupyter notebooks and standalone scripts

### Folder: [breast-cancer-prediction](./breast-cancer-prediction)

---

## 3. Fashion MNIST Image Classification

A CNN-based image classifier trained on the Fashion MNIST dataset, comprising grayscale images of clothing items from 10 categories.

**Technologies:** TensorFlow, Keras, Python  
**Dataset:** [`tensorflow.keras.datasets.fashion_mnist`](https://www.tensorflow.org/datasets/catalog/fashion_mnist)

### Highlights
- CNN architecture enhanced with Batch Normalization and Dropout for improved generalization
- EarlyStopping and ModelCheckpoint callbacks to optimize training
- Futuristic dark-themed visualizations for accuracy and loss metrics
- Achieves approximately 91–93% accuracy after tuning

### Folder: [Fashion-MNIST-Image-Classification](./Fashion-MNIST-Image-Classification)

---

## Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/MoustafaMohamed01/DL-Projects.git
cd DL-Projects
````

2. **Install Dependencies**
   Each project contains its own `requirements.txt` or setup instructions in its folder. To install dependencies, run:

```bash
pip install -r requirements.txt
```

3. **Execute Projects**
   Follow the README and instructions provided within each project folder to run notebooks or scripts.

---

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or bugfix, and submit a pull request. Suggestions and improvements are highly appreciated.

---

## Connect With Me

* **GitHub:** [MoustafaMohamed01](https://github.com/MoustafaMohamed01)
* **Kaggle:** [moustafamohamed01](https://www.kaggle.com/moustafamohamed01)
* **LinkedIn:** [Moustafa Mohamed](https://www.linkedin.com/in/moustafamohamed01/)

---
