# Fashion MNIST - CNN Image Classifier

A deep learning project using Convolutional Neural Networks (CNNs) to classify images from the Fashion MNIST dataset. This dataset includes grayscale images of clothing items (e.g., shirts, sneakers, bags), with 10 distinct classes.

---

## Model Overview

- **Architecture**: 3 Convolutional layers with Batch Normalization, MaxPooling, and Dropout
- **Dense Layers**: Flatten + Dense + Output Softmax
- **Optimization**: Adam Optimizer with Sparse Categorical Crossentropy
- **Callbacks**: EarlyStopping and ModelCheckpoint for better generalization

---

## Performance

- Achieves ~91% accuracy on the validation set after tuning.
- Includes real-time plots of accuracy and loss in a futuristic dark theme.

---

## Dataset

- Source: [`tensorflow.keras.datasets.fashion_mnist`](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- Train: 60,000 images
- Test: 10,000 images
- Image Size: 28x28 grayscale

---

## How to Run

 - **Step 1: Install dependencies**
```bash
    pip install -r requirements.txt
````
- **Step 2: Run the main script**
```bash
    python fashion_mnist_optimized.py
````

---

## Sample Output

* Accuracy & loss visualizations using `matplotlib`
* Neon-colored futuristic charts in dark mode

---

## Dependencies

* `TensorFlow`
* `NumPy`
* `Matplotlib`
* `scikit-learn`

See [`requirements.txt`](./requirements.txt) for full details.

---

## Project Structure

```
fashion-mnist-image-classification
├── fashion_mnist_optimized.py      # Main training script
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## Author

**Moustafa Mohamed**
Aspiring AI Developer
[LinkedIn](https://www.linkedin.com/in/moustafamohamed01) | [GitHub](https://github.com/MoustafaMohamed01) | [Portfolio](https://moustafamohamed.netlify.app/)

---
