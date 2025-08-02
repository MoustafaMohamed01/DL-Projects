# Face Mask Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) to classify whether a person is wearing a face mask or not. Built using TensorFlow and Keras, the model is trained on a labeled dataset of facial images with real-time inference capability, training visualizations, and GPU support.

---

## Key Features

- Deep learning model (CNN) built with TensorFlow and Keras
- Binary classification: `With Mask` vs `Without Mask`
- Data augmentation for robust training
- Dark-themed visualizations for loss and accuracy
- GPU acceleration (automatically detected)
- Model saving and custom image inference
- Public dataset from Kaggle

---

## Dataset

This project uses the publicly available **Face Mask Dataset** on Kaggle:
**[Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)**

After downloading, structure your data as follows:

```

data/
├── with_mask/
└── without_mask/

```

---

## Project Structure

```

face-mask-detection/
├── data/                         # Dataset images (from Kaggle)
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── face_mask_model.h5        # Saved trained model
│
├── plots/
│   ├── loss_plot.png             # Loss curve
│   └── accuracy_plot.png         # Accuracy curve
│
├── face_mask_detector.py         # Main Python script
├── face_mask_detector.ipynb      # Jupyter Notebook version
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

````

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MoustafaMohamed01/DL-Projects.git
cd DL-Projects/face-mask-detection
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Download and unzip the dataset from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset), and place the folders inside the `data/` directory as shown above.

### 4. Train the Model

```bash
python face_mask_detector.py
```

---

## Inference

After training, the script will prompt you to provide an image path:

```bash
Path of image: sample.png
```

Output:

* Raw prediction (probability)
* Final classification: `With Mask 😷` or `Without Mask 😐`

---

## Model Architecture

```text
Input (128x128x3)
├── Conv2D(32) + ReLU + MaxPooling
├── Conv2D(64) + ReLU + MaxPooling
├── Conv2D(128) + ReLU + MaxPooling
├── Flatten
├── Dense(128) + Dropout(0.5)
└── Dense(1) + Sigmoid
```

---

## Visualization

The training history (loss and accuracy) is plotted and saved to the `plots/` folder.

* `loss_plot.png`
* `accuracy_plot.png`

---

## Save & Load Model

Saved model path:

```
models/face_mask_model.h5
```

To reload the model later:

```python
from tensorflow.keras.models import load_model
model = load_model('models/face_mask_model.h5')
```

---

## Tech Stack

* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy, Matplotlib, Pillow
* Scikit-learn

---

## Author

**Moustafa Mohamed**

AI Developer | Machine Learning & Deep Learning & LLMs Specialist | Kaggle Notebooks Expert (Top 5%)

[GitHub](https://github.com/MoustafaMohamed01) • [Kaggle](https://www.kaggle.com/moustafamohamed01) • [LinkedIn](https://www.linkedin.com/in/moustafamohamed01/)

---
