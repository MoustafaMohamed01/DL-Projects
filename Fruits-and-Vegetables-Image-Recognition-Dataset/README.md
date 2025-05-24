# 🍎 Fruit and Vegetable Image Classification

## Project Overview
This project is a deep learning-based image classification system that identifies different fruits and vegetables using a Convolutional Neural Network (CNN). The model is trained on an image dataset of various fruits and vegetables and achieves classification through TensorFlow and Keras.

---

## Results  

### **Training Accuracy & Loss**  
![Training Accuracy & Loss](training_accuracy&loss.png)

---

## Features
- Uses a CNN model to classify images of fruits and vegetables.
- Trained on a dataset of labeled fruit and vegetable images.
- Implements data preprocessing and augmentation techniques.
- Provides visualization of training performance.
- Allows prediction on new images with confidence scores.

## Dataset
The dataset used in this project is available on Kaggle: [Fruit and Vegetable Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition).
The dataset consists of three parts:
- **Training Dataset** (`train/`): Used for training the CNN model.
- **Validation Dataset** (`validation/`): Used to validate model performance.
- **Test Dataset** (`test/`): Used to evaluate the final model performance.

## Model Architecture
The model is built using a CNN with the following layers:
- **Rescaling Layer**: Normalizes pixel values.
- **Convolutional Layers**: Extracts features from images using different filters.
- **MaxPooling Layers**: Reduces spatial dimensions while preserving important features.
- **Flatten Layer**: Converts feature maps into a 1D vector.
- **Dropout Layer**: Prevents overfitting by randomly dropping connections.
- **Dense Layers**: Final classification layers with Softmax activation.

## Installation & Setup
### Clone the Repository
```bash
git clone https://github.com/MoustafaMohamed01/Fruits-and-Vegetables-Image-Recognition-Dataset.git
cd Fruits-and-Vegetables-Image-Recognition-Dataset
```
### Install Dependencies
Ensure you have Python and the required libraries installed:
```bash
pip install numpy pandas tensorflow keras matplotlib seaborn
```
### Train the Model
Run the training script:
```bash
python Fruits_and_Vegetables_Image_Recognition_Dataset.py
```
Or open the Jupyter Notebook:
```bash
jupyter notebook Fruits_and_Vegetables_Image_Recognition_Dataset.ipynb
```

## 📊 Model Training Visualization
During training, the model performance is visualized using Matplotlib. 
The following metrics are plotted:
- **Training vs. Validation Accuracy**
- **Training vs. Validation Loss**

## Future Improvements
- Implement data augmentation for improved generalization.
- Fine-tune the CNN with pre-trained models like MobileNetV2.
- Develop a user-friendly web interface for real-time classification.

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.
