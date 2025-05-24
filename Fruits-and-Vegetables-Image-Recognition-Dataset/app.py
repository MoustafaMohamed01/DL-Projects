import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model = load_model("C:\AI\Projetcs\Fruits and Vegetables Recognition\Changes\Fruit& Vegetables Image Classifier.keras")
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
 'tomato', 'turnip', 'watermelon']

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        .stApp {
            max-width: 700px;
            margin: auto;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🍏 Fruits & Vegetables Recognition 🍎</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = uploaded_file
    image_load = tf.keras.utils.load_img(image, target_size=(180,180))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat = tf.expand_dims(img_arr, axis=0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    st.image(image)
    st.markdown(f"""
      <div style='text-align: center; font-size: 24px; font-weight: bold; color: #27ae60;'>
           Prediction: {data_cat[np.argmax(score)]} 
        </div>
      <div style='text-align: center; font-size: 20px; color: #8e44ad;'>
        Confidence: {np.max(score) * 100:.2f}%
       </div>
    """, unsafe_allow_html=True)

