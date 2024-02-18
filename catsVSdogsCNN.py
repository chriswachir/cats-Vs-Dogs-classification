import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cats_vs_dogs.h1')

st.title("Cats vs Dogs Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width=250)
    
    # Preprocess the image
    img = img.resize((150, 150))
    image_array = np.array(img)
    img = np.expand_dims(image_array, axis=0)
    img = img/255

    # Make a prediction
    prediction = model.predict(img)
    TH = 0.5
    prediction = int(prediction[0][0]>TH)
    classes = {0: 'cat', 1: 'dog'}

    st.write(f'The model predicts this image is a: {classes[prediction]}')
