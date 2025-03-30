import streamlit as st
import gdown
import os
import numpy as np
import tensorflow as tf
import cv2

# Define the Google Drive file ID of the model
FILE_ID = "1YYreU6VFtIMqV59WZDGG39bfWA0-Dd3P"  # Replace with your actual File ID
MODEL_PATH = "brain_tumor_classifier.keras"

# Function to download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait ⏳")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully! ✅")

# Load the model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to predict if it has a tumor.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Preprocess
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dimension

    # Predict
    prediction = model.predict(img)
    result = "Tumor Detected ❗" if prediction[0][0] > 0.5 else "No Tumor ✅"

    st.image(img.reshape(128, 128), caption="Uploaded Image", use_column_width=True, channels="GRAY")
    st.write("### Prediction:", result)
