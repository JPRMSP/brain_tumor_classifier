import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/brain_tumor_classifier.keras")

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (128, 128))  # Resize
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Reshape for model input
    return image

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to check for a brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Show prediction result
    st.subheader("Prediction:")
    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
