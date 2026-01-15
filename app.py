import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="COVID-19 X-ray Classification", layout="centered")

st.title("🫁 COVID-19 Radiography Classification")
st.write("Upload a chest X-ray image to predict the disease class")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_cnn_model.h5")

model = load_model()

# ⚠️ IMPORTANT: class order must match training order
CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

IMAGE_SIZE = 128

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.image(img, caption="Uploaded Image", channels="BGR", use_container_width=True)

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # Predict
    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
