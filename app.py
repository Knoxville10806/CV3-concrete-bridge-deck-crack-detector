import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["Cracked", "Non-cracked"]
MODEL_PATH = "models/crack_detection_mobilenetv3.keras"

st.set_page_config(page_title="Concrete Bridge Deck Crack Detector", page_icon="🌉", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    return CLASS_NAMES[predicted_idx], probs

st.title("Concrete Bridge Deck Crack Detector")
st.caption(
    "A coursework CNN model (GET 324) trained to detect cracks in concrete bridge deck photos. "
    "Educational project only — not a substitute for professional structural inspection."
)

model = load_model()
uploaded_file = st.file_uploader("Upload a concrete surface photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    label, probs = predict(model, img)
    st.write(f"**Prediction: {label}**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(int(p * 100), text=f"{name}: {p * 100:.1f}%")
