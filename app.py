import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from mtcnn import MTCNN

# -------------------------
# Load CSS + ASSETS
# -------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.image("assets/logo.png", width=150)
st.image("assets/banner.jpg", use_column_width=True)

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = "model/vitamin_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------
# Class Labels
# -------------------------
CLASSES = [
    "Vitamin A Deficiency",
    "Vitamin B12 Deficiency",
    "Vitamin C Deficiency",
    "Vitamin D Deficiency",
    "Iron Deficiency",
    "Healthy / No Deficiency"
]

RECOMMEND = {
    "Vitamin A Deficiency": ["Carrots", "Sweet Potatoes", "Spinach", "Eggs"],
    "Vitamin B12 Deficiency": ["Fish", "Eggs", "Milk", "Chicken"],
    "Vitamin C Deficiency": ["Oranges", "Lemon", "Strawberries", "Broccoli"],
    "Vitamin D Deficiency": ["Sunlight 20 minutes daily", "Fortified Milk", "Egg Yolks"],
    "Iron Deficiency": ["Spinach", "Beetroot", "Dates", "Red Meat"],
    "Healthy / No Deficiency": ["You are healthy! Maintain a balanced diet."]
}

SYMPTOMS = {
    "Vitamin A Deficiency": "Dry skin, night blindness, eye dryness",
    "Vitamin B12 Deficiency": "Fatigue, pale lips, tingling sensation",
    "Vitamin C Deficiency": "Low immunity, gum bleeding, fatigue",
    "Vitamin D Deficiency": "Weak bones, fatigue, muscle weakness",
    "Iron Deficiency": "Pale skin, dizziness, weakness",
    "Healthy / No Deficiency": "No symptoms detected. You appear healthy."
}

# -------------------------
# Face Detector
# -------------------------
detector = MTCNN()

# -------------------------
# Image Enhancement
# -------------------------
def enhance_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# -------------------------
# Preprocess for Model
# -------------------------
def prepare_for_model(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype("float32") / 255.0
    return np.expand_dims(face_img, axis=0)

# -------------------------
# Prediction
# -------------------------
def predict_deficiency(face_img):
    processed = prepare_for_model(face_img)
    preds = model.predict(processed)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100
    return CLASSES[class_id], confidence, preds

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧬 AI-Based Vitamin Deficiency Detector")

uploaded = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    faces = detector.detect_faces(img_np)

    if len(faces) == 0:
        st.error("No face detected! Try another image.")
    else:
        for face in faces:
            x, y, w, h = face["box"]
            face_img = img_np[y:y+h, x:x+w]
            enhanced_face = enhance_image(face_img)

            st.image(cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB), caption="Processed Face", width=260)

            label, confidence, all_probs = predict_deficiency(enhanced_face)

            st.write(f"<div class='result-card'><h3>{label}</h3>", unsafe_allow_html=True)
            st.info(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))

            st.subheader("Full Vitamin Probability:")
            for c, prob in zip(CLASSES, all_probs):
                st.write(f"{c} → {prob*100:.1f}%")
                st.progress(int(prob*100))

            st.subheader("Symptoms")
            st.write(SYMPTOMS[label])

            st.subheader("Recommended Foods")
            for food in RECOMMEND[label]:
                st.write(f"✔ {food}")

