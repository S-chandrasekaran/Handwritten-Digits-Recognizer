import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

st.set_page_config(page_title="MNIST CNN Digit Recognition", page_icon="🧠", layout="centered")
st.title("🧠 Robust Handwritten Digit Recognition (CNN + Preprocessing)")

st.write("""
This app trains a **Convolutional Neural Network (CNN)** on MNIST and uses robust preprocessing
to handle real-world handwritten digits. It aims for **≥99% test accuracy** and better predictions
on uploaded images.
""")

# -------------------------------
# Load MNIST dataset
# -------------------------------
@st.cache_data
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize and add channel dimension
    x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
    x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist()

# -------------------------------
# Build CNN model
# -------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------
# Train or load model
# -------------------------------
@st.cache_resource
def train_or_load_model(epochs=6):
    try:
        return tf.keras.models.load_model("mnist_cnn_model.h5")
    except Exception:
        model = build_cnn()
        model.fit(x_train, y_train, epochs=epochs,
                  validation_data=(x_test, y_test), verbose=0)
        model.save("mnist_cnn_model.h5")
        return model

if st.button("Train / Load CNN Model"):
    with st.spinner("Training model..."):
        model = train_or_load_model(epochs=6)
    st.success("✅ Model ready")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"### Final Test Accuracy: {test_acc:.4f}")

# -------------------------------
# Preprocess uploaded image
# -------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28))  # resize
    arr = np.array(img).astype("float32")

    # Invert if background is dark
    if arr.mean() < 128:
        arr = 255 - arr

    arr = arr / 255.0
    arr = arr[..., np.newaxis]  # add channel
    return arr

# -------------------------------
# Prediction section
# -------------------------------
st.write("## 🔍 Try Your Own Digit")
uploaded = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
    except Exception:
        st.error("⚠️ Model not found. Please train or load the model above.")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=180)

    processed = preprocess_image(img)
    pred = model.predict(processed[np.newaxis, ...], verbose=0)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    st.write(f"### Predicted Digit: **{digit}**")
    st.write(f"**Confidence:** {confidence:.4f}")

    proc_vis = (processed.squeeze() * 255).astype(np.uint8)
    st.image(proc_vis, caption="Preprocessed (28×28) used for prediction", width=180)

# -------------------------------
# Quick MNIST sample test
# -------------------------------
st.write("## 🧪 Quick MNIST Sample Test")
idx = st.slider("Sample index", 0, len(x_test)-1, 123)
if st.button("Predict MNIST sample"):
    try:
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
    except Exception:
        st.error("⚠️ Model not found. Please train or load the model above.")
        st.stop()
    sample = x_test[idx]
    pred = model.predict(sample[np.newaxis, ...], verbose=0)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))
    st.image(sample.squeeze(), caption=f"MNIST Sample (Label: {y_test[idx]})", width=180, clamp=True)
    st.write(f"### Predicted: **{digit}**  |  **Confidence:** {confidence:.4f}")
