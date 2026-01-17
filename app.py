import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("🧠 Handwritten Digit Recognition (MNIST)")

st.write("""
This app trains a simple neural network on the MNIST dataset (70,000 handwritten digits).
You can test accuracy and upload your own digit image for prediction.
""")

# -------------------------------
# Load and preprocess MNIST data
# -------------------------------
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Flatten
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# -------------------------------
# Build Model
# -------------------------------
def build_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------
# Train Model
# -------------------------------
if st.button("Train Model"):
    model = build_model()
    with st.spinner("Training the model..."):
        history = model.fit(x_train, y_train, epochs=5,
                            validation_data=(x_test, y_test),
                            verbose=0)
    st.success("✅ Training complete!")

    # Plot accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Test Accuracy')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"### Final Test Accuracy: {test_acc:.4f}")

    # Save model for later use
    model.save("mnist_model.h5")

# -------------------------------
# Upload Image for Prediction
# -------------------------------
st.write("## 🔍 Try Your Own Digit")
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load model
    try:
        model = tf.keras.models.load_model("mnist_model.h5")
    except:
        st.error("⚠️ Please train the model first!")
        st.stop()

    # Process uploaded image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((28, 28))  # resize to 28x28
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 784)

    # Prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.image(img, caption=f"Uploaded Digit", width=150)
    st.write(f"### Predicted Digit: {predicted_digit}")
