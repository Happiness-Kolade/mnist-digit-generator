import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train / 255.0  # normalize

# Title
st.title("🖊️ MNIST Digit Generator")

# Digit selection
digit = st.selectbox("Select a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    # Find all images of that digit
    indices = np.where(y_train == digit)[0]
    selected_indices = np.random.choice(indices, 5, replace=False)
    selected_images = x_train[selected_indices]

    # Display images side by side
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(selected_images[i], cmap="gray")
        axs[i].axis("off")

    st.pyplot(fig)
