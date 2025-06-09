import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

def load_model(model_path):
    param = np.load(model_path)
    w1 = param['w1']
    b1 = param['b1']
    w2 = param['w2']
    b2 = param['b2']
    return w1, b1, w2, b2

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(x_test, w1, b1, w2, b2):
    # Forward
    # Layer 1
    z1 = np.dot(x_test, w1) + b1
    a1 = relu(z1)
    # Layer 2
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    return a2.flatten() * 100

def canvas_config():
    canvas_result = st_canvas(
        height=280,
        width=280,
        fill_color="#000000",
        stroke_width=30,
        stroke_color="#FFFFFF",
        background_color="#000000",
        drawing_mode="freedraw",
        key="canvas",
    )
    return canvas_result

def main():
    w1, b1, w2, b2 = load_model('./data/model_trained.npz')

    st.set_page_config(page_title="Canvas Demo", layout="centered")
    st.title("Neural Network Demo")
    canvas_result = canvas_config()

    if st.button("Analyze"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data[:, :, 0] # Channel 0

            # Gaussian Blurr
            img_blurr = cv2.GaussianBlur(img, (3, 3), 0.2)

            # Resize to 28x28
            img_small = cv2.resize(img_blurr, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalisasi to 0-1
            img_norm = img_small / 255.0
            
            st.image(img_norm, width=140, clamp=True, channels="GRAY")
            
            # Flatten (1 x 784)
            input_array = img_norm.flatten().reshape(1, 784)

            predictions = predict(input_array, w1, b1, w2, b2)
            st.bar_chart({str(i): predictions[i] for i in range(10)})
    else:
        st.warning("Fill the canvas")

if __name__ == "__main__":
    main()
