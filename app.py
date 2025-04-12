import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load model
model = load_model("model.h5")

st.title("🐱 Is It a Cat? - Doodle Classifier")
st.write("Draw a doodle or upload a 28x28 image to see if it's a cat.")

# ---- DRAWING CANVAS ----
st.subheader("✍️ Draw Here (Black on White)")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Black ink
    stroke_width=10,
    stroke_color="#000000",
    background_color="#ffffff",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---- IMAGE UPLOAD ----
st.subheader("📤 Or Upload an Image")

uploaded_file = st.file_uploader("Choose a 28x28 image file (optional)", type=["png", "jpg", "jpeg"])

# ---- Predict Function ----
def predict(image):
    img = image.convert("L").resize((28, 28))  # grayscale and resize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)[0]
    cat_prob = prediction[1] * 100
    not_cat_prob = prediction[0] * 100
    return f"🐱 Cat: {cat_prob:.2f}%\n❌ Not Cat: {not_cat_prob:.2f}%"

# ---- Handle Input ----
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=140)
    st.write(predict(image))

elif canvas_result.image_data is not None:
    # Use the canvas image only if it’s not blank
    image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))  # invert so black = doodle
    st.image(image, caption="Drawn Image", width=140)
    st.write(predict(image))
