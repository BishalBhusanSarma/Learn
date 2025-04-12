import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set page config
st.set_page_config(page_title="Cat Detector", layout="centered")

# Title
st.title("🐱 Is it a Cat?")
st.write("Draw something below or upload an image and click **Submit** to check if it looks like a **cat**!")

# Load QuickDraw "cat" dataset
category = "cat"
data = np.load(f"quickdraw_data/{category}.npy")[:1000] / 255.0
labels = np.ones((data.shape[0],))  # Label 1 for cat

# Add noise for "not cat" samples
not_cat = np.random.rand(1000, 784)  # Random noise for non-cat images
X = np.concatenate([data, not_cat])
y = np.concatenate([labels, np.zeros((1000,))])  # Label 0 for not cat

# Reshape data to fit CNN input format (28x28x1)
X = X.reshape(-1, 28, 28, 1)

# One-hot encode the labels for categorical crossentropy loss
y = to_categorical(y, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Dropout to avoid overfitting
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two classes: Cat or Not Cat
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, validation_split=0.2)

# ✅ Evaluate model
loss, acc = model.evaluate(X_test, y_test)
st.write(f"Test Accuracy: {acc:.2f}")

# Choice: draw or upload
option = st.radio("Choose input method:", ("🎨 Draw on canvas", "📁 Upload an image"))

image_input = None

if option == "🎨 Draw on canvas":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=3,  # smaller brush
        stroke_color="black",
        background_color="white",
        height=512,
        width=512,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        image_input = img

elif option == "📁 Upload an image":
    uploaded_file = st.file_uploader("Upload a drawing or sketch", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image_input = Image.open(uploaded_file)

# Prediction
if image_input is not None:
    gray_img = image_input.convert("L").resize((28, 28))  # Convert to grayscale and resize
    img_array = np.array(gray_img).reshape(1, 28, 28, 1) / 255.0  # Reshape for CNN input

    st.image(image_input.resize((150, 150)), caption="Input Image")

    if st.button("Submit"):
        pred = model.predict(img_array)  # Predict using CNN
        cat_prob = pred[0][1] * 100
        not_cat_prob = pred[0][0] * 100
        st.markdown(f"### 🐱 Cat: `{cat_prob:.2f}%`")
        st.markdown(f"### ❌ Not Cat: `{not_cat_prob:.2f}%`")
