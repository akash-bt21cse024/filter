import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np
import io

st.set_page_config(page_title="FILTER IMAGE", layout='centered')
st.title("Image Upload")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Function to convert NumPy array to bytes for download
def numpy_array_to_bytes(np_array, format='PNG'):
    image = Image.fromarray(np_array)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

# Download function
def download(img, file_name):
    image_bytes = numpy_array_to_bytes(img, format='PNG')
    st.download_button(
        label="Download Image",
        data=image_bytes,
        file_name=file_name,
        mime="image/png"
    )

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = np.array(img)
    st.image(img, caption="Original Image", width=400)

    # Tabs for filters
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['BLUR', 'BRIGHTNESS', 'WARM', 'COOL', 'FACE BLUR', 'SKETCH'])
    
    with tab1:
        num_points = st.slider("Number of blur points", min_value=0, max_value=10, value=0)
        blr_3 = cv.blur(img, (4*num_points, 4*num_points)) if num_points else img
        st.image(blr_3, caption="Blurred Image", width=400)
        download(blr_3, "blurred_image.png")

    with tab2:
        num_points = st.slider("Brightness adjustment", min_value=-10, max_value=10, value=0)
        img_1 = cv.convertScaleAbs(img, alpha=1, beta=num_points * 10)
        st.image(img_1, caption="Brightness Adjusted", width=400)
        download(img_1, "brightness_adjusted_image.png")

    with tab3:
        num_points = st.slider("Warm effect points", min_value=0, max_value=8, value=0)
        warm_color = [255, 250, 205]
        background = np.full_like(img, warm_color, dtype=np.uint8)
        merge = cv.addWeighted(img, 0.1*(10-num_points), background, 0.1*num_points, 0)
        st.image(merge, caption="Warm Effect", width=400)
        download(merge, "warm_effect_image.png")

    with tab4:
        num_points = st.slider("Cool effect points", min_value=0, max_value=9, value=0)
        cool_color = [0, 0, 255]
        background = np.full_like(img, cool_color, dtype=np.uint8)
        merged = cv.addWeighted(img, 0.1*(10-num_points), background, 0.1*num_points, 0)
        st.image(merged, caption="Cool Effect", width=400)
        download(merged, "cool_effect_image.png")

    with tab5:
        classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        if classifier.empty():
            st.error("Haarcascade file not found. Ensure the file path is correct.")
        else:
            faces = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                st.warning("No faces detected.")
            else:
                x, y, w, h = faces[0]  # Assuming the first detected face is the main one
                num_points = st.slider("Face blur intensity", min_value=1, max_value=10, value=1)
                face = img[y:y+h, x:x+w]
                face = cv.blur(face, (4*num_points, 4*num_points))
                img[y:y+h, x:x+w] = face
                st.image(img, caption="Face Blurred", width=400)
                download(img, "face_blurred_image.png")

    with tab6:
        num_points = st.slider("Sketch intensity", min_value=1, max_value=5, value=2)
        blr = cv.blur(img, (6, 6))
        edge_1 = cv.Canny(blr, 25*num_points, 30*num_points)
        st.image(edge_1, caption="Sketch Effect", width=400)
        download(edge_1, "sketch_effect_image.png")
