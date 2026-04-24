import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Art Digitizer Pipeline")

# --- UI: UPLOAD ---
uploaded_file = st.file_uploader("Upload an image of the artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Upload", use_column_width=True)

    if st.button("Test Core Pipeline"):
        with st.spinner('Processing...'):
            # Just passing the image through to prove the UI works
            result_image = image
            st.success("Core UI and OpenCV environment are running!")

            st.image(result_image, caption="Processed Result", use_column_width=True)
            
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Artwork",
                data=byte_im,
                file_name="digitized_art.png",
                mime="image/png"
            )
