import streamlit as st
from PIL import Image
import numpy as np
import cv2
from rembg import remove
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

    # --- UI: SELECT TRACK ---
    st.subheader("Select Processing Method")
    method = st.radio(
        "Choose how to digitize this item:",
        ("AI Cutout (Irregular Shapes / Plates)", "Auto-Scan Rectangle (Coming Soon)")
    )

    if st.button("Process Image"):
        with st.spinner('Processing...'):
            
            if method == "AI Cutout (Irregular Shapes / Plates)":
                result_image = remove(image)
                st.success("Background removed successfully!")
            else:
                result_image = image
                st.info("OpenCV logic will be added here next.")

            # --- UI: RESULTS & DOWNLOAD ---
            st.image(result_image, caption="Processed Result", use_column_width=True)
            
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Cleaned Artwork",
                data=byte_im,
                file_name="digitized_art.png",
                mime="image/png"
            )
