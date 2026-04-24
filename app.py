import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
from streamlit_image_coordinates import streamlit_image_coordinates

def order_points(pts):
    """Orders coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Applies a perspective warp to flatten the image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Art Digitizer Pipeline")

# Initialize memory to store screen taps
if 'points' not in st.session_state:
    st.session_state.points = []

# --- UI: UPLOAD ---
uploaded_file = st.file_uploader("Upload an image of the artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Add a reset button to clear taps
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("📍 **Tap the 4 corners of the drawing.**")
    with col2:
        if st.button("Reset Taps"):
            st.session_state.points = []
            st.rerun()

    # Draw red dots where the user taps for visual feedback
    img_to_draw = image.copy()
    draw = ImageDraw.Draw(img_to_draw)
    for p in st.session_state.points:
        x, y = p
        r = min(image.size) * 0.02 # Responsive dot size
        draw.ellipse((x-r, y-r, x+r, y+r), fill='red')

    # Display the interactive image
    value = streamlit_image_coordinates(img_to_draw, key="coords", width='stretch')

    # Capture the tap coordinates
    if value is not None:
        point = (value["x"], value["y"])
        # Only add point if it's new and we have less than 4
        if point not in st.session_state.points and len(st.session_state.points) < 4:
            st.session_state.points.append(point)
            st.rerun()

    # Once 4 corners are tapped, reveal the processing button
    if len(st.session_state.points) == 4:
        st.success("4 corners selected! Ready to process.")
        
        if st.button("Crop & Flatten Image"):
            with st.spinner('Warping geometry...'):
                
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pts = np.array(st.session_state.points, dtype="float32")
                
                # Apply the warp based on your exact taps
                warped_cv = four_point_transform(image_cv, pts)
                
                result_image = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                
                st.subheader("Final Result")
                st.image(result_image, width='stretch')
                
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Cleaned Artwork",
                    data=byte_im,
                    file_name="digitized_art.png",
                    mime="image/png"
                )
