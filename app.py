import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

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

# --- UI: UPLOAD ---
uploaded_file = st.file_uploader("Upload an image of the artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Upload", use_container_width=True)

    if st.button("Run Auto-Scan (Rectangles)"):
        with st.spinner('Detecting edges and warping...'):
            
            # 1. Convert PIL image to OpenCV format (NumPy array in BGR)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 2. Resize for faster and more accurate edge detection
            ratio = image_cv.shape[0] / 500.0
            orig = image_cv.copy()
            image_resized = cv2.resize(image_cv, (int(image_cv.shape[1] / ratio), 500))

            # 3. Grayscale, blur, and find edges
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 75, 200)

            # 4. Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            screenCnt = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # If the contour has 4 points, we assume it's the paper
                if len(approx) == 4:
                    screenCnt = approx
                    break

            # 5. Apply warp if a rectangle was found
            if screenCnt is not None:
                warped_cv = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
                # Convert back to PIL for the UI
                result_image = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                st.success("Drawing successfully scanned and flattened!")
            else:
                result_image = image
                st.warning("Could not automatically find 4 corners. Showing original image.")

            # --- UI: RESULTS & DOWNLOAD ---
            st.image(result_image, caption="Processed Result", use_container_width=True)
            
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Cleaned Artwork",
                data=byte_im,
                file_name="digitized_art.png",
                mime="image/png"
            )
