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

def auto_detect_corners(image):
    """Attempts to automatically find the 4 corners of the artwork."""
    # Convert to grayscale and blur
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blur, 75, 200)
    
    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If our approximated contour has four points, we can assume we've found our screen
        if len(approx) == 4:
            return approx.reshape(4, 2)
            
    return None

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Art Digitizer Pipeline")

# Initialize memory to store screen taps
if 'points' not in st.session_state:
    st.session_state.points = []
# Initialize memory to track file changes so we can reset taps for new photos
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# --- UI: UPLOAD ---
uploaded_file = st.file_uploader("Upload an image of the artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Reset points if a new file is uploaded
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.points = []
        
        # --- AUTO DETECTION ---
        image = Image.open(uploaded_file)
        # --- SCALING FOR MOBILE UI ---
        MOBILE_WIDTH = 350
        scale_ratio = image.width / MOBILE_WIDTH
        
        auto_pts = auto_detect_corners(image)
        if auto_pts is not None:
            # Order them and scale down to UI coordinates
            ordered_auto_pts = order_points(auto_pts)
            st.session_state.points = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in ordered_auto_pts]
            st.toast("✅ Auto-detected corners!")
        
        st.rerun()

    image = Image.open(uploaded_file)
    
    # --- SCALING FOR MOBILE UI ---
    MOBILE_WIDTH = 350
    scale_ratio = image.width / MOBILE_WIDTH
    new_height = int(image.height / scale_ratio)
    ui_image = image.resize((MOBILE_WIDTH, new_height))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("📍 **Tap the 4 corners of the drawing.**")
    with col2:
        if st.button("Reset"):
            st.session_state.points = []
            st.rerun()

    # Draw red dots on the small UI image
    img_to_draw = ui_image.copy()
    draw = ImageDraw.Draw(img_to_draw)
    for p in st.session_state.points:
        x, y = p
        r = 5 # Fixed dot size for visibility on small screen
        draw.ellipse((x-r, y-r, x+r, y+r), fill='red')

    # Display the interactive image (now sized to fit your phone)
    value = streamlit_image_coordinates(img_to_draw, key="coords")

    # Capture the tap coordinates
    if value is not None:
        point = (value["x"], value["y"])
        if point not in st.session_state.points and len(st.session_state.points) < 4:
            st.session_state.points.append(point)
            st.rerun()

    # Once 4 corners are tapped, process using original resolution
    if len(st.session_state.points) == 4:
        st.success("4 corners selected! Ready to process.")
        
        if st.button("Crop & Flatten Image"):
            with st.spinner('Warping geometry...'):
                
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Scale the UI tap coordinates back up to the original high-res image
                pts = np.array(st.session_state.points, dtype="float32")
                pts_scaled = pts * scale_ratio 
                
                # Apply the warp
                warped_cv = four_point_transform(image_cv, pts_scaled)
                
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
