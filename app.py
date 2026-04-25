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
    """Smarter detection using adaptive thresholding, morphology, and convex hull."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Denoise while preserving edges
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Try multiple thresholding methods
    # Method A: Adaptive Thresholding (good for paper/shadows)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh) # Invert
    
    # 3. Morphological closing to fill gaps in edges
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        # Method B: Fallback to Canny
        edged = cv2.Canny(blur, 50, 150)
        dilated = cv2.dilate(edged, kernel, iterations=1)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    # Get the largest contour by area
    c = max(cnts, key=cv2.contourArea)
    
    # 4. Try to approximate to 4 points
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # 5. If it's a "custom" or noisy shape, use Convex Hull and find 4 extreme corners
    # This is much smarter for non-squared/irregular paper
    hull = cv2.convexHull(c).reshape(-1, 2)
    
    # Find points that minimize/maximize sum and difference (standard corner detection)
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    
    # Basic sanity check: ensure the area isn't tiny
    if cv2.contourArea(np.array([tl, tr, br, bl])) < (image.width * image.height * 0.05):
        return None
        
    return np.array([tl, tr, br, bl])

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Art Digitizer Pipeline")

# Initialize session state
if 'points_map' not in st.session_state:
    st.session_state.points_map = {}
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# --- UI: UPLOAD ---
uploaded_files = st.file_uploader("Upload images of the artwork", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Navigation and Progress
    num_files = len(uploaded_files)
    st.session_state.current_index = min(st.session_state.current_index, num_files - 1)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with col_nav2:
        st.write(f"**Image {st.session_state.current_index + 1} of {num_files}**")
        st.caption(f"File: {uploaded_files[st.session_state.current_index].name}")
    with col_nav3:
        if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
            st.session_state.current_index += 1
            st.rerun()

    current_file = uploaded_files[st.session_state.current_index]
    file_key = f"{current_file.name}_{current_file.size}"
    
    image = Image.open(current_file)
    MOBILE_WIDTH = 350
    scale_ratio = image.width / MOBILE_WIDTH
    new_height = int(image.height / scale_ratio)
    ui_image = image.resize((MOBILE_WIDTH, new_height))

    # Auto-detection for new files in the batch
    if file_key not in st.session_state.points_map:
        auto_pts = auto_detect_corners(image)
        if auto_pts is not None:
            ordered_auto_pts = order_points(auto_pts)
            st.session_state.points_map[file_key] = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in ordered_auto_pts]
            st.toast(f"✅ Auto-detected corners for {current_file.name}")
        else:
            st.session_state.points_map[file_key] = []
    
    # UI for current image
    col1, col2 = st.columns([3, 1])
    with col1:
        if not st.session_state.points_map[file_key]:
            st.warning("⚠️ **Auto-detection failed.** Please tap the 4 corners manually.")
        else:
            st.write("📍 **Verify or tap the 4 corners.**")
    with col2:
        if st.button("Reset Points"):
            st.session_state.points_map[file_key] = []
            st.rerun()

    # Draw points
    img_to_draw = ui_image.copy()
    draw = ImageDraw.Draw(img_to_draw)
    for p in st.session_state.points_map[file_key]:
        x, y = p
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill='red')

    # Interaction
    value = streamlit_image_coordinates(img_to_draw, key=f"coords_{file_key}")
    if value is not None:
        point = (value["x"], value["y"])
        if point not in st.session_state.points_map[file_key] and len(st.session_state.points_map[file_key]) < 4:
            st.session_state.points_map[file_key].append(point)
            st.rerun()

    # Processing
    if len(st.session_state.points_map[file_key]) == 4:
        st.success("4 corners selected!")
        if st.button("Crop & Flatten This Image"):
            with st.spinner('Warping geometry...'):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pts = np.array(st.session_state.points_map[file_key], dtype="float32")
                pts_scaled = pts * scale_ratio 
                warped_cv = four_point_transform(image_cv, pts_scaled)
                result_image = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                
                st.subheader("Final Result")
                st.image(result_image, use_container_width=True)
                
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                st.download_button(
                    label=f"Download {current_file.name} (Processed)",
                    data=buf.getvalue(),
                    file_name=f"digitized_{current_file.name}.png",
                    mime="image/png"
                )
else:
    st.info("Please upload one or more images to get started.")
