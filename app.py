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
    """Applies perspective warp."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def auto_detect_corners(image):
    """Refined detection that ignores the image boundary."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    # FILTER: Remove contours that are essentially the image border
    filtered_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h * w * 0.95) and area > (h * w * 0.05): # Not the whole image, but significant
            filtered_cnts.append(c)
            
    if not filtered_cnts: return None
    
    c = max(filtered_cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    hull = cv2.convexHull(c).reshape(-1, 2)
    s, diff = hull.sum(axis=1), np.diff(hull, axis=1)
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]])

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Art Digitizer Pipeline")

# Initialize session state
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0

# --- UI: CONFIG ---
mode = st.radio("Select Processing Mode:", ["Standard Paper (Warp)", "Odd Shape (AI Cutout)"], horizontal=True)

# --- UI: UPLOAD ---
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.session_state.current_index = min(st.session_state.current_index, num_files - 1)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Prev") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with col_nav2:
        st.write(f"**{st.session_state.current_index + 1} / {num_files}**")
        st.caption(uploaded_files[st.session_state.current_index].name)
    with col_nav3:
        if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
            st.session_state.current_index += 1
            st.rerun()

    current_file = uploaded_files[st.session_state.current_index]
    file_key = f"{current_file.name}_{current_file.size}"
    image = Image.open(current_file)
    scale_ratio = image.width / 350
    ui_image = image.resize((350, int(image.height / scale_ratio)))

    if mode == "Standard Paper (Warp)":
        if file_key not in st.session_state.points_map:
            auto_pts = auto_detect_corners(image)
            if auto_pts is not None:
                ordered_auto_pts = order_points(auto_pts)
                st.session_state.points_map[file_key] = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in ordered_auto_pts]
                st.toast("✅ Auto-detected paper!")
            else:
                st.session_state.points_map[file_key] = []

        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.points_map[file_key]:
                st.warning("⚠️ **Detection failed.** Tap 4 corners manually.")
            else:
                st.write("📍 **Verify the 4 corners.**")
        with col2:
            if st.button("Reset"):
                st.session_state.points_map[file_key] = []
                st.rerun()

        img_to_draw = ui_image.copy()
        draw = ImageDraw.Draw(img_to_draw)
        for p in st.session_state.points_map[file_key]:
            x, y = p
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red')

        value = streamlit_image_coordinates(img_to_draw, key=f"coords_{file_key}")
        if value is not None:
            point = (value["x"], value["y"])
            if point not in st.session_state.points_map[file_key] and len(st.session_state.points_map[file_key]) < 4:
                st.session_state.points_map[file_key].append(point)
                st.rerun()

        if len(st.session_state.points_map[file_key]) == 4:
            if st.button("Process & Flatten"):
                with st.spinner('Warping...'):
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
                    result_cv = four_point_transform(image_cv, pts_scaled)
                    result_image = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
                    st.image(result_image, use_container_width=True)
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button("Download Scan", buf.getvalue(), f"scan_{current_file.name}.png", "image/png")

    else: # AI Cutout Mode
        st.info("💡 AI will remove the background. This may take a few seconds.")
        st.image(ui_image, use_container_width=True)
        if st.button("Remove Background"):
            with st.spinner('AI analyzing layers...'):
                try:
                    from rembg import remove
                    result_image = remove(image)
                    st.image(result_image, use_container_width=True)
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button("Download Cutout", buf.getvalue(), f"cutout_{current_file.name}.png", "image/png")
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("Please upload images to get started.")
