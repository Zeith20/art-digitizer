import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import traceback

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Smart Art Digitizer")

# --- OPTIONAL DEPENDENCIES ---
@st.cache_resource
def load_rembg():
    try:
        from rembg import remove
        return remove, True
    except Exception: return None, False

rembg_remove, REMBG_AVAILABLE = load_rembg()

@st.cache_resource
def load_coords():
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        return streamlit_image_coordinates, True
    except Exception: return None, False

streamlit_image_coordinates, IMAGE_COORDS_AVAILABLE = load_coords()

# --- SERVER-SIDE IMAGE CACHING (PREVENTS HANGS) ---
@st.cache_data
def process_bg_removal(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if REMBG_AVAILABLE:
        return rembg_remove(img)
    return img

@st.cache_data
def get_warped_result(img_bytes, pts, scale_ratio):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pts_scaled = np.array(pts, dtype="float32") * scale_ratio
    # order points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_scaled.sum(axis=1)
    rect[0] = pts_scaled[np.argmin(s)]
    rect[2] = pts_scaled[np.argmax(s)]
    diff = np.diff(pts_scaled, axis=1)
    rect[1] = pts_scaled[np.argmin(diff)]
    rect[3] = pts_scaled[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(w1), int(w2))
    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(h1), int(h2))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def find_corners_advanced(pill_image_with_alpha):
    img_np = np.array(pill_image_with_alpha)
    if img_np.shape[2] < 4: return None
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c).reshape(-1, 2)
    s, diff = hull.sum(axis=1), np.diff(hull, axis=1)
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]], dtype="float32")

# Session State (Only tiny metadata here)
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'scanned_keys' not in st.session_state: st.session_state.scanned_keys = set()
if 'last_click' not in st.session_state: st.session_state.last_click = None

try:
    # 1. LIGHTWEIGHT UPLOAD
    uploaded_files = st.file_uploader("Upload drawings", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        num_files = len(uploaded_files)
        st.session_state.current_index = max(0, min(st.session_state.current_index, num_files - 1))
        
        # NAVIGATION
        c_nav1, c_nav2, c_nav3 = st.columns([1, 1, 1])
        with c_nav1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with c_nav2: st.write(f"**{st.session_state.current_index + 1} / {num_files}**")
        with c_nav3:
            if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
                st.session_state.current_index += 1
                st.rerun()

        current_file = uploaded_files[st.session_state.current_index]
        file_key = f"{current_file.name}_{current_file.size}"
        file_bytes = current_file.getvalue()

        # FAST UI RESIZE
        img_temp = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = img_temp.size
        scale_ratio = w / 450
        ui_image = img_temp.resize((450, int(h / scale_ratio)))

        # --- STEP 1: AI SCAN ---
        if file_key not in st.session_state.scanned_keys:
            st.image(ui_image, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing...'):
                        cleansed = process_bg_removal(file_bytes)
                        pts = find_corners_advanced(cleansed)
                        if pts is not None:
                            st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts]
                        st.session_state.scanned_keys.add(file_key)
                        st.rerun()
            else: st.error("AI Library unavailable.")

        # --- STEP 2: RESULTS ---
        else:
            pts = st.session_state.points_map.get(file_key, [])
            
            # Decide what to show as result
            if len(pts) == 4:
                result_img = get_warped_result(file_bytes, pts, scale_ratio)
                st.subheader("✨ Perspective Corrected Scan")
            else:
                result_img = process_bg_removal(file_bytes)
                st.subheader("✨ Background Removed Cutout")
            
            st.image(result_img, use_container_width=True)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button("💾 Download", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
            with c_res2:
                if st.button("🔄 Reset This Image", use_container_width=True):
                    st.session_state.scanned_keys.remove(file_key)
                    if file_key in st.session_state.points_map: del st.session_state.points_map[file_key]
                    st.rerun()

            # --- STEP 3: MANUAL TOOL (ISOLATED) ---
            @st.fragment
            def manual_tool():
                st.divider()
                with st.expander("📍 Fine-tune Corners"):
                    curr_pts = st.session_state.points_map.get(file_key, [])
                    temp_draw = ui_image.copy()
                    draw = ImageDraw.Draw(temp_draw)
                    for i, p in enumerate(curr_pts):
                        draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                    
                    value = streamlit_image_coordinates(temp_draw, key=f"coords_{file_key}")
                    if value:
                        click = (int(value["x"]), int(value["y"]))
                        if click != st.session_state.last_click:
                            st.session_state.last_click = click
                            if len(curr_pts) < 4:
                                if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                                st.session_state.points_map[file_key].append(click)
                            else:
                                st.session_state.points_map[file_key] = [click]
                            st.rerun()

                    if len(st.session_state.points_map.get(file_key, [])) == 4:
                        if st.button("🚀 Re-Apply Points", use_container_width=True, type="primary"):
                            st.rerun() # This triggers the main logic to call get_warped_result
            manual_tool()

    else: st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 Connection or Processing Error.")
    st.code(traceback.format_exc())
