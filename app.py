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

# --- UTILS & ANALYSIS ---
def analyze_shape(pill_image_with_alpha):
    """Detects if artwork is rectangular or irregular."""
    img_np = np.array(pill_image_with_alpha)
    if img_np.shape[2] < 4: return None, "none"
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, "none"
    c = max(cnts, key=cv2.contourArea)
    
    # Rectangularity check
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    rect_area = rect[1][0] * rect[1][1]
    is_rectangular = (rect_area > 0 and (area / rect_area) > 0.82)
    
    if is_rectangular:
        hull = cv2.convexHull(c).reshape(-1, 2)
        s, d = hull.sum(axis=1), np.diff(hull, axis=1)
        pts = np.array([hull[np.argmin(s)], hull[np.argmin(d)], hull[np.argmax(s)], hull[np.argmax(d)]], dtype="float32")
        return pts, "rectangle"
    return None, "irregular"

@st.cache_data
def get_cutout(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return rembg_remove(img) if REMBG_AVAILABLE else img

@st.cache_data
def get_flattened(img_bytes, pts, scale):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pts_scaled = np.array(pts, dtype="float32") * scale
    
    # order points
    rect = np.zeros((4, 2), dtype="float32")
    s, d = pts_scaled.sum(axis=1), np.diff(pts_scaled, axis=1)
    rect[0], rect[2] = pts_scaled[np.argmin(s)], pts_scaled[np.argmax(s)]
    rect[1], rect[3] = pts_scaled[np.argmin(d)], pts_scaled[np.argmax(d)]
    
    (tl, tr, br, bl) = rect
    w = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    h = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_cv, M, (w, h))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# Session State
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'scanned' not in st.session_state: st.session_state.scanned = set()
if 'last_click' not in st.session_state: st.session_state.last_click = None

try:
    # BULK UPLOAD
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

        # UI Resize
        img_p = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = img_p.size
        scale = w / 450
        ui_img = img_p.resize((450, int(h / scale)))

        # --- STEP 1: AI SCAN ---
        if file_key not in st.session_state.scanned:
            st.image(ui_img, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing shape...'):
                        cleansed = get_cutout(file_bytes)
                        pts, shape = analyze_shape(cleansed)
                        if shape == "rectangle" and pts is not None:
                            st.session_state.points_map[file_key] = [(int(p[0]/scale), int(p[1]/scale)) for p in pts]
                        st.session_state.scanned.add(file_key)
                        st.rerun()
            else: st.error("AI Unavailable.")

        # --- STEP 2: RESULTS ---
        else:
            pts = st.session_state.points_map.get(file_key, [])
            if len(pts) == 4:
                res = get_flattened(file_bytes, pts, scale)
                st.subheader("✨ Perspective Scan")
            else:
                res = get_cutout(file_bytes)
                st.subheader("✨ Clean Cutout")
            
            st.image(res, use_container_width=True)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                buf = io.BytesIO(); res.save(buf, format="PNG")
                st.download_button("💾 Download", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
            with c_res2:
                if st.button("🔄 Reset Image", use_container_width=True):
                    st.session_state.scanned.remove(file_key)
                    if file_key in st.session_state.points_map: del st.session_state.points_map[file_key]
                    st.rerun()

            # --- STEP 3: MANUAL ---
            @st.fragment
            def manual_tool():
                st.divider()
                with st.expander("📍 Fine-tune Corners"):
                    cur_pts = st.session_state.points_map.get(file_key, [])
                    tmp = ui_img.copy(); draw = ImageDraw.Draw(tmp)
                    for i, p in enumerate(cur_pts): draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                    
                    val = streamlit_image_coordinates(tmp, key=f"c_{file_key}")
                    if val:
                        clk = (int(val["x"]), int(val["y"]))
                        if clk != st.session_state.last_click:
                            st.session_state.last_click = clk
                            if len(cur_pts) < 4:
                                if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                                st.session_state.points_map[file_key].append(clk)
                            else: st.session_state.points_map[file_key] = [clk]
                            st.rerun()
                    if len(cur_pts) == 4 and st.button("🚀 Apply Points", use_container_width=True, type="primary"): st.rerun()
            manual_tool()

    else: st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 Error occurred. Try clearing app cache.")
    st.code(traceback.format_exc())
