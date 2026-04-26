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

# --- UTILS & CACHED PROCESSING ---
@st.cache_data
def get_cutout(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return rembg_remove(img) if REMBG_AVAILABLE else img

@st.cache_data
def analyze_shape(cutout_bytes, scale):
    """Detects if artwork is rectangular."""
    img_np = np.array(Image.open(io.BytesIO(cutout_bytes)))
    if img_np.shape[2] < 4: return None, "none"
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, "none"
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    area = cv2.contourArea(c)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area > 0 and (area / rect_area) > 0.82:
        hull = cv2.convexHull(c).reshape(-1, 2)
        s, d = hull.sum(axis=1), np.diff(hull, axis=1)
        raw_pts = np.array([hull[np.argmin(s)], hull[np.argmin(d)], hull[np.argmax(s)], hull[np.argmax(diff)] if 'diff' in locals() else hull[np.argmax(d)]], dtype="float32")
        # Fixed coordinate ordering logic
        s, d = hull.sum(axis=1), np.diff(hull, axis=1)
        tl, br = hull[np.argmin(s)], hull[np.argmax(s)]
        tr, bl = hull[np.argmin(d)], hull[np.argmax(d)]
        ui_pts = [(int(p[0]/scale), int(p[1]/scale)) for p in [tl, tr, br, bl]]
        return ui_pts, "rectangle"
    return None, "irregular"

@st.cache_data
def get_flattened(img_bytes, ui_pts, scale):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pts_scaled = np.array(ui_pts, dtype="float32") * scale
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
if 'processed' not in st.session_state: st.session_state.processed = set()
if 'last_click' not in st.session_state: st.session_state.last_click = None

try:
    # 1. FILE UPLOAD
    uploaded_file = st.file_uploader("Upload an image to digitize", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        file_bytes = uploaded_file.getvalue()

        img_raw = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = img_raw.size
        scale = w / 450
        ui_img = img_raw.resize((450, int(h / scale)))

        # --- STEP 1: AI SCAN ---
        if file_key not in st.session_state.processed:
            st.subheader("Original Image")
            st.image(ui_img, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing...'):
                        cutout = get_cutout(file_bytes)
                        buf = io.BytesIO(); cutout.save(buf, format="PNG")
                        pts, shape = analyze_shape(buf.getvalue(), scale)
                        if shape == "rectangle" and pts:
                            st.session_state.points_map[file_key] = pts
                        st.session_state.processed.add(file_key)
                        st.rerun()
            else: st.error("AI Library unavailable.")

        # --- STEP 2: RESULTS ---
        else:
            pts = st.session_state.points_map.get(file_key, [])
            if len(pts) == 4:
                res = get_flattened(file_bytes, pts, scale)
                st.subheader("✨ Perspective Corrected Scan")
            else:
                res = get_cutout(file_bytes)
                st.subheader("✨ Clean Cutout")
            
            st.image(res, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                buf = io.BytesIO(); res.save(buf, format="PNG")
                st.download_button("💾 Download PNG", buf.getvalue(), f"scan_{uploaded_file.name}.png", "image/png", use_container_width=True)
            with c2:
                if st.button("🔄 Reset Image", use_container_width=True):
                    st.session_state.processed.discard(file_key)
                    st.session_state.points_map.pop(file_key, None)
                    st.rerun()

            # --- STEP 3: MANUAL FIX ---
            @st.fragment
            def manual_tool():
                st.divider()
                with st.expander("📍 Fine-tune Corners"):
                    cur_pts = st.session_state.points_map.get(file_key, [])
                    tmp = ui_img.copy(); draw = ImageDraw.Draw(tmp)
                    for i, p in enumerate(cur_pts):
                        draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                    
                    val = streamlit_image_coordinates(tmp, key=f"coord_{file_key}")
                    if val:
                        clk = (int(val["x"]), int(val["y"]))
                        if clk != st.session_state.last_click:
                            st.session_state.last_click = clk
                            if len(cur_pts) < 4:
                                if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                                st.session_state.points_map[file_key].append(clk)
                            else:
                                st.session_state.points_map[file_key] = [clk]
                            st.rerun()
                    if len(cur_pts) == 4:
                        if st.button("🚀 Apply Manual Points", use_container_width=True, type="primary"):
                            st.rerun()
            manual_tool()

    else:
        st.info("Upload an image of your artwork to begin.")

except Exception as e:
    st.error("🚨 An error occurred. Please refresh.")
    st.code(traceback.format_exc())
