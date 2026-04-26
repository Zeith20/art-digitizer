import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import base64
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
def load_canvas():
    try:
        from streamlit_drawable_canvas import st_canvas
        return st_canvas, True
    except Exception: return None, False

st_canvas, CANVAS_AVAILABLE = load_canvas()

# --- UTILS ---
def get_image_base64(img):
    """Converts PIL image to base64 for guaranteed client-side rendering."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(w1), int(w2))
    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(h1), int(h2))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

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

# Session State
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}

try:
    uploaded_files = st.file_uploader("Upload drawings", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        file_list_sig = "-".join([f"{f.name}_{f.size}" for f in uploaded_files])
        if 'last_upload_sig' not in st.session_state or st.session_state.last_upload_sig != file_list_sig:
            st.session_state.last_upload_sig = file_list_sig
            st.session_state.current_index = 0
            st.rerun()

        num_files = len(uploaded_files)
        current_file = uploaded_files[st.session_state.current_index]
        file_key = f"{current_file.name}_{current_file.size}"

        img_raw = Image.open(current_file).convert("RGB")
        width, height = img_raw.size
        display_width = 450
        scale_ratio = width / display_width
        ui_image = img_raw.resize((display_width, int(height / scale_ratio)))

        # Pre-convert to Base64 to ensure canvas never shows black
        ui_base64 = get_image_base64(ui_image)

        # Navigation
        c_nav1, c_nav2, c_nav3 = st.columns([1, 1, 1])
        with c_nav1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with c_nav2: st.write(f"**{st.session_state.current_index + 1} / {len(uploaded_files)}**")
        with c_nav3:
            if st.button("Next ➡️") and st.session_state.current_index < len(uploaded_files) - 1:
                st.session_state.current_index += 1
                st.rerun()

        has_scanned = f"{file_key}_ai" in st.session_state.processed_images

        # --- STEP 1: AI SCAN ---
        if not has_scanned:
            st.info("Ready for AI Auto-Scan.")
            st.image(ui_image, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing...'):
                        cleansed = rembg_remove(img_raw)
                        st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                        pts = find_corners_advanced(cleansed)
                        if pts is not None:
                            st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts]
                            image_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
                            warped_cv = four_point_transform(image_cv, pts)
                            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                        st.rerun()
            else: st.error("AI Library unavailable.")

        # --- STEP 2: RESULTS & CORRECTION ---
        else:
            if f"{file_key}_warp" in st.session_state.processed_images:
                st.subheader("✨ Digitized Result")
                st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    buf = io.BytesIO()
                    st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
                    st.download_button("💾 Download PNG", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
                with c2:
                    if st.button("🔄 Redo AI Scan", use_container_width=True):
                        del st.session_state.processed_images[f"{file_key}_ai"]
                        if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
                        st.session_state.points_map[file_key] = []
                        st.rerun()

            # --- HARD-FIXED MANUAL TOOL ---
            st.divider()
            st.subheader("📍 Manual Corner Correction")
            st.caption("Drag dots to refine. No page reloads! Redo points by clicking 'Reset' above.")
            
            if CANVAS_AVAILABLE:
                existing_pts = st.session_state.points_map.get(file_key, [])
                initial_drawing = {"version": "4.4.0", "objects": []}
                for i, p in enumerate(existing_pts):
                    initial_drawing["objects"].append({
                        "type": "circle", "left": p[0]-8, "top": p[1]-8, "radius": 8, 
                        "fill": "rgba(255, 0, 0, 0.5)", "stroke": "white", "strokeWidth": 2,
                        "selectable": True, "hasControls": False, "hasBorders": False # NO RESIZING
                    })

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    background_image=ui_image, 
                    background_url=ui_base64, # FORCED VISIBILITY
                    update_streamlit=False, 
                    height=ui_image.height,
                    width=ui_image.width,
                    drawing_mode="point" if len(existing_pts) < 4 else "transform",
                    point_display_radius=8,
                    initial_drawing=initial_drawing if existing_pts else None,
                    key=f"canvas_final_{file_key}",
                )

                if st.button("🚀 Apply Manual Correction", use_container_width=True, type="primary"):
                    if canvas_result.json_data and "objects" in canvas_result.json_data:
                        new_pts = []
                        for obj in canvas_result.json_data["objects"]:
                            if obj["type"] == "circle":
                                new_pts.append((int(obj["left"] + 8), int(obj["top"] + 8)))
                        
                        if len(new_pts) == 4:
                            st.session_state.points_map[file_key] = new_pts
                            image_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
                            pts_scaled = np.array(new_pts, dtype="float32") * scale_ratio 
                            warped_cv = four_point_transform(image_cv, pts_scaled)
                            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.rerun()
                        else: st.error(f"Please select exactly 4 points. You have {len(new_pts)}.")

            with st.expander("🖼️ View AI Mask (Cutout)"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)

    else: st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 Error!")
    st.code(traceback.format_exc())
