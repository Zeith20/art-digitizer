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

# --- CACHED IMAGE PROCESSING ---
@st.cache_data
def get_ui_image(file_bytes, display_width=450):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    scale = w / display_width
    ui_img = img.resize((display_width, int(h / scale)))
    return img, ui_img, scale

# --- UTILS ---
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
    """Hull-based extreme corner detection (better for skewed photos)."""
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

        current_file = uploaded_files[st.session_state.current_index]
        file_key = f"{current_file.name}_{current_file.size}"
        original_image, ui_image, scale_ratio = get_ui_image(current_file.getvalue())

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
                        cleansed = rembg_remove(original_image)
                        st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                        pts = find_corners_advanced(cleansed)
                        if pts is not None:
                            st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts]
                            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            warped_cv = four_point_transform(image_cv, pts)
                            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                        st.rerun()
            else: st.error("AI Library (rembg) unavailable.")

        # --- STEP 2: RESULTS & STABLE CORRECTION ---
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

            # --- STABLE MANUAL REFINEMENT (NO BUGGY LIBRARIES) ---
            st.divider()
            with st.expander("📍 Fine-tune Corners (No Refresh Mode)", expanded=False):
                st.caption("Adjust sliders to move the corners. Watch the red dots below.")
                
                pts = st.session_state.points_map.get(file_key, [(0,0), (ui_image.width,0), (ui_image.width, ui_image.height), (0, ui_image.height)])
                if len(pts) < 4: pts = [(0,0), (ui_image.width,0), (ui_image.width, ui_image.height), (0, ui_image.height)]
                
                new_pts = []
                # Fast Slider Grid
                cols = st.columns(2)
                labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                for i in range(4):
                    with cols[i % 2]:
                        st.write(f"**{labels[i]}**")
                        px = st.slider(f"X {i}", 0, ui_image.width, int(pts[i][0]), key=f"x_{file_key}_{i}")
                        py = st.slider(f"Y {i}", 0, ui_image.height, int(pts[i][1]), key=f"y_{file_key}_{i}")
                        new_pts.append((px, py))
                
                # Visual Feedback: Image with dots drawn on it (Instantly updated by sliders)
                feedback_img = ui_image.copy()
                draw = ImageDraw.Draw(feedback_img)
                for i, p in enumerate(new_pts):
                    draw.ellipse([p[0]-8, p[1]-8, p[0]+8, p[1]+8], fill="red", outline="white", width=2)
                    draw.text((p[0]+12, p[1]+12), str(i+1), fill="red")
                st.image(feedback_img, caption="Real-time adjustment preview", use_container_width=True)

                if st.button("🚀 Apply These Corners", use_container_width=True, type="primary"):
                    st.session_state.points_map[file_key] = new_pts
                    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                    pts_scaled = np.array(new_pts, dtype="float32") * scale_ratio 
                    warped_cv = four_point_transform(image_cv, pts_scaled)
                    st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                    st.rerun()

            with st.expander("🖼️ View AI Mask (Cutout)"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)

    else: st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 Error!")
    st.code(traceback.format_exc())
