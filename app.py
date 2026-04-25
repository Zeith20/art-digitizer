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
IMAGE_COORDS_ERROR = None
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    IMAGE_COORDS_AVAILABLE = True
except Exception as e:
    IMAGE_COORDS_AVAILABLE = False
    IMAGE_COORDS_ERROR = str(e)

CANVAS_ERROR = None
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception as e:
    CANVAS_AVAILABLE = False
    CANVAS_ERROR = str(e)

REMBG_ERROR = None
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception as e:
    REMBG_AVAILABLE = False
    REMBG_ERROR = str(e)

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

def find_corners_from_mask(pill_image_with_alpha):
    img_np = np.array(pill_image_with_alpha)
    if img_np.shape[2] < 4: return None
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    # Fallback to extreme points of the hull
    hull = cv2.convexHull(c).reshape(-1, 2)
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]])

# Session State Initialization
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

        # Navigation
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with col_nav2:
            st.write(f"**{st.session_state.current_index + 1} / {num_files}**")
        with col_nav3:
            if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
                st.session_state.current_index += 1
                st.rerun()

        original_image = Image.open(current_file).convert("RGB")
        width, height = original_image.size
        display_width = 450
        scale_ratio = width / display_width
        ui_image = original_image.resize((display_width, int(height / scale_ratio)))

        has_ai_scan = f"{file_key}_ai" in st.session_state.processed_images

        # --- STEP 1: AI SCAN (Visible only if not yet scanned) ---
        if not has_ai_scan:
            st.subheader("Original Image")
            st.image(ui_image, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('AI analyzing layers and corners...'):
                        # 1. Background Removal
                        cleansed = rembg_remove(original_image)
                        st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                        
                        # 2. Corner Detection
                        pts = find_corners_from_mask(cleansed)
                        if pts is not None:
                            pts_ordered = order_points(pts)
                            st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts_ordered]
                            
                            # 3. Initial Warp
                            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            warped_cv = four_point_transform(image_cv, pts_ordered)
                            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.toast("✅ AI Scan Complete!")
                        else:
                            st.warning("Could not auto-detect corners perfectly. Please set manually.")
                        st.rerun()
            else:
                st.error("AI functionality (rembg) is not available.")

        # --- STEP 2: RESULTS & CORRECTION (Visible only after scan) ---
        else:
            # 2a. The Result
            if f"{file_key}_warp" in st.session_state.processed_images:
                st.subheader("✨ Final Flattened Result")
                st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    buf = io.BytesIO()
                    st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
                    st.download_button("💾 Download PNG", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
                with col_res2:
                    if st.button("🔄 Redo AI Scan", use_container_width=True):
                        del st.session_state.processed_images[f"{file_key}_ai"]
                        if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
                        st.rerun()

            # 2b. Manual Correction Tool
            st.divider()
            st.subheader("📍 Fine-tune Corners")
            st.caption("If the AI scan wasn't perfect, click the 4 corners below to fix them.")
            
            current_pts = st.session_state.points_map.get(file_key, [])
            
            if IMAGE_COORDS_AVAILABLE:
                temp_ui = ui_image.copy()
                draw = ImageDraw.Draw(temp_ui)
                for i, p in enumerate(current_pts):
                    draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                    draw.text((p[0]+10, p[1]+10), str(i+1), fill="red")
                
                value = streamlit_image_coordinates(temp_ui, key=f"coords_{file_key}")
                if value:
                    new_p = (int(value["x"]), int(value["y"]))
                    if not current_pts or np.linalg.norm(np.array(new_p) - np.array(current_pts[-1])) > 10:
                        if len(current_pts) < 4:
                            if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                            st.session_state.points_map[file_key].append(new_p)
                        else:
                            # If 4 points already exist, reset and start over for correction
                            st.session_state.points_map[file_key] = [new_p]
                        st.rerun()

                if len(current_pts) == 4:
                    if st.button("🚀 Re-Apply Warp with New Corners", use_container_width=True, type="primary"):
                        with st.spinner("Re-processing..."):
                            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            pts_scaled = np.array(current_pts, dtype="float32") * scale_ratio 
                            warped_cv = four_point_transform(image_cv, pts_scaled)
                            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.rerun()
            
            # 2c. Cutout Preview
            with st.expander("View AI Background Removal (Cutout)"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)

    else:
        st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 An error occurred!")
    st.code(traceback.format_exc())
