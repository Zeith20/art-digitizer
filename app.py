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
    except Exception:
        return None, False

rembg_remove, REMBG_AVAILABLE = load_rembg()

@st.cache_resource
def load_coords():
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        return streamlit_image_coordinates, True
    except Exception:
        return None, False

streamlit_image_coordinates, IMAGE_COORDS_AVAILABLE = load_coords()

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
    """Restored high-quality perspective-aware corner detection."""
    img_np = np.array(pill_image_with_alpha)
    if img_np.shape[2] < 4: return None
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    
    # Try poly approximation first
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
        
    # Fallback to high-quality extreme points of the hull
    hull = cv2.convexHull(c).reshape(-1, 2)
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    # tl, tr, br, bl
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]], dtype="float32")

# Session State Initialization
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}
if 'is_correcting' not in st.session_state: st.session_state.is_correcting = False

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

        # Load images via cache
        original_image, ui_image, scale_ratio = get_ui_image(current_file.getvalue())

        # Navigation
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.session_state.is_correcting = False
                st.rerun()
        with col_nav2:
            st.write(f"**{st.session_state.current_index + 1} / {num_files}**")
        with col_nav3:
            if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
                st.session_state.current_index += 1
                st.session_state.is_correcting = False
                st.rerun()

        has_scanned = f"{file_key}_ai" in st.session_state.processed_images

        # --- STEP 1: AI SCAN ---
        if not has_scanned:
            st.info("Ready for AI Auto-Scan.")
            st.image(ui_image, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    try:
                        with st.spinner('AI analyzing perspective and layers...'):
                            cleansed = rembg_remove(original_image)
                            st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                            pts = find_corners_advanced(cleansed)
                            if pts is not None:
                                # Store as scaled ints to prevent jitter
                                st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts]
                                image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                                warped_cv = four_point_transform(image_cv, pts)
                                st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.rerun()
                    except Exception as ai_err:
                        st.error(f"AI Scan failed: {ai_err}")
            else:
                st.error("AI Library (rembg) is not available.")

        # --- STEP 2: RESULTS & REFINEMENT ---
        else:
            if f"{file_key}_warp" in st.session_state.processed_images:
                st.subheader("✨ Digitized Result")
                st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    buf = io.BytesIO()
                    st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
                    st.download_button("💾 Download Scan", buf.getvalue(), f"digitized_{current_file.name}.png", "image/png", use_container_width=True)
                with col_btn2:
                    if st.button("🔄 Redo Scan", use_container_width=True):
                        if f"{file_key}_ai" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_ai"]
                        if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
                        st.session_state.points_map[file_key] = []
                        st.session_state.is_correcting = False
                        st.rerun()

            # Manual Correction Wrapper (Fragment prevents full page reload)
            @st.fragment
            def manual_correction_fragment():
                st.divider()
                expand_it = st.session_state.is_correcting or len(st.session_state.points_map.get(file_key, [])) < 4
                with st.expander("📍 Fine-tune Artwork Corners", expanded=expand_it):
                    st.write("Click the 4 corners below to fix the AI detection.")
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
                                st.session_state.is_correcting = True
                                if len(current_pts) < 4:
                                    if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                                    st.session_state.points_map[file_key].append(new_p)
                                else:
                                    st.session_state.points_map[file_key] = [new_p]
                                st.rerun()

                        if len(current_pts) == 4:
                            if st.button("🚀 Re-Apply Flattening", use_container_width=True, type="primary"):
                                with st.spinner("Processing..."):
                                    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                                    pts_scaled = np.array(current_pts, dtype="float32") * scale_ratio 
                                    warped_cv = four_point_transform(image_cv, pts_scaled)
                                    st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                                    st.session_state.is_correcting = False
                                    st.rerun()
            
            # Execute the fragment
            manual_correction_fragment()
            
            with st.expander("🖼️ View AI Mask (Cutout)"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)

    else:
        st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 An unexpected error occurred!")
    st.code(traceback.format_exc())
