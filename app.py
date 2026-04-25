import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
from streamlit_image_coordinates import streamlit_image_coordinates

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
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4: return approx.reshape(4, 2)
    hull = cv2.convexHull(c).reshape(-1, 2)
    s, diff = hull.sum(axis=1), np.diff(hull, axis=1)
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]])

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Smart Art Digitizer")

if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}

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
    scale_ratio = original_image.width / 350
    ui_image = original_image.resize((350, int(original_image.height / scale_ratio)))

    # --- PRIMARY ACTION ---
    if st.button("🌟 Start AI Auto-Scan", use_container_width=True):
        with st.spinner('Analyzing layers...'):
            from rembg import remove
            cleansed = remove(original_image)
            st.session_state.processed_images[f"{file_key}_ai"] = cleansed
            pts = find_corners_from_mask(cleansed)
            if pts is not None:
                st.session_state.points_map[file_key] = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in order_points(pts)]
                st.toast("✅ Auto-detected corners!")
            st.rerun()

    # --- STABLE MANUAL INTERFACE ---
    st.divider()
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.write("📍 **Tap 4 corners manually** if auto-scan fails:")
    with col_h2:
        if st.button("Reset"):
            st.session_state.points_map[file_key] = []
            if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
            st.rerun()

    # Draw existing points
    img_display = ui_image.copy()
    current_pts = st.session_state.points_map.get(file_key, [])
    if current_pts:
        draw = ImageDraw.Draw(img_display)
        for p in current_pts:
            x, y = p
            draw.ellipse((x-6, y-6, x+6, y+6), fill='red', outline='white', width=2)

    # Core interaction
    value = streamlit_image_coordinates(img_display, key=f"v4_{file_key}")
    
    if value:
        new_pt = (value["x"], value["y"])
        if new_pt not in current_pts and len(current_pts) < 4:
            st.session_state.points_map[file_key].append(new_pt)
            st.rerun()

    # Processing Manual/Auto
    if len(st.session_state.points_map.get(file_key, [])) == 4:
        if st.button("🚀 Finalize & Flatten Artwork", use_container_width=True, type="primary"):
            with st.spinner("Warping..."):
                image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
                warped_cv = four_point_transform(image_cv, pts_scaled)
                st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                st.rerun()

    # --- RESULTS ---
    if f"{file_key}_warp" in st.session_state.processed_images:
        st.subheader("Final Result")
        st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
        st.download_button("💾 Download Digitized Scan", buf.getvalue(), f"scan_{current_file.name}.png", "image/png")

    if f"{file_key}_ai" in st.session_state.processed_images:
        with st.expander("View AI Background Removal (Cutout)"):
            st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)
            buf_ai = io.BytesIO()
            st.session_state.processed_images[f"{file_key}_ai"].save(buf_ai, format="PNG")
            st.download_button("💾 Download Cutout Only", buf_ai.getvalue(), f"cutout_{current_file.name}.png", "image/png")
else:
    st.info("Upload images to begin.")
