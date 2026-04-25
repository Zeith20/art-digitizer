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

def find_corners_from_mask(pill_image_with_alpha):
    """Detects 4 corners from the alpha channel of a rembg result."""
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

def render_manual_ui(file_key, ui_image, original_image, scale_ratio):
    """Renders the interactive image and processing button for manual selection."""
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        st.write("📍 **Tap 4 corners** to manually adjust:")
    with col_m2:
        if st.button("Reset", key=f"reset_{file_key}"):
            st.session_state.points_map[file_key] = []
            st.rerun()

    img_to_draw = ui_image.copy()
    if file_key in st.session_state.points_map:
        draw = ImageDraw.Draw(img_to_draw)
        for p in st.session_state.points_map[file_key]:
            x, y = p
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red')

    # Use a specific key for the coord component to avoid state loss
    value = streamlit_image_coordinates(img_to_draw, key=f"coords_{file_key}")
    if value is not None:
        point = (value["x"], value["y"])
        if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
        if point not in st.session_state.points_map[file_key] and len(st.session_state.points_map[file_key]) < 4:
            st.session_state.points_map[file_key].append(point)
            st.rerun()

    if file_key in st.session_state.points_map and len(st.session_state.points_map[file_key]) == 4:
        if st.button("🚀 Apply Manual Warp", use_container_width=True, key=f"apply_{file_key}"):
            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
            warped_cv = four_point_transform(image_cv, pts_scaled)
            st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
            st.rerun()

# --- PAGE SETUP ---
st.set_page_config(page_title="Art Digitizer", layout="centered")
st.title("🎨 Smart Art Digitizer")

# Initialize session state
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}

# --- UI: UPLOAD ---
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

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Prev") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with col_nav2:
        st.write(f"**Image {st.session_state.current_index + 1} of {num_files}**")
    with col_nav3:
        if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
            st.session_state.current_index += 1
            st.rerun()

    original_image = Image.open(current_file).convert("RGB")
    scale_ratio = original_image.width / 350
    ui_image = original_image.resize((350, int(original_image.height / scale_ratio)))

    # --- ACTION BUTTON ---
    if st.button("🌟 Start Smart Digitization (AI + Warp)", use_container_width=True):
        with st.spinner('AI analyzing layers...'):
            from rembg import remove
            cleansed_image = remove(original_image)
            st.session_state.processed_images[f"{file_key}_ai"] = cleansed_image
            pts = find_corners_from_mask(cleansed_image)
            if pts is not None:
                st.session_state.points_map[file_key] = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in order_points(pts)]
                image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
                warped_cv = four_point_transform(image_cv, pts_scaled)
                st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                st.toast("✅ Smart scan complete!")
            else:
                st.warning("Could not auto-detect corners. Adjust manually below.")

    # --- UI: DISPLAY RESULTS ---
    if f"{file_key}_warp" in st.session_state.processed_images:
        st.subheader("Result")
        st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
        st.download_button("💾 Download Result", buf.getvalue(), f"digitized_{current_file.name}.png", "image/png")
        
        with st.expander("🛠️ Manually adjust corners"):
            render_manual_ui(file_key, ui_image, original_image, scale_ratio)
            
        if st.button("🗑️ Clear and Restart"):
            if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
            if f"{file_key}_ai" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_ai"]
            st.session_state.points_map[file_key] = []
            st.rerun()

    elif f"{file_key}_ai" in st.session_state.processed_images:
        st.subheader("AI Cutout")
        st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state.processed_images[f"{file_key}_ai"].save(buf, format="PNG")
        st.download_button("💾 Download Cutout", buf.getvalue(), f"cutout_{current_file.name}.png", "image/png")
        
        with st.expander("🛠️ Try perspective warp manually"):
            render_manual_ui(file_key, ui_image, original_image, scale_ratio)

    else:
        # If no result exists yet, show the manual UI by default
        render_manual_ui(file_key, ui_image, original_image, scale_ratio)

else:
    st.info("Please upload images to get started.")
