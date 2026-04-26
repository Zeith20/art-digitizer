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

def analyze_shape_and_get_corners(pill_image_with_alpha):
    """Analyzes mask to decide if it's a rectangle or an irregular shape."""
    img_np = np.array(pill_image_with_alpha)
    if img_np.shape[2] < 4: return None, "none"
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, "none"
    c = max(cnts, key=cv2.contourArea)
    
    # Analyze rectangularity
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    area = cv2.contourArea(c)
    rect_points = cv2.minAreaRect(c)
    rect_area = rect_points[1][0] * rect_points[1][1]
    
    # Solidity/Rectangularity check (Is it mostly a box?)
    is_rectangular = (len(approx) == 4) or (rect_area > 0 and (area / rect_area) > 0.85)
    
    if is_rectangular:
        # TL, TR, BR, BL fallback for rectangles
        hull = cv2.convexHull(c).reshape(-1, 2)
        s = hull.sum(axis=1)
        diff = np.diff(hull, axis=1)
        pts = np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]], dtype="float32")
        return pts, "rectangle"
    
    return None, "irregular"

# Session State
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}
if 'last_click' not in st.session_state: st.session_state.last_click = None

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

        # Fast Load
        if f"img_{file_key}" not in st.session_state:
            img = Image.open(current_file).convert("RGB")
            st.session_state[f"img_{file_key}"] = img
            w, h = img.size
            scale = w / 450
            st.session_state[f"ui_{file_key}"] = img.resize((450, int(h / scale)))
            st.session_state[f"scale_{file_key}"] = scale

        original_image = st.session_state[f"img_{file_key}"]
        ui_image = st.session_state[f"ui_{file_key}"]
        scale_ratio = st.session_state[f"scale_{file_key}"]

        # Navigation
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

        has_scanned = f"{file_key}_ai" in st.session_state.processed_images

        # --- STEP 1: AI SCAN ---
        if not has_scanned:
            st.info("Ready for AI Auto-Scan.")
            st.image(ui_image, use_container_width=True)
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing shape...'):
                        cleansed = rembg_remove(original_image)
                        st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                        
                        pts, shape_type = analyze_shape_and_get_corners(cleansed)
                        
                        if shape_type == "rectangle" and pts is not None:
                            st.session_state.points_map[file_key] = [(int(p[0] / scale_ratio), int(p[1] / scale_ratio)) for p in pts]
                            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            warped_cv = four_point_transform(image_cv, pts)
                            st.session_state.processed_images[f"{file_key}_result"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.toast("✅ Detected Rectangular Art - Applied Perspective Correction")
                        else:
                            # Irregular shape - just show the cutout
                            st.session_state.processed_images[f"{file_key}_result"] = cleansed
                            st.toast("✅ Detected Irregular Shape - Background Removed")
                        st.rerun()
            else: st.error("AI Library unavailable.")

        # --- STEP 2: RESULTS ---
        else:
            if f"{file_key}_result" in st.session_state.processed_images:
                st.subheader("✨ Digitized Result")
                st.image(st.session_state.processed_images[f"{file_key}_result"], use_container_width=True)
                
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    buf = io.BytesIO()
                    st.session_state.processed_images[f"{file_key}_result"].save(buf, format="PNG")
                    st.download_button("💾 Download PNG", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
                with c_res2:
                    if st.button("🔄 Redo AI", use_container_width=True):
                        if f"{file_key}_ai" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_ai"]
                        if f"{file_key}_result" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_result"]
                        st.session_state.points_map[file_key] = []
                        st.rerun()

            # --- STEP 3: MANUAL TOOL ---
            @st.fragment
            def manual_tool():
                st.divider()
                with st.expander("📍 Fine-tune Corners / Forced Warp"):
                    st.caption("Click 4 corners if you want to force a perspective flattening on an irregular shape.")
                    pts = st.session_state.points_map.get(file_key, [])
                    
                    if IMAGE_COORDS_AVAILABLE:
                        temp_ui = ui_image.copy()
                        draw = ImageDraw.Draw(temp_ui)
                        for i, p in enumerate(pts):
                            draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                            draw.text((p[0]+10, p[1]+10), str(i+1), fill="red")
                        
                        value = streamlit_image_coordinates(temp_ui, key=f"coords_{file_key}")
                        if value:
                            click = (int(value["x"]), int(value["y"]))
                            if click != st.session_state.last_click:
                                st.session_state.last_click = click
                                if len(pts) < 4:
                                    st.session_state.points_map[file_key].append(click)
                                else:
                                    st.session_state.points_map[file_key] = [click]
                                st.rerun()

                    if len(st.session_state.points_map.get(file_key, [])) == 4:
                        if st.button("🚀 Apply Manual Perspective", use_container_width=True, type="primary"):
                            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
                            warped_cv = four_point_transform(image_cv, pts_scaled)
                            st.session_state.processed_images[f"{file_key}_result"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                            st.rerun()

            manual_tool()

            with st.expander("🖼️ View Original Background-Removed Mask"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)

    else: st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 An error occurred!")
    st.code(traceback.format_exc())
