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
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4: return approx.reshape(4, 2)
    hull = cv2.convexHull(c).reshape(-1, 2)
    s, diff = hull.sum(axis=1), np.diff(hull, axis=1)
    return np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]])

# Session State Initialization
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}

# Debug Sidebar
with st.sidebar:
    st.header("Debug Info")
    st.write(f"Streamlit Version: `{st.__version__}`")
    st.write(f"Canvas: {'✅' if CANVAS_AVAILABLE else '❌'}")
    if CANVAS_ERROR: st.error(f"Canvas Error: {CANVAS_ERROR}")
    st.write(f"Coords: {'✅' if IMAGE_COORDS_AVAILABLE else '❌'}")
    if IMAGE_COORDS_ERROR: st.error(f"Coords Error: {IMAGE_COORDS_ERROR}")
    st.write(f"AI: {'✅' if REMBG_AVAILABLE else '❌'}")
    if REMBG_ERROR: st.warning(f"AI Warning: {REMBG_ERROR}")

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

        # --- AUTO SCAN ---
        if REMBG_AVAILABLE:
            if st.button("🌟 Start AI Auto-Scan", use_container_width=True):
                with st.spinner('AI analyzing layers...'):
                    cleansed = rembg_remove(original_image)
                    st.session_state.processed_images[f"{file_key}_ai"] = cleansed
                    pts = find_corners_from_mask(cleansed)
                    if pts is not None:
                        pts_ordered = order_points(pts)
                        st.session_state.points_map[file_key] = [(float(p[0] / scale_ratio), float(p[1] / scale_ratio)) for p in pts_ordered]
                        st.session_state[f"canvas_key_{file_key}"] = st.session_state.get(f"canvas_key_{file_key}", 0) + 1
                        
                        image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                        warped_cv = four_point_transform(image_cv, pts_ordered)
                        st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                        st.toast("✅ Auto-detected and flattened!")
                    st.rerun()
        else:
            st.info("AI Auto-Scan unavailable.")

        # --- MANUAL INTERFACE ---
        st.divider()
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.write("📍 **Define 4 corners**:")
        with col_h2:
            if st.button("Reset"):
                st.session_state.points_map[file_key] = []
                st.session_state[f"canvas_key_{file_key}"] = st.session_state.get(f"canvas_key_{file_key}", 0) + 1
                if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
                st.rerun()

        current_pts = st.session_state.points_map.get(file_key, [])
        interface_shown = False

        if CANVAS_AVAILABLE:
            try:
                canvas_key = st.session_state.get(f"canvas_key_{file_key}", 0)
                initial_drawing = {"version": "4.4.0", "objects": []}
                for p in current_pts:
                    initial_drawing["objects"].append({
                        "type": "circle", "left": p[0] - 5, "top": p[1] - 5, "radius": 5,
                        "fill": "red", "stroke": "white", "strokeWidth": 2, "selectable": True, "hasControls": False,
                    })

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="white",
                    background_image=ui_image,
                    update_streamlit=True,
                    height=ui_image.height,
                    width=ui_image.width,
                    drawing_mode="point" if len(current_pts) < 4 else "transform",
                    point_display_radius=5,
                    initial_drawing=initial_drawing if current_pts else None,
                    key=f"canvas_{file_key}_{canvas_key}",
                )

                if canvas_result.json_data is not None:
                    new_pts = []
                    for obj in canvas_result.json_data["objects"]:
                        if obj["type"] == "circle":
                            new_pts.append((float(obj["left"] + 5), float(obj["top"] + 5)))
                    
                    if new_pts != current_pts:
                        st.session_state.points_map[file_key] = new_pts
                        if (len(current_pts) < 4 and len(new_pts) >= 4) or (len(current_pts) >= 4 and len(new_pts) < 4):
                            st.rerun()
                interface_shown = True
            except Exception as e:
                st.sidebar.error(f"Canvas runtime error: {e}")
                CANVAS_AVAILABLE = False

        if not interface_shown and IMAGE_COORDS_AVAILABLE:
            try:
                st.info("Using click-to-select interface.")
                temp_ui = ui_image.copy()
                draw = ImageDraw.Draw(temp_ui)
                for i, p in enumerate(current_pts):
                    draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="red", outline="white")
                    draw.text((p[0]+8, p[1]+8), str(i+1), fill="red")
                
                value = streamlit_image_coordinates(temp_ui, key=f"coords_{file_key}")
                if value:
                    new_p = (float(value["x"]), float(value["y"]))
                    if not current_pts or np.linalg.norm(np.array(new_p) - np.array(current_pts[-1])) > 5:
                        if len(current_pts) < 4:
                            if file_key not in st.session_state.points_map:
                                st.session_state.points_map[file_key] = []
                            st.session_state.points_map[file_key].append(new_p)
                            st.rerun()
                interface_shown = True
            except Exception as e:
                st.sidebar.error(f"Coords runtime error: {e}")
                IMAGE_COORDS_AVAILABLE = False

        if not interface_shown:
            st.warning("⚠️ Interactive interface failed to load. Using manual sliders fallback.")
            st.image(ui_image, use_container_width=True)
            
            # Manual coordinate entry fallback
            st.subheader("Manual Corner Entry")
            new_pts = []
            cols = st.columns(2)
            for i in range(4):
                with cols[i % 2]:
                    default_x = int(current_pts[i][0]) if i < len(current_pts) else (0 if i in [0, 3] else ui_image.width)
                    default_y = int(current_pts[i][1]) if i < len(current_pts) else (0 if i in [0, 1] else ui_image.height)
                    
                    x = st.slider(f"P{i+1} X (Horizontal)", 0, ui_image.width, default_x, key=f"sl_x_{file_key}_{i}")
                    y = st.slider(f"P{i+1} Y (Vertical)", 0, ui_image.height, default_y, key=f"sl_y_{file_key}_{i}")
                    new_pts.append((float(x), float(y)))
            
            if st.button("Apply Manual Points", use_container_width=True):
                st.session_state.points_map[file_key] = new_pts
                st.toast("Points updated manually!")
                st.rerun()
            
            # If we have manually set points, we still allow the Finalize button below
            if len(st.session_state.points_map.get(file_key, [])) == 4:
                interface_shown = True

        # Warp Execution
        if len(st.session_state.points_map.get(file_key, [])) == 4:
            if st.button("🚀 Finalize & Flatten Artwork", use_container_width=True, type="primary"):
                with st.spinner("Applying geometry warp..."):
                    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                    pts_scaled = np.array(st.session_state.points_map[file_key], dtype="float32") * scale_ratio 
                    warped_cv = four_point_transform(image_cv, pts_scaled)
                    st.session_state.processed_images[f"{file_key}_warp"] = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
                    st.rerun()

        # --- DISPLAY RESULTS ---
        if f"{file_key}_warp" in st.session_state.processed_images:
            st.subheader("Final Result")
            st.image(st.session_state.processed_images[f"{file_key}_warp"], use_container_width=True)
            buf = io.BytesIO()
            st.session_state.processed_images[f"{file_key}_warp"].save(buf, format="PNG")
            st.download_button("💾 Download Digitized Scan", buf.getvalue(), f"scan_{current_file.name}.png", "image/png")

        if f"{file_key}_ai" in st.session_state.processed_images:
            with st.expander("View AI Background Removal (Cutout)"):
                st.image(st.session_state.processed_images[f"{file_key}_ai"], use_container_width=True)
    else:
        st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 An error occurred!")
    st.code(traceback.format_exc())
