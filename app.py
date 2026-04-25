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

# Session State Initialization
if 'points_map' not in st.session_state: st.session_state.points_map = {}
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}

# Debug Sidebar
with st.sidebar:
    st.header("Settings & Debug")
    advanced_mode = st.checkbox("Enable Advanced Drag (Canvas)", value=False)
    st.divider()
    st.write(f"Streamlit: `{st.__version__}`")
    st.write(f"Click Tool: {'✅' if IMAGE_COORDS_AVAILABLE else '❌'}")
    st.write(f"Drag Tool: {'✅' if CANVAS_AVAILABLE else '❌'}")
    if CANVAS_ERROR: st.caption(f"Canvas Error: {CANVAS_ERROR}")

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
                    # Simplified corner logic for AI scan
                    st.session_state.points_map[file_key] = [(0,0), (display_width,0), (display_width, ui_image.height), (0, ui_image.height)]
                    st.toast("AI scan done! Adjust points manually below.")
                    st.rerun()

        # --- MANUAL INTERFACE ---
        st.divider()
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.subheader("📍 Define 4 Corners")
        with col_h2:
            if st.button("Reset"):
                st.session_state.points_map[file_key] = []
                if f"{file_key}_warp" in st.session_state.processed_images: del st.session_state.processed_images[f"{file_key}_warp"]
                st.rerun()

        current_pts = st.session_state.points_map.get(file_key, [])
        
        # PRIMARY INTERFACE: Click-to-select (Most Stable)
        if IMAGE_COORDS_AVAILABLE and not advanced_mode:
            st.info(f"Selected: {len(current_pts)} / 4 corners. Click on the image to place points.")
            temp_ui = ui_image.copy()
            draw = ImageDraw.Draw(temp_ui)
            for i, p in enumerate(current_pts):
                draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                draw.text((p[0]+10, p[1]+10), str(i+1), fill="red")
            
            value = streamlit_image_coordinates(temp_ui, key=f"coords_{file_key}")
            if value:
                new_p = (int(value["x"]), int(value["y"]))
                # Anti-double-click check
                if not current_pts or np.linalg.norm(np.array(new_p) - np.array(current_pts[-1])) > 10:
                    if len(current_pts) < 4:
                        if file_key not in st.session_state.points_map: st.session_state.points_map[file_key] = []
                        st.session_state.points_map[file_key].append(new_p)
                        st.rerun()

        # SECONDARY INTERFACE: Canvas (For dragging)
        elif CANVAS_AVAILABLE and advanced_mode:
            st.warning("Advanced Mode: Drag circles to adjust. If image is missing, switch back in sidebar.")
            initial_drawing = {"version": "4.4.0", "objects": []}
            for p in current_pts:
                initial_drawing["objects"].append({
                    "type": "circle", "left": p[0] - 5, "top": p[1] - 5, "radius": 5,
                    "fill": "red", "stroke": "white", "strokeWidth": 2, "selectable": True, "hasControls": False,
                })

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                background_image=ui_image,
                update_streamlit=True,
                height=ui_image.height,
                width=ui_image.width,
                drawing_mode="point" if len(current_pts) < 4 else "transform",
                point_display_radius=8,
                initial_drawing=initial_drawing if current_pts else None,
                key=f"canvas_{file_key}",
            )

            if canvas_result.json_data is not None:
                new_pts = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "circle":
                        new_pts.append((int(obj["left"] + 5), int(obj["top"] + 5)))
                
                if len(new_pts) > 0 and new_pts != current_pts:
                    st.session_state.points_map[file_key] = new_pts
                    if len(new_pts) != len(current_pts): st.rerun()

        else:
            st.error("No interactive interface available. Please check sidebar.")

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

    else:
        st.info("Upload images to begin.")

except Exception as e:
    st.error("🚨 An error occurred!")
    st.code(traceback.format_exc())
