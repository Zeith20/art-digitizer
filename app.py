import streamlit as st
import io
from PIL import Image
import traceback

# Modular Imports
from src.utils.dependencies import REMBG_AVAILABLE
from src.utils.state import (
    initialize_session_state, mark_as_scanned, is_scanned, 
    get_current_pts, set_points
)
from src.core.processing import get_cutout, get_flattened, resize_for_ui
from src.core.analysis import analyze_shape_and_get_pts
from src.ui.components import manual_correction_component, sidebar_navigation

# --- APP BOOTSTRAP ---
initialize_session_state()
st.set_page_config(page_title="Art Digitizer Pro", layout="centered")
st.title("🎨 Smart Art Digitizer")

try:
    # 1. FILE UPLOAD (Streaming Batch Support)
    uploaded_files = st.file_uploader(
        "Upload artworks from your folder", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        
        # Sidebar & Top Navigation
        sidebar_navigation(num_files)
        
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1; st.rerun()
        with col_nav2: st.write(f"**{st.session_state.current_index + 1} / {num_files}**")
        with col_nav3:
            if st.button("Next ➡️") and st.session_state.current_index < num_files - 1:
                st.session_state.current_index += 1; st.rerun()

        # Load Active File
        current_file = uploaded_files[st.session_state.current_index]
        file_key = f"{current_file.name}_{current_file.size}"
        file_bytes = current_file.getvalue()
        
        # UI Pre-processing
        img_raw = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        ui_img, scale_ratio = resize_for_ui(img_raw)

        # --- STEP 1: AI SCAN PIPELINE ---
        if not is_scanned(file_key):
            st.subheader(f"📄 {current_file.name}")
            st.image(ui_img, use_container_width=True)
            
            if REMBG_AVAILABLE:
                if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                    with st.spinner('Analyzing perspective...'):
                        # Background Removal
                        cutout = get_cutout(file_bytes)
                        buf = io.BytesIO(); cutout.save(buf, format="PNG")
                        
                        # Shape Analysis
                        pts, shape = analyze_shape_and_get_pts(buf.getvalue(), scale_ratio)
                        if shape == "rectangle" and pts:
                            set_points(file_key, pts)
                        
                        mark_as_scanned(file_key)
                        st.rerun()
            else:
                st.error("AI Library (rembg) unavailable.")

        # --- STEP 2: RESULTS & REFINEMENT ---
        else:
            pts = get_current_pts(file_key)
            
            if len(pts) == 4:
                res_img = get_flattened(file_bytes, pts, scale_ratio)
                st.subheader("✨ Perspective Scan")
            else:
                res_img = get_cutout(file_bytes)
                st.subheader("✨ Background-Removed Cutout")
            
            st.image(res_img, use_container_width=True)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                buf = io.BytesIO(); res_img.save(buf, format="PNG")
                st.download_button("💾 Download", buf.getvalue(), f"digitized_{current_file.name}.png", "image/png", use_container_width=True)
            with c_res2:
                if st.button("🔄 Reset This Image", use_container_width=True):
                    st.session_state.scanned_files.discard(file_key)
                    st.session_state.points_map.pop(file_key, None)
                    st.rerun()

            # Manual correction component (fragmented)
            manual_correction_component(file_key, ui_img)
            
            with st.expander("🖼️ View Original AI Mask"):
                st.image(get_cutout(file_bytes), use_container_width=True)

    else:
        st.info("Select images from your folder to begin processing.")

except Exception as e:
    st.error("🚨 An application error occurred.")
    st.code(traceback.format_exc())
