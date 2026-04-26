import streamlit as st
import io
import sys
import os
from PIL import Image
import traceback

# --- BOOTSTRAP ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modular Imports
from dependencies import REMBG_AVAILABLE
from state import (
    initialize_session_state, mark_as_scanned, is_scanned, 
    get_current_pts, set_points
)
from processing import get_cutout, get_flattened_v2, resize_for_ui
from analysis import analyze_shape_and_get_pts
from components import manual_correction_component, sidebar_navigation

# Initialize state and config
initialize_session_state()
st.set_page_config(page_title="Art Digitizer Pro", layout="centered")
st.title("🎨 Smart Art Digitizer")

# Global Reset
with st.sidebar:
    st.header("Batch Management")
    if st.button("🚨 Reset Process", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

try:
    # 1. DURABLE STREAMING UPLOADER
    # We use a unique key to prevent uploader flickering
    uploaded_files = st.file_uploader(
        "Select your folder images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="folder_uploader_v2"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        # Prevent index out of bounds
        if st.session_state.current_index >= num_files:
            st.session_state.current_index = 0
            st.rerun()
            
        # 2. NAVIGATION (Unified Single Line)
        sidebar_navigation(num_files)
        
        col_nav = st.columns([1, 1, 1])
        with col_nav[0]:
            if st.button("⬅️ Previous", use_container_width=True) and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with col_nav[1]:
            st.markdown(f"<div style='text-align: center; padding-top: 10px; font-weight: bold;'>{st.session_state.current_index + 1} / {num_files}</div>", unsafe_allow_html=True)
        with col_nav[2]:
            if st.button("Next ➡️", use_container_width=True) and st.session_state.current_index < num_files - 1:
                st.session_state.current_index += 1
                st.rerun()

        # 3. ON-DEMAND PROCESSING (Saves Memory)
        current_file = uploaded_files[st.session_state.current_index]
        file_key = f"{current_file.name}_{current_file.size}"
        
        try:
            # Process pixels ONLY when needed
            file_bytes = current_file.getvalue()
            img_raw = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            ui_img, scale_ratio = resize_for_ui(img_raw)

            # --- STEP 1: AI SCAN ---
            if not is_scanned(file_key):
                st.subheader(f"📄 {current_file.name}")
                st.image(ui_img, use_container_width=True)
                if REMBG_AVAILABLE:
                    if st.button("🚀 Start AI Auto-Digitize", use_container_width=True, type="primary"):
                        with st.spinner('Analyzing shape...'):
                            cutout = get_cutout(file_bytes)
                            buf = io.BytesIO(); cutout.save(buf, format="PNG")
                            pts, shape = analyze_shape_and_get_pts(buf.getvalue(), scale_ratio)
                            if shape == "rectangle" and pts: set_points(file_key, pts)
                            mark_as_scanned(file_key)
                            st.rerun()
                else: st.error("AI Unavailable.")

            # --- STEP 2: RESULTS ---
            else:
                pts = get_current_pts(file_key)
                if len(pts) == 4:
                    res_img = get_flattened_v2(file_bytes, pts, scale_ratio, masked=True)
                    st.subheader("✨ Final Scan")
                else:
                    res_img = get_cutout(file_bytes)
                    st.subheader("✨ Clean Cutout")
                
                st.image(res_img, use_container_width=True)
                
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    buf = io.BytesIO(); res_img.save(buf, format="PNG")
                    st.download_button("💾 Download PNG", buf.getvalue(), f"scan_{current_file.name}.png", "image/png", use_container_width=True)
                with c_res2:
                    if st.button("🔄 Reset This Image", use_container_width=True):
                        st.session_state.scanned_files.discard(file_key)
                        st.session_state.points_map.pop(file_key, None)
                        st.rerun()

                manual_correction_component(file_key, ui_img)
                
        except Exception as file_err:
            st.error(f"⚠️ Could not read {current_file.name}. The file might be too large or corrupted.")
            if st.button("Skip to Next"):
                st.session_state.current_index = (st.session_state.current_index + 1) % num_files
                st.rerun()

    else:
        st.info("Select one or more images from your folder to begin.")

except Exception as e:
    st.error("🚨 An unexpected error occurred.")
    st.code(traceback.format_exc())
