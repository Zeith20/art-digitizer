import streamlit as st
from PIL import Image, ImageDraw
from dependencies import IMAGE_COORDS_TOOL, IMAGE_COORDS_AVAILABLE
from from state import get_current_pts, set_points

@st.fragment
def manual_correction_component(file_key, ui_image):
    """
    Isolated manual correction tool.
    Uses st.fragment to prevent full page reloads on every click.
    """
    st.divider()
    with st.expander("📍 Fine-tune Corners"):
        st.caption("Click 4 corners to fix detection. This tool updates instantly.")
        pts = get_current_pts(file_key)
        
        if IMAGE_COORDS_AVAILABLE:
            # Draw visual feedback
            temp_ui = ui_image.copy()
            draw = ImageDraw.Draw(temp_ui)
            for i, p in enumerate(pts):
                draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="red", outline="white", width=2)
                draw.text((p[0]+10, p[1]+10), str(i+1), fill="red")
            
            # Capture click
            value = IMAGE_COORDS_TOOL(temp_ui, key=f"coords_{file_key}")
            
            if value:
                click = (int(value["x"]), int(value["y"]))
                
                # Check for new click (state sync)
                if click != st.session_state.last_click:
                    st.session_state.last_click = click
                    if len(pts) < 4:
                        pts.append(click)
                    else:
                        pts = [click] # Restart correction
                    set_points(file_key, pts)
                    st.rerun() #targeted fragment rerun

        if len(get_current_pts(file_key)) == 4:
            if st.button("🚀 Apply Manual Points", use_container_width=True, type="primary"):
                st.rerun() # Full app rerun to update result display

def sidebar_navigation(num_files):
    """Sidebar navigation controls for batch processing."""
    with st.sidebar:
        st.header("Batch Navigation")
        st.info(f"📁 {num_files} images loaded.")
        
        # Safe selection
        st.session_state.current_index = st.number_input(
            "Current Image Index", 1, num_files, st.session_state.current_index + 1
        ) - 1
        
        st.divider()
        st.caption("Admin")
        from from state import reset_all_progress
        if st.button("🗑️ Reset All Progress", use_container_width=True):
            reset_all_progress()
