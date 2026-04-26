import streamlit as st

@st.cache_resource
def load_rembg():
    """Safe loader for rembg library."""
    try:
        from rembg import remove
        return remove, True
    except Exception:
        return None, False

@st.cache_resource
def load_coords_tool():
    """Safe loader for streamlit-image-coordinates library."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        return streamlit_image_coordinates, True
    except Exception:
        return None, False

REMBG_REMOVE, REMBG_AVAILABLE = load_rembg()
IMAGE_COORDS_TOOL, IMAGE_COORDS_AVAILABLE = load_coords_tool()
