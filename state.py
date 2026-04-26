import streamlit as st

def initialize_session_state():
    """Ensures all required session state variables exist."""
    defaults = {
        'points_map': {},
        'current_index': 0,
        'scanned_files': set(),
        'last_click': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def reset_all_progress():
    """Wipes session memory for a fresh start."""
    st.session_state.scanned_files = set()
    st.session_state.points_map = {}
    st.session_state.current_index = 0
    st.rerun()

def get_current_pts(file_key):
    """Safely retrieves points for a specific file."""
    return st.session_state.points_map.get(file_key, [])

def set_points(file_key, points):
    """Saves points for a specific file."""
    st.session_state.points_map[file_key] = points

def mark_as_scanned(file_key):
    """Tracks that a file has been processed by AI."""
    st.session_state.scanned_files.add(file_key)

def is_scanned(file_key):
    """Checks if a file has been processed by AI."""
    return file_key in st.session_state.scanned_files
