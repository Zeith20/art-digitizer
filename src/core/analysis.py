import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

@st.cache_data(max_entries=10)
def analyze_shape_and_get_pts(cutout_bytes: bytes, ui_scale: float):
    """
    Analyzes the mask to detect if the object is rectangular and returns corners.
    
    Returns:
        tuple: (ui_points, shape_type)
    """
    img_np = np.array(Image.open(io.BytesIO(cutout_bytes)))
    if img_np.shape[2] < 4:
        return None, "none"
        
    alpha = img_np[:, :, 3]
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return None, "none"
        
    c = max(cnts, key=cv2.contourArea)
    
    # Analysis
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    rect_area = rect[1][0] * rect[1][1]
    
    # 82% density threshold for rectangularity
    is_rectangular = (rect_area > 0 and (area / rect_area) > 0.82)
    
    if is_rectangular:
        hull = cv2.convexHull(c).reshape(-1, 2)
        s = hull.sum(axis=1)
        d = np.diff(hull, axis=1)
        
        # Detected Corners
        tl = hull[np.argmin(s)]
        br = hull[np.argmax(s)]
        tr = hull[np.argmin(d)]
        bl = hull[np.argmax(d)]
        
        raw_pts = [tl, tr, br, bl]
        ui_pts = [(int(p[0] / ui_scale), int(p[1] / ui_scale)) for p in raw_pts]
        return ui_pts, "rectangle"
        
    return None, "irregular"
