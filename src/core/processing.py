import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from src.utils.dependencies import REMBG_REMOVE, REMBG_AVAILABLE

@st.cache_data(max_entries=10)
def get_cutout(img_bytes: bytes) -> Image.Image:
    """Removes background from image bytes and returns RGBA Image."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if REMBG_AVAILABLE:
        # Returns RGBA image with transparency
        return REMBG_REMOVE(img)
    return img.convert("RGBA")

@st.cache_data(max_entries=10)
def get_flattened_v2(img_bytes: bytes, ui_pts: list, scale: float, masked: bool = True) -> Image.Image:
    """
    Applies four-point perspective transform.
    If masked=True, it applies background removal first to prevent background bleed.
    """
    # 1. Get the image to warp (Original or Cutout)
    if masked:
        base_img = get_cutout(img_bytes) # This is already cached
    else:
        base_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    
    img_np = np.array(base_img)
    # Convert to BGR for OpenCV, preserving Alpha if present
    if img_np.shape[2] == 4:
        image_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
    else:
        image_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    pts_scaled = np.array(ui_pts, dtype="float32") * scale
    
    # Sort points: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_scaled.sum(axis=1)
    d = np.diff(pts_scaled, axis=1)
    
    rect[0] = pts_scaled[np.argmin(s)]
    rect[2] = pts_scaled[np.argmax(s)]
    rect[1] = pts_scaled[np.argmin(d)]
    rect[3] = pts_scaled[np.argmax(d)]
    
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_cv, M, (width, height))
    
    # Return as PIL Image (RGBA)
    if warped.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))
    else:
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def resize_for_ui(img: Image.Image, target_width: int = 450):
    """Resizes image for display while maintaining aspect ratio."""
    w, h = img.size
    scale = w / target_width
    ui_img = img.resize((target_width, int(h / scale)))
    return ui_img, scale
