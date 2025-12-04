#Kody Graham
#12/02/2025
#This class will handle the front end

import os, sys, subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Make sure the app/ folder itself is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference import TamperDetector

@st.cache_resource
def load_detector() -> TamperDetector:
    return TamperDetector()


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def main():
    st.set_page_config(page_title= "PixelSentinel - Image Forensics AI", layout = "wide",)
    st.title("Pixel Sentinel - Image Forensics AI")
    st.write("Upload an image and let the model analyze whether it appears unaltered or tampered, "
             "estimate type of manipulation, and highlight suspicious regions using a Grad-CAM heatmap.")

    detector = load_detector()

    if not detector.is_trained:
        st.warning("No trained model found. Please train a model first.")

    st.sidebar.header("Upload")
    uploaded_file = st.sidebar.file_uploader("Choose an image (JPG / JPEG / PNG)", type=["jpg", "jpeg", "png"])

    st.sidebar.markdown("---")
    st.sidebar.write("Current model")
    st.sidebar.write(
        "- Tamper head: 'original' vs 'tampered'\n"
        "- Type head: 'original', 'jpeg', 'blur', 'noise',\n"
        "  'copy_move', 'splice', 'inpaint'")

    if uploaded_file is None:
        st.info("Upload an image using the sidebar to begin")
        return
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    st.image(img_rgb, caption= "Original image", use_container_width=True)

    with st.spinner("Analyzing image with AI model..."):
        result = detector.analyze_image(img_bgr)

    tamper_label = result["tamper_label"]
    tamper_confidence = result["tamper_confidence"]
    type_label = result["type_label"]
    type_confidence = result["type_confidence"]
    overlay_bgr = result["overlay_bgr"]

    st.subheader("Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Integrity Classification")
        if tamper_label == "tampered":
            st.error(
                f"Integrity: {tamper_label.upper()}  "
                f"({tamper_confidence * 100:.1f}% confidence)"
            )
        else:
            st.success(
                f"Integrity: {tamper_label.upper()}  "
                f"({tamper_confidence * 100:.1f}% confidence)"
            )

        st.markdown("### Manipulation Type")
        st.write(
            f"Predicted type: '{type_label}'  "
            f"({type_confidence * 100:.1f}% confidence)"
        )

        # Explanation text
        if tamper_label == "tampered":
            st.markdown(
                "- The model believes this image has been ALTERED.\n"
                "- The Grad-CAM heatmap highlights WHERE in the image it "
                "found the strongest tampering cues.\n"
                "- Types like 'splice' and 'inpaint' roughly correspond to "
                "Photoshop-style object insertion/removal."
            )
        else:
            st.markdown(
                "- The model believes this image is ORIGINAL (within the "
                "tampering patterns it was trained on).\n"
                "- No heatmap is displayed because it did not confidently "
                "classify it as tampered."
            )

    with col2:
        st.markdown("### Tamper Heatmap Overlay")

        if overlay_bgr is not None:
            overlay_rgb = bgr_to_rgb(overlay_bgr)
            st.image(
                overlay_rgb,
                caption="Grad-CAM heatmap overlay (suspicious regions are highlighted)",
                use_container_width=True,
            )
        else:
            st.info(
                "No heatmap available - the model did not classify this image as tampered."
            )

    st.markdown("---")
    st.caption(
        "This demo is an image forensics project powered by "
        "TensorFlow, OpenCV, and Grad-CAM, mostly to show I can work with TensorFlow. "
        "It detects synthetic tampering patterns and visual artifacts, "
        "not ALL possible photo edits.")


def _launch_streamlit():

    if os.environ.get("PIXELSENTINELAI_STREAMLIT_CHILD") == "1":
        main()
        return

    os.environ["PIXELSENTINELAI_STREAMLIT_CHILD"] = "1"

    app_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True, cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    _launch_streamlit()