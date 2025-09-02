"""
Camera Feed page for TrashNet Streamlit app.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import *
from src.inference.predictor import TrashClassifier

st.set_page_config(
    page_title="Camera Feed - TrashNet",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Real-Time Camera Classification")

@st.cache_resource
def load_classifier():
    """Load the trained classifier (cached)."""
    try:
        if not MODEL_PATH.exists():
            return None
        return TrashClassifier(model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def capture_and_classify():
    """Capture image from camera and classify."""
    classifier = load_classifier()
    if classifier is None:
        st.error("Model not found. Please train a model first.")
        return
    
    # Camera settings
    st.sidebar.title("📷 Camera Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(CONFIDENCE_THRESHOLD),
        step=0.05
    )
    classifier.confidence_threshold = confidence_threshold
    
    camera_index = st.sidebar.selectbox(
        "Camera Index",
        options=[0, 1, 2],
        index=0,
        help="Select camera index (0 is usually the default camera)"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Camera Capture")
        
        # Placeholder for camera feed
        camera_placeholder = st.empty()
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            start_camera = st.button("🎥 Start Camera", type="primary")
        
        with col_btn2:
            capture_image = st.button("📸 Capture & Classify")
        
        with col_btn3:
            stop_camera = st.button("⏹️ Stop Camera")
    
    with col2:
        st.subheader("🎯 Classification Results")
        result_placeholder = st.empty()
        
        st.subheader("📊 Live Predictions")
        predictions_placeholder = st.empty()
    
    # Camera state management
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    
    # Start camera
    if start_camera:
        try:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
            
            st.session_state.cap = cv2.VideoCapture(camera_index)
            
            if st.session_state.cap.isOpened():
                st.session_state.camera_active = True
                st.success("✅ Camera started successfully!")
            else:
                st.error("❌ Failed to open camera. Check camera index and permissions.")
                st.session_state.camera_active = False
        
        except Exception as e:
            st.error(f"Error starting camera: {str(e)}")
            st.session_state.camera_active = False
    
    # Stop camera
    if stop_camera:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.camera_active = False
        st.info("📷 Camera stopped")
    
    # Camera feed and classification
    if st.session_state.camera_active and st.session_state.cap is not None:
        
        # Continuous feed mode
        if st.sidebar.checkbox("🔄 Continuous Classification", value=False):
            
            # Auto-classification settings
            classification_interval = st.sidebar.slider(
                "Classification Interval (seconds)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
            
            # Initialize timing
            if 'last_classification' not in st.session_state:
                st.session_state.last_classification = 0
            
            # Continuous loop
            frame_placeholder = camera_placeholder.container()
            
            while st.session_state.camera_active:
                ret, frame = st.session_state.cap.read()
                
                if ret:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Check if it's time for classification
                    current_time = time.time()
                    if current_time - st.session_state.last_classification >= classification_interval:
                        
                        # Classify current frame
                        try:
                            result = classifier.predict_image(frame)
                            
                            # Display results
                            with result_placeholder.container():
                                confidence = result['confidence']
                                class_name = result['class_name']
                                meets_threshold = result['meets_threshold']
                                
                                if meets_threshold:
                                    st.success(f"✅ **{class_name.title()}**")
                                    st.metric("Confidence", f"{confidence:.1%}")
                                else:
                                    st.warning(f"⚠️ **{class_name.title()}** (Low Confidence)")
                                    st.metric("Confidence", f"{confidence:.1%}")
                            
                            # Display all predictions
                            with predictions_placeholder.container():
                                st.write("**All Predictions:**")
                                for i, (cls, prob) in enumerate(zip(CLASS_NAMES, result['all_probabilities'])):
                                    st.write(f"{cls.title()}: {prob:.1%}")
                            
                            st.session_state.last_classification = current_time
                        
                        except Exception as e:
                            result_placeholder.error(f"Classification error: {str(e)}")
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.1)
                
                else:
                    st.error("Failed to read from camera")
                    break
        
        else:
            # Manual capture mode
            ret, frame = st.session_state.cap.read()
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Manual classification
                if capture_image:
                    try:
                        with st.spinner("🔍 Classifying image..."):
                            result = classifier.predict_image(frame)
                        
                        # Display results
                        with result_placeholder.container():
                            confidence = result['confidence']
                            class_name = result['class_name']
                            meets_threshold = result['meets_threshold']
                            
                            if meets_threshold:
                                st.success(f"✅ **Predicted: {class_name.title()}**")
                            else:
                                st.warning(f"⚠️ **Predicted: {class_name.title()}** (Low Confidence)")
                            
                            st.metric("Confidence Score", f"{confidence:.1%}")
                            
                            # Show timestamp
                            st.caption(f"Captured at: {time.strftime('%H:%M:%S')}")
                        
                        # Display all predictions
                        with predictions_placeholder.container():
                            st.write("**Detailed Predictions:**")
                            
                            # Create a simple bar chart using text
                            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, result['all_probabilities'])):
                                bar_length = int(prob * 20)  # Scale to 20 characters
                                bar = "█" * bar_length + "░" * (20 - bar_length)
                                st.write(f"{cls.title():<10} {bar} {prob:.1%}")
                    
                    except Exception as e:
                        result_placeholder.error(f"Classification error: {str(e)}")
            
            else:
                camera_placeholder.error("Failed to read from camera")
    
    else:
        # Camera not active
        camera_placeholder.info("📷 Click 'Start Camera' to begin")
        result_placeholder.info("🎯 Results will appear here")
        predictions_placeholder.info("📊 Detailed predictions will appear here")
    
    # Instructions
    st.markdown("---")
    st.subheader("📖 Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Manual Mode:**")
        st.write("1. Click 'Start Camera' to activate camera feed")
        st.write("2. Position waste item in camera view")
        st.write("3. Click 'Capture & Classify' to analyze")
        st.write("4. View results in the right panel")
    
    with col2:
        st.write("**Continuous Mode:**")
        st.write("1. Start camera and enable 'Continuous Classification'")
        st.write("2. Set classification interval")
        st.write("3. System will automatically classify every few seconds")
        st.write("4. Results update in real-time")
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues:**
        
        - **Camera not opening**: Try different camera indices (0, 1, 2)
        - **Permission denied**: Check browser camera permissions
        - **Poor classification**: Ensure good lighting and clear view of item
        - **Slow performance**: Increase classification interval in continuous mode
        
        **Tips for Better Results:**
        
        - Use good lighting conditions
        - Position item clearly in frame
        - Avoid cluttered backgrounds
        - Hold item steady during capture
        """)

def main():
    """Main camera page."""
    
    # Check if model exists
    if not MODEL_PATH.exists():
        st.error("❌ Model not found. Please train a model first using `python scripts/train_model.py`")
        st.stop()
    
    # Main camera interface
    capture_and_classify()
    
    # Cleanup on app exit
    if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
        # Note: Streamlit doesn't have a proper cleanup mechanism
        # Users should manually stop the camera
        pass

if __name__ == "__main__":
    main()