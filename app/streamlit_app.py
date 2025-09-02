#!/usr/bin/env python3
"""
Streamlit web application for TrashNet classification.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Handle OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("❌ OpenCV (cv2) is not installed. Please install it using: `pip install opencv-python`")

# Import project modules
try:
    from src.config.settings import *
    from src.inference.predictor import TrashClassifier
    from src.utils.helpers import Logger
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"❌ Failed to import project modules: {e}")
    st.info("Please ensure all dependencies are installed: `pip install -r requirements.txt`")


# Page configuration
st.set_page_config(
    page_title="TrashNet Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confident-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .uncertain-prediction {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .class-info {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Load the trained classifier (cached)."""
    if not MODULES_AVAILABLE or not CV2_AVAILABLE:
        return None
    
    try:
        if not MODEL_PATH.exists():
            st.error(f"Model not found at {MODEL_PATH}")
            st.info("Please train a model first using: `python scripts/train_model.py`")
            return None
        
        classifier = TrashClassifier(model_path=MODEL_PATH)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def process_uploaded_image(uploaded_file):
    """Process uploaded image file."""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array, image_bgr
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


def display_prediction_results(result, image_rgb):
    """Display prediction results with visualizations."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Input Image")
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Main prediction
        confidence = result['confidence']
        class_name = result['class_name']
        meets_threshold = result['meets_threshold']
        
        # Prediction box styling
        box_class = "confident-prediction" if meets_threshold else "uncertain-prediction"
        status_icon = "✅" if meets_threshold else "⚠️"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h3>{status_icon} Predicted Class: {class_name.title()}</h3>
            <h4>Confidence: {confidence:.1%}</h4>
            <p>Threshold: {CONFIDENCE_THRESHOLD:.1%} 
            {"(Met)" if meets_threshold else "(Not Met)"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # All probabilities
        st.subheader("📊 All Class Probabilities")
        probs_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Probability': result['all_probabilities']
        }).sort_values('Probability', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            probs_df, 
            x='Probability', 
            y='Class',
            orientation='h',
            color='Probability',
            color_continuous_scale='Viridis',
            title="Classification Probabilities"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_class_information():
    """Display information about trash classes."""
    st.subheader("♻️ Waste Classification Guide")
    
    class_info = {
        "glass": {
            "icon": "🍾",
            "description": "Glass bottles, jars, containers",
            "examples": "Wine bottles, jam jars, glass containers",
            "recycling": "Highly recyclable - can be recycled indefinitely"
        },
        "paper": {
            "icon": "📄",
            "description": "Paper products, documents, newspapers",
            "examples": "Newspapers, magazines, office paper, books",
            "recycling": "Recyclable - can be made into new paper products"
        },
        "cardboard": {
            "icon": "📦",
            "description": "Cardboard boxes, packaging materials",
            "examples": "Shipping boxes, cereal boxes, packaging",
            "recycling": "Highly recyclable - commonly recycled material"
        },
        "plastic": {
            "icon": "🥤",
            "description": "Plastic containers, bottles, packaging",
            "examples": "Water bottles, food containers, plastic bags",
            "recycling": "Varies by type - check recycling codes"
        },
        "metal": {
            "icon": "🥫",
            "description": "Metal cans, containers, foil",
            "examples": "Aluminum cans, tin cans, metal containers",
            "recycling": "Highly recyclable - valuable recycling material"
        },
        "trash": {
            "icon": "🗑️",
            "description": "Non-recyclable waste materials",
            "examples": "Mixed materials, contaminated items",
            "recycling": "Not recyclable - goes to landfill or incineration"
        }
    }
    
    cols = st.columns(3)
    for i, (class_name, info) in enumerate(class_info.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="class-info">
                <h4>{info['icon']} {class_name.title()}</h4>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Examples:</strong> {info['examples']}</p>
                <p><strong>Recycling:</strong> {info['recycling']}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">♻️ TrashNet Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered waste classification for better recycling**")
    
    # Check dependencies first
    if not CV2_AVAILABLE or not MODULES_AVAILABLE:
        st.error("❌ Missing dependencies detected!")
        
        with st.expander("🔧 Installation Instructions"):
            st.markdown("""
            **To fix this issue, run the following commands:**
            
            ```bash
            # Install all dependencies
            pip install -r requirements.txt
            
            # Or install OpenCV specifically
            pip install opencv-python
            
            # Or use the installation script
            python install_dependencies.py
            ```
            
            **Alternative OpenCV packages to try:**
            - `pip install opencv-python-headless` (for servers)
            - `pip install opencv-contrib-python` (with extra features)
            """)
        
        st.info("💡 After installing dependencies, refresh this page.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Model status
    classifier = load_classifier()
    if classifier is None:
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Confidence threshold adjustment
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(CONFIDENCE_THRESHOLD),
        step=0.05,
        help="Minimum confidence required for a prediction to be considered reliable"
    )
    classifier.confidence_threshold = confidence_threshold
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Classify Image", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Upload an Image for Classification")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of waste material to classify"
        )
        
        if uploaded_file is not None:
            # Process image
            image_rgb, image_bgr = process_uploaded_image(uploaded_file)
            
            if image_rgb is not None and image_bgr is not None:
                # Make prediction
                with st.spinner("🔍 Analyzing image..."):
                    result = classifier.predict_image(image_bgr)
                
                # Display results
                display_prediction_results(result, image_rgb)
                
                # Additional analysis
                st.subheader("🔍 Detailed Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Class", result['class_name'].title())
                with col2:
                    st.metric("Confidence Score", f"{result['confidence']:.1%}")
                with col3:
                    status = "Reliable" if result['meets_threshold'] else "Uncertain"
                    st.metric("Prediction Status", status)
                
                # Top-k predictions
                st.subheader("🏆 Top 3 Predictions")
                top_predictions = classifier.get_top_k_predictions(image_bgr, k=3)
                
                for i, pred in enumerate(top_predictions, 1):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"**#{i}**")
                    with col2:
                        st.write(f"**{pred['class_name'].title()}**")
                    with col3:
                        st.write(f"{pred['confidence']:.1%}")
        
        else:
            st.info("👆 Please upload an image to get started!")
            
            # Sample images section
            st.subheader("📋 Try Sample Images")
            st.write("Don't have an image? Try these sample classifications:")
            
            sample_images_info = """
            - **Glass**: Wine bottles, mason jars, glass containers
            - **Paper**: Newspapers, magazines, office documents
            - **Cardboard**: Amazon boxes, cereal boxes, packaging
            - **Plastic**: Water bottles, food containers, plastic bags
            - **Metal**: Aluminum cans, tin cans, metal containers
            - **Trash**: Mixed waste, contaminated materials
            """
            st.markdown(sample_images_info)
    
    with tab2:
        st.header("📊 Batch Image Analysis")
        st.info("🚧 Coming Soon: Upload multiple images for batch processing")
        
        # Placeholder for batch analysis
        st.subheader("Features in Development:")
        features = [
            "📁 Upload multiple images at once",
            "📈 Generate classification reports",
            "💾 Export results to CSV/Excel",
            "📊 Visualize batch statistics",
            "🔄 Compare model performance"
        ]
        
        for feature in features:
            st.write(f"- {feature}")
    
    with tab3:
        st.header("ℹ️ About TrashNet Classifier")
        
        # Project information
        st.subheader("🎯 Project Overview")
        st.write("""
        TrashNet Classifier is an AI-powered system that automatically classifies waste materials 
        into different categories to improve recycling efficiency. The system uses deep learning 
        models trained on thousands of waste images.
        """)
        
        # Model information
        st.subheader("🤖 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Architecture:** MobileNetV2 (Transfer Learning)")
            st.write("**Classes:** 6 waste categories")
            st.write("**Input Size:** 224x224 pixels")
        
        with col2:
            st.write("**Training Data:** TrashNet Dataset")
            st.write("**Accuracy:** ~75% on test set")
            st.write("**Framework:** TensorFlow/Keras")
        
        # Class information
        display_class_information()
        
        # Technical details
        st.subheader("🔧 Technical Details")
        with st.expander("View Technical Specifications"):
            st.code(f"""
Model Path: {MODEL_PATH}
Input Shape: {INPUT_SHAPE}
Classes: {CLASS_NAMES}
Default Confidence Threshold: {CONFIDENCE_THRESHOLD}
            """)
        
        # Usage instructions
        st.subheader("📖 How to Use")
        st.write("""
        1. **Upload Image**: Click on the file uploader and select an image
        2. **Wait for Analysis**: The AI model will process your image
        3. **Review Results**: Check the predicted class and confidence score
        4. **Interpret Confidence**: Green box = reliable, Yellow box = uncertain
        5. **Check All Probabilities**: See how the model scored each class
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("**TrashNet Classifier** - Helping make recycling smarter with AI 🌱")


if __name__ == "__main__":
    main()