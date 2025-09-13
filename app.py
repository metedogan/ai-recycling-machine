#!/usr/bin/env python3
"""
TrashNet AI Classifier - Simple Streamlit App
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import tensorflow as tf
from pathlib import Path

# Page config
st.set_page_config(
    page_title="TrashNet AI Classifier",
    page_icon="🗂️",
    layout="wide"
)

# Constants
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
MODEL_PATH = "models/model.keras"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .confident {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .uncertain {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the AI model with multiple fallback methods."""
    if not Path(MODEL_PATH).exists():
        st.error(f"❌ AI model not found: {MODEL_PATH}")
        st.info("Please ensure model.keras is in the models/ folder")
        return None
    
    # Try multiple loading approaches
    loading_methods = [
        ("Standard loading with compile=False", lambda: tf.keras.models.load_model(MODEL_PATH, compile=False)),
        ("Loading with custom objects", lambda: tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=None)),
        ("Loading with safe mode", lambda: tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)),
    ]
    
    for method_name, load_func in loading_methods:
        try:
            st.info(f"🔄 Trying: {method_name}")
            model = load_func()
            
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success(f"✅ Model loaded successfully using: {method_name}")
            return model
            
        except Exception as e:
            st.warning(f"❌ {method_name} failed: {str(e)[:100]}...")
            continue
    
    # If all methods fail, show comprehensive error
    st.error("❌ All model loading methods failed!")
    st.error("This is likely a TensorFlow version compatibility issue.")
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        **Try these solutions:**
        
        1. **Update TensorFlow:**
        ```bash
        pip install tensorflow==2.15.0
        ```
        
        2. **Or try older version:**
        ```bash
        pip install tensorflow==2.13.0
        ```
        
        3. **Check model file:**
        - Model should be ~25MB
        - Try re-downloading the model
        
        4. **Alternative: Use demo mode**
        - App will run without AI predictions
        """)
    
    return None

def classify_image(image, model):
    """Classify an image."""
    if model is None:
        # Demo mode - return random predictions
        import random
        random.seed(42)  # Consistent demo results
        
        # Generate realistic-looking demo predictions
        demo_probs = [random.uniform(0.05, 0.95) for _ in CLASS_NAMES]
        demo_probs = np.array(demo_probs)
        demo_probs = demo_probs / np.sum(demo_probs)  # Normalize
        
        class_index = np.argmax(demo_probs)
        class_name = CLASS_NAMES[class_index]
        confidence = demo_probs[class_index]
        
        return {
            'class': class_name,
            'confidence': confidence,
            'all_probabilities': demo_probs,
            'demo_mode': True
        }
    
    try:
        # Preprocess
        image_resized = cv2.resize(image, (224, 224))
        image_array = np.expand_dims(image_resized, axis=0)
        image_normalized = image_array / 255.0
        
        # Predict
        predictions = model.predict(image_normalized, verbose=0)
        confidence = np.max(predictions)
        class_index = np.argmax(predictions)
        class_name = CLASS_NAMES[class_index]
        
        return {
            'class': class_name,
            'confidence': confidence,
            'all_probabilities': predictions[0],
            'demo_mode': False
        }
    except Exception as e:
        st.error(f"Classification error: {e}")
        return None

def main():
    """Main app."""
    
    # Header
    st.markdown('<h1 class="main-header">🗂️ TrashNet AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Upload a waste image and get instant AI classification!**")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Running in DEMO MODE - predictions are simulated")
        st.info("Fix the model loading issue above to get real AI predictions")
    else:
        st.success("✅ AI model loaded and ready!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a photo of waste to classify"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Image", use_column_width=True)
            
            # Convert for processing
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
    
    with col2:
        st.subheader("🤖 AI Results")
        
        if uploaded_file:
            with st.spinner("🔍 AI is analyzing..."):
                result = classify_image(image_bgr, model)
            
            if result:
                # Main result
                confidence = result['confidence']
                class_name = result['class']
                is_confident = confidence > 0.7
                is_demo = result.get('demo_mode', False)
                
                # Result box
                box_class = "confident" if is_confident else "uncertain"
                icon = "✅" if is_confident else "⚠️"
                demo_badge = " 🎭 DEMO" if is_demo else ""
                
                st.markdown(f"""
                <div class="result-box {box_class}">
                    <h2>{icon} {class_name.title()}{demo_badge}</h2>
                    <h3>{confidence:.1%} Confident</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if is_demo:
                    st.info("🎭 This is a simulated prediction - fix model loading for real AI results")
                
                # Recycling info
                recycling_info = {
                    "glass": "♻️ Highly recyclable - can be recycled indefinitely",
                    "paper": "♻️ Recyclable - becomes new paper products", 
                    "cardboard": "♻️ Highly recyclable - commonly accepted",
                    "plastic": "⚠️ Check recycling codes - varies by type",
                    "metal": "♻️ Very valuable - always recycle",
                    "trash": "❌ Non-recyclable - goes to landfill"
                }
                
                st.info(recycling_info[class_name])
                
                # All probabilities
                st.subheader("📊 All Predictions")
                
                probs_df = pd.DataFrame({
                    'Category': [name.title() for name in CLASS_NAMES],
                    'Probability': result['all_probabilities']
                }).sort_values('Probability', ascending=True)
                
                fig = px.bar(
                    probs_df, 
                    x='Probability', 
                    y='Category',
                    orientation='h',
                    color='Probability',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Upload an image to get started!")
            
            st.markdown("""
            **Try these waste types:**
            - 🍾 Glass bottles, jars
            - 📄 Paper, newspapers  
            - 📦 Cardboard boxes
            - 🥤 Plastic bottles, containers
            - 🥫 Metal cans, foil
            - 🗑️ Mixed waste
            """)
    
    # Tips section
    st.markdown("---")
    st.subheader("💡 Tips for Best Results")
    
    tip_cols = st.columns(4)
    
    with tip_cols[0]:
        st.markdown("**📸 Good Photo**\n- Clear lighting\n- Single item\n- Full view")
    
    with tip_cols[1]:
        st.markdown("**🎯 Clean Item**\n- Remove labels\n- Clean surface\n- Typical angle")
    
    with tip_cols[2]:
        st.markdown("**📱 File Types**\n- JPG, PNG\n- Any size\n- Mobile photos OK")
    
    with tip_cols[3]:
        st.markdown("**♻️ Categories**\n- 6 waste types\n- ~75% accuracy\n- Recycling info")

if __name__ == "__main__":
    main()