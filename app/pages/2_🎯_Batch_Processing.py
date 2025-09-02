"""
Batch Processing page for TrashNet Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import zipfile
import io
from PIL import Image
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import *
from src.inference.predictor import TrashClassifier

st.set_page_config(
    page_title="Batch Processing - TrashNet",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Batch Image Processing")

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

def process_uploaded_files(uploaded_files, classifier, confidence_threshold):
    """Process multiple uploaded files."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            # Process image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            # Make prediction
            result = classifier.predict_image(image_array)
            
            # Store result
            results.append({
                'filename': uploaded_file.name,
                'predicted_class': result['class_name'],
                'confidence': result['confidence'],
                'meets_threshold': result['meets_threshold'],
                'all_probabilities': result['all_probabilities']
            })
            
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'predicted_class': 'Error',
                'confidence': 0.0,
                'meets_threshold': False,
                'error': str(e)
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def process_zip_file(zip_file, classifier, confidence_threshold):
    """Process images from a zip file."""
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            st.warning("No image files found in the zip archive.")
            return []
        
        # Process images
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_path in enumerate(image_files):
            try:
                # Update progress
                progress = (i + 1) / len(image_files)
                progress_bar.progress(progress)
                filename = os.path.basename(image_path)
                status_text.text(f"Processing {filename} ({i+1}/{len(image_files)})")
                
                # Load and process image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_array = np.array(image)
                
                # Make prediction
                result = classifier.predict_image(image_array)
                
                # Store result
                results.append({
                    'filename': filename,
                    'predicted_class': result['class_name'],
                    'confidence': result['confidence'],
                    'meets_threshold': result['meets_threshold'],
                    'all_probabilities': result['all_probabilities']
                })
                
            except Exception as e:
                results.append({
                    'filename': os.path.basename(image_path),
                    'predicted_class': 'Error',
                    'confidence': 0.0,
                    'meets_threshold': False,
                    'error': str(e)
                })
        
        progress_bar.empty()
        status_text.empty()
    
    return results

def display_batch_results(results):
    """Display batch processing results."""
    if not results:
        st.warning("No results to display.")
        return
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.subheader("📊 Batch Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(df))
    
    with col2:
        successful = len(df[df['predicted_class'] != 'Error'])
        st.metric("Successfully Processed", successful)
    
    with col3:
        if successful > 0:
            confident = len(df[df['meets_threshold'] == True])
            st.metric("Confident Predictions", confident)
        else:
            st.metric("Confident Predictions", 0)
    
    with col4:
        if successful > 0:
            avg_confidence = df[df['predicted_class'] != 'Error']['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Average Confidence", "N/A")
    
    # Class distribution
    st.subheader("📈 Predicted Class Distribution")
    
    if successful > 0:
        class_counts = df[df['predicted_class'] != 'Error']['predicted_class'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                title="Predicted Classes Count",
                labels={'x': 'Class', 'y': 'Count'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Class Distribution (%)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_errors = st.checkbox("Show Errors", value=True)
    
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)
    
    with col3:
        selected_classes = st.multiselect(
            "Filter by Class",
            options=CLASS_NAMES + ['Error'],
            default=CLASS_NAMES + ['Error']
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if not show_errors:
        filtered_df = filtered_df[filtered_df['predicted_class'] != 'Error']
    
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    filtered_df = filtered_df[filtered_df['predicted_class'].isin(selected_classes)]
    
    # Display table
    display_columns = ['filename', 'predicted_class', 'confidence', 'meets_threshold']
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        column_config={
            'filename': 'File Name',
            'predicted_class': 'Predicted Class',
            'confidence': st.column_config.ProgressColumn(
                'Confidence',
                min_value=0.0,
                max_value=1.0,
                format="%.1%"
            ),
            'meets_threshold': 'Meets Threshold'
        }
    )
    
    # Download results
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name="batch_classification_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Add summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Total Images', 'Successfully Processed', 'Confident Predictions', 'Average Confidence'],
                'Value': [
                    len(df),
                    len(df[df['predicted_class'] != 'Error']),
                    len(df[df['meets_threshold'] == True]),
                    f"{df[df['predicted_class'] != 'Error']['confidence'].mean():.1%}" if successful > 0 else "N/A"
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📊 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name="batch_classification_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    """Main batch processing page."""
    
    # Load classifier
    classifier = load_classifier()
    if classifier is None:
        st.error("Model not found. Please train a model first using `python scripts/train_model.py`")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar settings
    st.sidebar.title("🔧 Processing Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(CONFIDENCE_THRESHOLD),
        step=0.05
    )
    classifier.confidence_threshold = confidence_threshold
    
    # Main content
    st.subheader("📁 Upload Images for Batch Processing")
    
    # Upload options
    upload_option = st.radio(
        "Choose upload method:",
        ["Multiple Files", "ZIP Archive"]
    )
    
    if upload_option == "Multiple Files":
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple image files for batch classification"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Process Images", type="primary"):
                with st.spinner("Processing images..."):
                    results = process_uploaded_files(uploaded_files, classifier, confidence_threshold)
                
                if results:
                    display_batch_results(results)
                    
                    # Store results in session state
                    st.session_state['batch_results'] = results
    
    elif upload_option == "ZIP Archive":
        zip_file = st.file_uploader(
            "Choose a ZIP file containing images",
            type=['zip'],
            help="Upload a ZIP archive containing image files"
        )
        
        if zip_file:
            st.info(f"📦 ZIP file uploaded: {zip_file.name}")
            
            if st.button("🚀 Process ZIP Archive", type="primary"):
                with st.spinner("Extracting and processing images..."):
                    results = process_zip_file(zip_file, classifier, confidence_threshold)
                
                if results:
                    display_batch_results(results)
                    
                    # Store results in session state
                    st.session_state['batch_results'] = results
    
    # Display previous results if available
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Previous Results")
        
        if st.button("🔄 Show Previous Results"):
            display_batch_results(st.session_state['batch_results'])

if __name__ == "__main__":
    main()