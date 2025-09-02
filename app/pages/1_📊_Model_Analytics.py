"""
Model Analytics page for TrashNet Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import *
from src.utils.helpers import FileManager
from src.data.loader import DatasetLoader

st.set_page_config(
    page_title="Model Analytics - TrashNet",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Analytics Dashboard")

# Sidebar
st.sidebar.title("Analytics Options")

def load_training_history():
    """Load training history if available."""
    try:
        history_files = list(MODELS_DIR.glob("training_history_*.json"))
        if history_files:
            latest_file = max(history_files, key=lambda x: x.stat().st_mtime)
            return FileManager.load_json(latest_file), latest_file.name
        return None, None
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None, None

def load_evaluation_results():
    """Load evaluation results if available."""
    try:
        eval_files = list(MODELS_DIR.glob("evaluation_results_*.json"))
        if eval_files:
            latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
            return FileManager.load_json(latest_file), latest_file.name
        return None, None
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        return None, None

def display_training_metrics(history, filename):
    """Display training metrics."""
    st.subheader(f"📈 Training Metrics - {filename}")
    
    # Create metrics dataframe
    epochs = range(1, len(history['accuracy']) + 1)
    df = pd.DataFrame({
        'Epoch': list(epochs) * 4,
        'Metric': (['Training Accuracy'] * len(epochs) + 
                  ['Validation Accuracy'] * len(epochs) +
                  ['Training Loss'] * len(epochs) +
                  ['Validation Loss'] * len(epochs)),
        'Value': (history['accuracy'] + history['val_accuracy'] + 
                 history['loss'] + history['val_loss']),
        'Type': (['Accuracy'] * (len(epochs) * 2) + 
                ['Loss'] * (len(epochs) * 2))
    })
    
    # Plot accuracy
    col1, col2 = st.columns(2)
    
    with col1:
        acc_df = df[df['Type'] == 'Accuracy']
        fig_acc = px.line(
            acc_df, x='Epoch', y='Value', color='Metric',
            title='Training & Validation Accuracy',
            labels={'Value': 'Accuracy'}
        )
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        loss_df = df[df['Type'] == 'Loss']
        fig_loss = px.line(
            loss_df, x='Epoch', y='Value', color='Metric',
            title='Training & Validation Loss',
            labels={'Value': 'Loss'}
        )
        fig_loss.update_layout(height=400)
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Summary metrics
    st.subheader("📋 Training Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Train Accuracy", f"{history['accuracy'][-1]:.3f}")
    with col2:
        st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.3f}")
    with col3:
        st.metric("Best Val Accuracy", f"{max(history['val_accuracy']):.3f}")
    with col4:
        st.metric("Total Epochs", len(history['accuracy']))

def display_confusion_matrix(eval_results):
    """Display confusion matrix."""
    st.subheader("🎯 Confusion Matrix")
    
    cm = np.array(eval_results['confusion_matrix'])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_classification_report(eval_results):
    """Display classification report."""
    st.subheader("📊 Classification Report")
    
    report = eval_results['classification_report']
    
    # Create dataframe for metrics
    metrics_data = []
    for class_name in CLASS_NAMES:
        if class_name in report:
            metrics_data.append({
                'Class': class_name.title(),
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display table
    st.dataframe(df_metrics, use_container_width=True)
    
    # Plot metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig_precision = px.bar(
            df_metrics, x='Class', y='Precision',
            title='Precision by Class',
            color='Precision',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_precision, use_container_width=True)
    
    with col2:
        fig_recall = px.bar(
            df_metrics, x='Class', y='Recall',
            title='Recall by Class',
            color='Recall',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig_recall, use_container_width=True)

def display_dataset_statistics():
    """Display dataset statistics."""
    st.subheader("📈 Dataset Statistics")
    
    try:
        loader = DatasetLoader()
        X, y = loader.load_full_dataset()
        
        if len(X) > 0:
            distribution = loader.get_class_distribution(y)
            
            # Create distribution dataframe
            dist_df = pd.DataFrame([
                {'Class': class_name.title(), 'Count': count}
                for class_name, count in distribution.items()
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    dist_df, x='Class', y='Count',
                    title='Dataset Class Distribution',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    dist_df, values='Count', names='Class',
                    title='Class Distribution (Percentage)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Dataset Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(X))
            with col2:
                st.metric("Number of Classes", len(CLASS_NAMES))
            with col3:
                st.metric("Largest Class", max(distribution, key=distribution.get).title())
            with col4:
                st.metric("Smallest Class", min(distribution, key=distribution.get).title())
        
        else:
            st.warning("No dataset found. Please ensure the dataset is properly loaded.")
    
    except Exception as e:
        st.error(f"Error loading dataset statistics: {e}")

def main():
    """Main analytics page."""
    
    # Load data
    history, history_file = load_training_history()
    eval_results, eval_file = load_evaluation_results()
    
    # Analytics options
    show_training = st.sidebar.checkbox("Show Training Metrics", value=True)
    show_evaluation = st.sidebar.checkbox("Show Evaluation Results", value=True)
    show_dataset = st.sidebar.checkbox("Show Dataset Statistics", value=True)
    
    # Display sections
    if show_training and history:
        display_training_metrics(history, history_file)
        st.markdown("---")
    elif show_training:
        st.warning("No training history found. Train a model first using `python scripts/train_model.py`")
    
    if show_evaluation and eval_results:
        display_confusion_matrix(eval_results)
        st.markdown("---")
        display_classification_report(eval_results)
        st.markdown("---")
    elif show_evaluation:
        st.warning("No evaluation results found. Train and evaluate a model first.")
    
    if show_dataset:
        display_dataset_statistics()

if __name__ == "__main__":
    main()