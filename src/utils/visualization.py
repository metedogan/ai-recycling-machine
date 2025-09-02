"""Visualization utilities for TrashNet project."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TrainingVisualizer:
    """Visualizes training metrics and results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None) -> None:
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].axis('off')
        
        # Final metrics summary
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        axes[1, 1].text(0.1, 0.8, f'Final Training Accuracy: {final_train_acc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Final Validation Accuracy: {final_val_acc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Final Training Loss: {final_train_loss:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Final Validation Loss: {final_val_loss:.4f}', fontsize=12)
        axes[1, 1].set_title('Final Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, distribution: Dict[str, int], 
                              save_path: Optional[str] = None) -> None:
        """Plot class distribution."""
        classes = list(distribution.keys())
        counts = list(distribution.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, images: np.ndarray, true_labels: np.ndarray,
                              predictions: np.ndarray, class_names: List[str],
                              num_samples: int = 12, save_path: Optional[str] = None) -> None:
        """Plot sample predictions with true and predicted labels."""
        num_samples = min(num_samples, len(images))
        rows = int(np.ceil(num_samples / 4))
        
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
        
        for i in range(num_samples):
            if i < len(images):
                # Display image
                image = images[i]
                if image.shape[-1] == 3:  # RGB
                    axes[i].imshow(image)
                else:  # Grayscale
                    axes[i].imshow(image, cmap='gray')
                
                # Get labels
                true_class = class_names[true_labels[i]]
                pred_class = class_names[np.argmax(predictions[i])]
                confidence = np.max(predictions[i])
                
                # Set title with color coding
                color = 'green' if true_class == pred_class else 'red'
                axes[i].set_title(f'True: {true_class}\\nPred: {pred_class} ({confidence:.2f})',
                                color=color, fontsize=10)
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DatasetVisualizer:
    """Visualizes dataset samples and statistics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def show_sample_images(self, images: np.ndarray, labels: np.ndarray,
                          class_names: List[str], samples_per_class: int = 3,
                          save_path: Optional[str] = None) -> None:
        """Show sample images from each class."""
        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                               figsize=(samples_per_class * 3, num_classes * 3))
        
        for class_idx, class_name in enumerate(class_names):
            # Get images for this class
            class_images = images[labels == class_idx]
            
            for sample_idx in range(samples_per_class):
                row_idx = class_idx
                col_idx = sample_idx
                
                if len(class_images) > sample_idx:
                    if num_classes == 1:
                        ax = axes[col_idx]
                    else:
                        ax = axes[row_idx, col_idx]
                    
                    image = class_images[sample_idx]
                    ax.imshow(image)
                    ax.set_title(f'{class_name}')
                    ax.axis('off')
                else:
                    if num_classes == 1:
                        axes[col_idx].axis('off')
                    else:
                        axes[row_idx, col_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()