# TrashNet Classification - Modular Architecture

A modular and reusable CNN-based trash classification system built with TensorFlow/Keras. This refactored version provides a clean, maintainable architecture for training and deploying trash classification models.

## 🏗️ Project Structure

```
trashnet/
├── src/                          # Main source code
│   ├── config/                   # Configuration settings
│   │   └── settings.py          # Project constants and paths
│   ├── data/                    # Data processing modules
│   │   ├── loader.py           # Dataset loading utilities
│   │   └── preprocessor.py     # Image preprocessing
│   ├── models/                  # Model architecture and training
│   │   ├── model_builder.py    # Model construction
│   │   └── trainer.py          # Training pipeline
│   ├── inference/               # Inference and prediction
│   │   └── predictor.py        # Classification utilities
│   └── utils/                   # Helper utilities
│       ├── helpers.py          # General utilities
│       └── visualization.py    # Plotting and visualization
├── scripts/                     # Executable scripts
│   ├── train_model.py          # Model training script
│   ├── predict.py              # Prediction script
│   └── preprocess_data.py      # Data preprocessing script
├── examples/                    # Usage examples
│   ├── basic_usage.py          # Basic functionality examples
│   └── camera_demo.py          # Real-time camera demo
├── data/                        # Dataset directory
├── models/                      # Saved models directory
├── logs/                        # Training logs directory
└── requirements.txt             # Python dependencies
```

## 🚀 Features

### Modular Design

- **Separation of Concerns**: Each module handles specific functionality
- **Reusable Components**: Easy to extend and modify
- **Clean Interfaces**: Well-defined APIs between modules

### Multiple Model Architectures

- **MobileNetV2**: Pre-trained transfer learning model
- **Custom CNN**: Lightweight custom architecture
- **Extensible**: Easy to add new architectures

### Comprehensive Data Pipeline

- **Flexible Loading**: Support for various data formats
- **Preprocessing**: Automated image resizing and normalization
- **Augmentation**: Built-in data augmentation for training

### Advanced Training Features

- **Callbacks**: Early stopping, learning rate scheduling, model checkpointing
- **Visualization**: Training metrics and confusion matrices
- **Logging**: Comprehensive training logs

### Inference Capabilities

- **Single Image**: Predict individual images
- **Batch Processing**: Handle multiple images
- **Real-time**: Camera-based live classification
- **Confidence Thresholding**: Configurable confidence levels

### Web Application

- **Streamlit Interface**: User-friendly web interface
- **Multi-page App**: Separate pages for different functionalities
- **Interactive Visualization**: Real-time charts and metrics
- **Batch Upload**: Process multiple images via web interface
- **Camera Integration**: Browser-based camera access

## 📦 Installation

### Option 1: Direct Installation

```bash
# Clone the repository
git clone <repository-url>
cd trashnet

# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

### Option 2: Using Setup

```bash
python setup.py install
```

## 🎯 Quick Start

### 1. Data Preprocessing

```bash
# Preprocess original dataset
python scripts/preprocess_data.py --source-dir data/dataset-original --dest-dir data/dataset-resized
```

### 2. Train a Model

```bash
# Train MobileNet model
python scripts/train_model.py --model-type mobilenet --epochs 50

# Train custom CNN
python scripts/train_model.py --model-type custom --epochs 30 --batch-size 32
```

### 3. Make Predictions

```bash
# Predict single image
python scripts/predict.py --image-path path/to/image.jpg

# Real-time camera prediction
python scripts/predict.py --camera

# Get top-3 predictions
python scripts/predict.py --image-path path/to/image.jpg --top-k 3
```

### 4. Launch Web Application

```bash
# Launch Streamlit web app
python scripts/run_app.py

# Or use make command
make app

# Development mode with auto-reload
make app-dev
```

## 🌐 Streamlit Web Application

The project includes a comprehensive web application built with Streamlit that provides an intuitive interface for trash classification.

### Features

#### 🏠 Main Classification Page

- **Image Upload**: Drag-and-drop or browse to upload images
- **Real-time Prediction**: Instant classification results
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Class Information**: Detailed recycling guidelines for each waste type

#### 📊 Model Analytics Dashboard

- **Training Metrics**: Interactive plots of training history
- **Confusion Matrix**: Heatmap visualization of model performance
- **Classification Report**: Detailed precision, recall, and F1-scores
- **Dataset Statistics**: Class distribution and dataset insights

#### 🎯 Batch Processing

- **Multiple File Upload**: Process many images at once
- **ZIP Archive Support**: Upload and process entire folders
- **Export Results**: Download results as CSV or Excel
- **Progress Tracking**: Real-time processing status

#### 📷 Camera Feed

- **Live Classification**: Real-time camera-based classification
- **Continuous Mode**: Automatic classification at set intervals
- **Manual Capture**: Click-to-classify functionality
- **Adjustable Settings**: Configurable confidence thresholds

### Launching the App

```bash
# Simple launch
python scripts/run_app.py

# Custom port and host
python scripts/run_app.py --port 8080 --host 0.0.0.0

# Development mode with auto-reload
python scripts/run_app.py --dev

# Using Makefile
make app          # Standard launch
make app-dev      # Development mode
```

### App Structure

```
app/
├── streamlit_app.py           # Main application entry point
└── pages/                     # Multi-page components
    ├── 1_📊_Model_Analytics.py    # Analytics dashboard
    ├── 2_🎯_Batch_Processing.py   # Batch upload and processing
    └── 3_📷_Camera_Feed.py        # Real-time camera interface
```

## 💻 Usage Examples

### Basic Usage

```python
from src.data.loader import DatasetLoader
from src.models.model_builder import TrashNetModelBuilder
from src.inference.predictor import TrashClassifier

# Load dataset
loader = DatasetLoader()
X, y = loader.load_full_dataset()

# Build model
builder = TrashNetModelBuilder()
model = builder.build_mobilenet_model()
model = builder.compile_model(model)

# Make predictions
classifier = TrashClassifier(model_path="models/trained_model.keras")
result = classifier.predict_image(image)
print(f"Predicted: {result['class_name']} (confidence: {result['confidence']:.3f})")
```

### Real-time Classification

```python
from src.inference.predictor import RealTimeClassifier

# Initialize real-time classifier
classifier = RealTimeClassifier(
    model_path="models/trained_model.keras",
    confidence_threshold=0.8
)

# Run real-time classification
result = classifier.run_real_time_classification()
if result:
    print(f"Detected: {result['class_name']}")
```

### Custom Training Pipeline

```python
from src.models.trainer import ModelTrainer
from src.models.model_builder import TrashNetModelBuilder

# Setup training
builder = TrashNetModelBuilder()
trainer = ModelTrainer(builder)

# Train model
results = trainer.train_model(
    X_train, y_train, X_val, y_val,
    model_type='mobilenet',
    epochs=50,
    use_data_augmentation=True
)
```

## 🔧 Configuration

All configuration is centralized in `src/config/settings.py`:

```python
# Model configuration
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Classes
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# Inference
CONFIDENCE_THRESHOLD = 0.8
```

## 📊 Dataset

The system supports the TrashNet dataset with 6 classes:

- **Glass**: 501 images
- **Paper**: 594 images
- **Cardboard**: 403 images
- **Plastic**: 482 images
- **Metal**: 410 images
- **Trash**: 137 images

### Dataset Structure

```
data/
├── dataset-original/          # Original high-resolution images
│   ├── glass/
│   ├── paper/
│   ├── cardboard/
│   ├── plastic/
│   ├── metal/
│   └── trash/
└── dataset-resized/          # Preprocessed images (512x384)
    ├── glass/
    ├── paper/
    ├── cardboard/
    ├── plastic/
    ├── metal/
    └── trash/
```

## 🎨 Visualization

The system includes comprehensive visualization tools:

- **Training Metrics**: Loss and accuracy plots
- **Confusion Matrix**: Model performance analysis
- **Class Distribution**: Dataset statistics
- **Sample Predictions**: Visual prediction results

## 🔍 Model Performance

### MobileNetV2 (Transfer Learning)

- **Accuracy**: ~86% (with proper training)
- **Speed**: Fast inference
- **Size**: Compact model

### Custom CNN

- **Accuracy**: ~75% (baseline)
- **Speed**: Very fast inference
- **Size**: Lightweight

## 🛠️ Extending the System

### Adding New Model Architectures

```python
# In src/models/model_builder.py
def build_resnet_model(self):
    base_model = ResNet50(...)
    # Add custom layers
    return model
```

### Adding New Data Sources

```python
# In src/data/loader.py
def load_custom_dataset(self, data_path):
    # Custom loading logic
    return images, labels
```

### Custom Preprocessing

```python
# In src/data/preprocessor.py
def custom_augmentation(self, image):
    # Custom augmentation logic
    return processed_image
```

## 📝 Scripts Reference

### Training Script

```bash
python scripts/train_model.py [OPTIONS]

Options:
  --model-type {mobilenet,custom}  Model architecture
  --epochs INT                     Number of epochs
  --batch-size INT                 Batch size
  --learning-rate FLOAT            Learning rate
  --no-augmentation               Disable data augmentation
  --output-dir PATH               Output directory
```

### Prediction Script

```bash
python scripts/predict.py [OPTIONS]

Options:
  --model-path PATH               Model file path
  --image-path PATH               Image to classify
  --camera                        Use camera input
  --confidence-threshold FLOAT    Confidence threshold
  --top-k INT                     Show top-k predictions
```

### Preprocessing Script

```bash
python scripts/preprocess_data.py [OPTIONS]

Options:
  --source-dir PATH               Original images directory
  --dest-dir PATH                 Output directory
  --height INT                    Target height
  --width INT                     Target width
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the original repository for details.

## 🙏 Acknowledgments

- Original TrashNet dataset creators
- Stanford CS 229 course
- TensorFlow/Keras community

## 📞 Support

For issues and questions:

1. Check the examples in `examples/`
2. Review the configuration in `src/config/settings.py`
3. Check logs in the `logs/` directory
4. Open an issue on the repository

---

**Note**: This is a refactored version of the original TrashNet project with improved modularity, maintainability, and extensibility.
