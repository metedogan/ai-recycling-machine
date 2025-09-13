# Makefile for TrashNet Classification Project

.PHONY: help install clean preprocess train predict demo test lint format

# Default target
help:
	@echo "TrashNet Classification - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     - User setup (inference only)"
	@echo "  install-dev - Developer setup (full features)"
	@echo "  install-deps - Install user dependencies only"
	@echo "  fix-opencv  - Fix OpenCV installation issues"
	@echo "  clean       - Clean generated files and directories"
	@echo ""
	@echo "Data:"
	@echo "  preprocess  - Preprocess dataset (resize images)"
	@echo ""
	@echo "Usage:"
	@echo "  quickstart  - Complete setup and launch app"
	@echo "  verify      - Check model and dependencies"
	@echo ""
	@echo "Inference:"
	@echo "  predict     - Run prediction on sample image"
	@echo "  demo        - Run camera demo
  app         - Launch Streamlit web app
  app-dev     - Launch Streamlit app in dev mode"
	@echo ""
	@echo "Development:"
	@echo "  test        - Run tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo ""
	@echo "Examples:"
	@echo "  examples    - Run basic usage examples"

# Installation
install:
	python setup_user.py
	@echo "User setup completed!"

install-dev:
	python setup_environment.py
	@echo "Developer setup completed!"

install-deps:
	pip install -r requirements_user.txt
	@echo "User dependencies installed!"

fix-opencv:
	pip install opencv-python
	@echo "OpenCV installation attempted!"

# Clean up
clean:
	rm -rf models/*.keras
	rm -rf logs/*
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Cleanup completed!"

# Data preprocessing
preprocess:
	python scripts/preprocess_data.py
	@echo "Data preprocessing completed!"

# Training
train:
	python scripts/train_model.py --model-type mobilenet --epochs 50
	@echo "MobileNet training completed!"

train-custom:
	python scripts/train_model.py --model-type custom --epochs 30
	@echo "Custom CNN training completed!"

train-quick:
	python scripts/train_model.py --model-type mobilenet --epochs 5 --batch-size 16
	@echo "Quick training completed!"

# Inference
predict:
	@if [ -f "model.keras" ]; then \
		python scripts/predict.py --image-path data/dataset-resized/glass/glass1.jpg; \
	else \
		echo "No trained model found. Run 'make train' first."; \
	fi

demo:
	@if [ -f "model.keras" ]; then \
		python examples/camera_demo.py; \
	else \
		echo "No trained model found. Run 'make train' first."; \
	fi

# Web Application
app:
	python scripts/run_app.py
	@echo "Streamlit app started!"

app-dev:
	python scripts/run_app.py --dev
	@echo "Streamlit app started in development mode!"

# Examples
examples:
	python examples/basic_usage.py
	@echo "Examples completed!"

# Development tools
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "Tests completed!"

lint:
	@echo "Running linting..."
	flake8 src/ scripts/ examples/ --max-line-length=100
	@echo "Linting completed!"

format:
	@echo "Formatting code..."
	black src/ scripts/ examples/ --line-length=100
	isort src/ scripts/ examples/
	@echo "Code formatting completed!"

# Setup development environment
dev-setup: install
	pip install pytest flake8 black isort
	@echo "Development environment setup completed!"

# Full pipeline
full-pipeline: clean preprocess train predict
	@echo "Full pipeline completed!"

# Quick start for new users
quickstart:
	@echo "🚀 TrashNet Quick Start:"
	@echo "1. Setting up environment..."
	make install
	@echo "2. Verifying model..."
	make verify
	@echo "3. Launching app..."
	make app
	@echo "✅ Quick start completed! Check your browser at http://localhost:8501"

verify:
	@echo "🔍 Verifying TrashNet setup..."
	@python -c "import cv2, tensorflow, streamlit; print('✅ All dependencies OK')"
	@python -c "from pathlib import Path; print('✅ Model found' if Path('model.keras').exists() else '❌ Model missing')"
	@echo "Verification completed!"