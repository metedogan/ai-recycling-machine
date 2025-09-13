#!/usr/bin/env python3
"""
One-click setup and launch for TrashNet Classifier.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def print_welcome():
    """Print welcome message."""
    print("\n" + "🗂️" + "="*50)
    print("    TrashNet AI Classifier - Setup")
    print("    Classify waste instantly with AI!")
    print("="*52)


def install_packages():
    """Install required packages."""
    print("\n📦 Installing AI packages...")
    
    packages = [
        "streamlit>=1.28.0",
        "opencv-python>=4.5.0", 
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "plotly>=5.15.0",
        "pandas>=1.5.0"
    ]
    
    for package in packages:
        name = package.split('>=')[0]
        print(f"   Installing {name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"   ⚠️ {name} installation failed, trying alternative...")
            # Try without version constraint
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", name], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"   ❌ Could not install {name}")
                return False
    
    print("   ✅ All packages installed!")
    return True


def check_model():
    """Check if model exists."""
    model_path = Path("models/model.keras")
    if model_path.exists():
        print("✅ AI model found and ready!")
        return True
    else:
        print("❌ AI model not found at models/model.keras")
        print("   Please ensure model.keras is in the models/ folder")
        return False


def test_setup():
    """Test if everything works."""
    print("\n🔍 Testing setup...")
    
    try:
        import streamlit
        import cv2
        import tensorflow
        print("   ✅ All packages working!")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def launch_app():
    """Launch the app."""
    print("\n🚀 Launching TrashNet Classifier...")
    print("   Opening in your browser in 3 seconds...")
    
    # Wait a moment
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open('http://localhost:8501')
    except:
        pass
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Run 'python setup.py' to restart!")
    except Exception as e:
        print(f"\n❌ Launch error: {e}")
        print("   Try manually: python -m streamlit run app.py")


def main():
    """Main setup function."""
    print_welcome()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. You have:", sys.version)
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Install packages
    if not install_packages():
        print("\n❌ Package installation failed")
        print("💡 Try manually: pip install streamlit opencv-python tensorflow")
        return
    
    # Check model
    if not check_model():
        print("\n⚠️ Setup incomplete - model missing")
        return
    
    # Test setup
    if not test_setup():
        print("\n❌ Setup test failed")
        return
    
    print("\n🎉 Setup complete!")
    
    # Launch app
    launch_app()


if __name__ == "__main__":
    main()