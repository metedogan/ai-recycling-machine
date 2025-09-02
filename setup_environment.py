#!/usr/bin/env python3
"""
Quick environment setup script for TrashNet.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def main():
    """Main setup function."""
    print("🚀 TrashNet Environment Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip"):
        print("⚠️ Pip upgrade failed, continuing anyway...")
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                          "Installing requirements"):
            print("❌ Failed to install requirements")
            return False
    else:
        print("⚠️ requirements.txt not found, installing core packages...")
        core_packages = [
            "tensorflow>=2.10.0",
            "opencv-python>=4.5.0", 
            "streamlit>=1.28.0",
            "numpy>=1.21.0",
            "pillow>=8.0.0",
            "plotly>=5.15.0",
            "pandas>=1.5.0"
        ]
        
        for package in core_packages:
            if not run_command(f"{sys.executable} -m pip install {package}", 
                              f"Installing {package.split('>=')[0]}"):
                print(f"❌ Failed to install {package}")
                return False
    
    # Test imports
    print("\n🔍 Testing critical imports...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("tensorflow", "TensorFlow"), 
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Try installing manually:")
        for module, name in test_imports:
            if name in failed_imports:
                if module == "cv2":
                    print(f"   pip install opencv-python")
                elif module == "PIL":
                    print(f"   pip install Pillow")
                else:
                    print(f"   pip install {module}")
        return False
    
    # Create directories
    print("\n📁 Creating project directories...")
    directories = ["models", "logs", "data/dataset-resized", "data/dataset-original"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    print("\n🎉 Environment setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Place your dataset in data/dataset-original/")
    print("2. Run: python scripts/preprocess_data.py")
    print("3. Run: python scripts/train_model.py")
    print("4. Run: python scripts/run_app.py")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 If you continue to have issues, try:")
        print("   - Using a virtual environment")
        print("   - Installing packages one by one")
        print("   - Checking your Python version")
        sys.exit(1)
    else:
        print("\n✨ Ready to use TrashNet!")
        sys.exit(0)