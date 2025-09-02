#!/usr/bin/env python3
"""
Installation script for TrashNet dependencies.
"""

import subprocess
import sys
import pkg_resources
from pathlib import Path


def check_package(package_name):
    """Check if a package is installed."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    """Main installation function."""
    print("🚀 Installing TrashNet dependencies...")
    
    # Read requirements
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📦 Found {len(requirements)} packages to install")
    
    # Install packages
    failed_packages = []
    
    for requirement in requirements:
        package_name = requirement.split('>=')[0].split('==')[0]
        
        print(f"\n📥 Installing {package_name}...")
        
        if install_package(requirement):
            print(f"✅ Successfully installed {package_name}")
        else:
            failed_packages.append(requirement)
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Installation Summary")
    print(f"{'='*50}")
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print(f"\n💡 Try installing manually:")
        print(f"pip install {' '.join(failed_packages)}")
        return False
    else:
        print("✅ All packages installed successfully!")
        
        # Verify critical packages
        critical_packages = ['cv2', 'tensorflow', 'streamlit', 'numpy']
        print(f"\n🔍 Verifying critical packages...")
        
        for pkg in critical_packages:
            try:
                if pkg == 'cv2':
                    import cv2
                    print(f"✅ OpenCV version: {cv2.__version__}")
                elif pkg == 'tensorflow':
                    import tensorflow as tf
                    print(f"✅ TensorFlow version: {tf.__version__}")
                elif pkg == 'streamlit':
                    import streamlit as st
                    print(f"✅ Streamlit version: {st.__version__}")
                elif pkg == 'numpy':
                    import numpy as np
                    print(f"✅ NumPy version: {np.__version__}")
            except ImportError as e:
                print(f"❌ Failed to import {pkg}: {e}")
                return False
        
        print(f"\n🎉 Installation completed successfully!")
        print(f"🚀 You can now run: python scripts/run_app.py")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)