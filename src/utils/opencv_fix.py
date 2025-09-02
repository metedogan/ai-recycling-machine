"""
OpenCV import fix for different environments.
"""

import sys
import subprocess


def install_opencv():
    """Install OpenCV if not available."""
    try:
        import cv2
        return True
    except ImportError:
        print("OpenCV not found. Attempting to install...")
        
        # Try different OpenCV packages
        opencv_packages = [
            'opencv-python',
            'opencv-python-headless',
            'cv2-python'
        ]
        
        for package in opencv_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                import cv2
                print(f"✅ Successfully installed {package}")
                return True
            except (subprocess.CalledProcessError, ImportError):
                continue
        
        print("❌ Failed to install OpenCV. Please install manually:")
        print("pip install opencv-python")
        return False


def safe_import_cv2():
    """Safely import OpenCV with fallback options."""
    try:
        import cv2
        return cv2
    except ImportError:
        if install_opencv():
            import cv2
            return cv2
        else:
            raise ImportError(
                "OpenCV (cv2) is required but not installed. "
                "Please install it using: pip install opencv-python"
            )


# For easy importing
cv2 = safe_import_cv2()