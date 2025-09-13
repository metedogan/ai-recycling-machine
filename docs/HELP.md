# 🆘 TrashNet Help & Troubleshooting

## Quick Fixes

### ❌ "No module named 'cv2'"
```bash
pip install opencv-python
```

### ❌ "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### ❌ "No module named 'streamlit'"
```bash
pip install streamlit
```

### ❌ "Model not found"
- Make sure `model.keras` is in the same folder as `setup.py`
- Check the file isn't corrupted (should be ~25MB)

### ❌ App won't start
```bash
# Try manual launch
python -m streamlit run app.py

# Or install everything manually
pip install streamlit opencv-python tensorflow numpy pillow plotly pandas
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Storage**: 1GB free space
- **Internet**: Required for initial setup

## Getting Better Results

### 📸 Photo Tips
- **Good lighting** - use natural light or bright room lighting
- **Clean background** - avoid cluttered scenes  
- **Single item** - focus on one piece of waste
- **Full view** - show the entire item
- **Steady shot** - avoid blurry photos

### 🎯 Item Preparation
- **Clean items** when possible
- **Remove excess labels** if easy
- **Show typical view** (how you'd normally see it)
- **Avoid mixed materials** in one photo

## Understanding Results

### Confidence Levels
- **90%+**: Very reliable
- **70-90%**: Good confidence  
- **50-70%**: Uncertain - double check
- **<50%**: Low confidence - try different photo

### Categories Explained
- **Glass**: Bottles, jars, containers (not windows/mirrors)
- **Paper**: Newspapers, documents, books (not cardboard)
- **Cardboard**: Boxes, packaging (corrugated material)
- **Plastic**: Bottles, containers, bags (check recycling codes)
- **Metal**: Cans, containers, foil (aluminum, steel)
- **Trash**: Mixed materials, contaminated items

## Technical Issues

### Python Version Check
```bash
python --version
# Should show 3.8 or higher
```

### Package Installation Issues
```bash
# Try user installation
pip install --user streamlit opencv-python tensorflow

# Or use conda
conda install streamlit opencv tensorflow
```

### Model Issues
- **File size**: model.keras should be ~25MB
- **Location**: Must be in same folder as setup.py
- **Permissions**: Make sure file is readable

### Performance Issues
- **Slow predictions**: Normal on older computers
- **Memory errors**: Close other applications
- **Crashes**: Try smaller images

## Alternative Launch Methods

### Method 1: Direct Streamlit
```bash
python -m streamlit run app.py
```

### Method 2: Manual Setup
```bash
pip install streamlit opencv-python tensorflow
python app.py
```

### Method 3: Step by Step
```bash
# 1. Install Python packages
pip install streamlit
pip install opencv-python  
pip install tensorflow
pip install numpy pillow plotly pandas

# 2. Run app
python -m streamlit run app.py
```

## Still Need Help?

### Check These First
1. **Python version** - must be 3.8+
2. **Model file** - model.keras in correct location
3. **Internet connection** - needed for package installation
4. **Disk space** - need ~1GB free

### Error Messages
- **Copy the exact error message**
- **Note what you were doing when it happened**
- **Try the suggested fixes above**

### System Info
Run this to get system information:
```bash
python --version
pip list | grep -E "(streamlit|opencv|tensorflow)"
ls -la model.keras
```

## Common Solutions

### "Permission Denied"
```bash
pip install --user streamlit opencv-python tensorflow
```

### "Command not found: python"
Try `python3` instead of `python`

### "Port already in use"
```bash
python -m streamlit run app.py --server.port 8502
```

### "Browser won't open"
Manually go to: `http://localhost:8501`

---

**Still stuck? The error message usually contains the solution!**