# 🗂️ TrashNet AI Classifier

**Instantly classify waste with AI - No setup hassle!**

Upload a photo → Get instant classification → Improve recycling! 🌱

## ⚡ Quick Start (30 seconds)

### 1. Install & Run
```bash
python setup.py
```
That's it! The app will open in your browser automatically.

### 2. Use the App
- **📸 Upload Image**: Drag & drop any waste photo
- **🤖 Get Results**: Instant AI classification
- **♻️ Learn**: See recycling information

## 📱 What It Does

| Upload This | Get This Result | Recycling Info |
|-------------|-----------------|----------------|
| 🍾 Glass bottle | "Glass - 95% confident" | ♻️ Highly recyclable |
| 📄 Newspaper | "Paper - 87% confident" | ♻️ Recyclable |
| 📦 Amazon box | "Cardboard - 92% confident" | ♻️ Highly recyclable |
| 🥤 Plastic bottle | "Plastic - 89% confident" | ⚠️ Check recycling codes |
| 🥫 Soda can | "Metal - 96% confident" | ♻️ Very valuable |
| 🗑️ Mixed waste | "Trash - 78% confident" | ❌ Non-recyclable |

## 🔧 Requirements

- **Python 3.8+** (check with `python --version`)
- **Internet connection** (for initial setup)
- **4GB RAM** (recommended)

## 🆘 Problems?

**App won't start?**
```bash
pip install streamlit opencv-python tensorflow
python -m streamlit run app.py
```

**Model missing?**
- Make sure `model.keras` is in the same folder as `setup.py`

**Still stuck?**
- Check `HELP.md` for detailed troubleshooting

## 🌍 Environmental Impact

Every correct classification helps:
- ✅ **Reduce contamination** in recycling streams
- ✅ **Increase recycling rates** by proper sorting  
- ✅ **Educate users** about waste categories
- ✅ **Support sustainability** efforts

## 📊 Technical Details

- **AI Model**: MobileNetV2 (75% accuracy)
- **Categories**: 6 waste types
- **Speed**: <1 second per image
- **Training Data**: 2,500+ real waste images

---

**Ready to classify? Run `python setup.py` and start helping the environment! 🚀**