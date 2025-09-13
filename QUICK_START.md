# TrashNet AI - Quick Start

## For Regular Users

### 1. Install & Run (One Command)
```bash
python setup.py
```
This will:
- Install all requirements
- Launch the web app
- Open in your browser

### 2. Use the App
1. **Upload** a photo of waste
2. **Get** instant AI classification  
3. **Learn** recycling information

## For Developers

### Project Structure
```
trashnet/
├── app.py              # Main application
├── setup.py            # Installer
├── models/             # AI models
├── notebooks/          # Development
├── scripts/            # Utilities
├── data/               # Training data
└── docs/               # Documentation
```

### Development Commands
```bash
# Run app directly
streamlit run app.py

# Train new model
python scripts/train_model.py

# Process data
python scripts/preprocess_data.py
```

## Need Help?
- Check `docs/HELP.md` for troubleshooting
- See `docs/` folder for detailed documentation