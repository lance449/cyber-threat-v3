# Setup Instructions for New Device

This repository has been cleaned and optimized to prevent file corruption issues. Follow these steps to set up the system on a new device.

## ✅ What Was Fixed

1. **Removed Large Files** - Large CSV files (327MB, 65MB) removed from git tracking
2. **Removed Binary Files** - Model files (.joblib, .pkl) removed from git tracking  
3. **Removed Dependencies** - node_modules and venv directories excluded
4. **Fixed Line Endings** - Configured for cross-platform compatibility
5. **Updated .gitignore** - Properly excludes problematic files

## 🚀 Setup Steps

### 1. Clone the Repository
```bash
git clone https://github.com/lance449/cyber-threat-v3.git
cd cyber-threat-v3
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install
```

### 4. Prepare Data Files (If Needed)

The large CSV files and model files are NOT in the repository. You have two options:

**Option A: Use existing data files** (if you have them)
- Place CSV files in `backend/data/consolidated/`
- Place model files in `backend/models/`

**Option B: Generate new data and models**
```bash
cd backend
python train_system_fixed.py
```

This will:
- Process the data files (if available)
- Train the models
- Generate all necessary files

### 5. Run the System

```bash
# From project root
python start_system.py
```

Or run separately:

```bash
# Terminal 1 - Backend
cd backend
python app_simple.py

# Terminal 2 - Frontend  
cd frontend
npm start
```

## 📁 Important Files

### Files That Are NOT in Repository:
- ❌ `backend/data/consolidated/*.csv` (large data files)
- ❌ `backend/models/**/*.joblib` (trained models)
- ❌ `backend/models/**/*.pkl` (trained models)
- ❌ `backend/data/processed/*.csv` (processed data)
- ❌ `backend/data/processed/*.joblib` (preprocessing objects)
- ❌ `node_modules/` (npm dependencies)
- ❌ `venv/` (Python virtual environment)

### Files That ARE in Repository:
- ✅ All source code (`.py`, `.js`, `.jsx` files)
- ✅ Configuration files (`.json`, `.txt`)
- ✅ Requirements files (`requirements.txt`, `package.json`)
- ✅ Documentation (`.md` files)
- ✅ Small data files (patterns, feature names, etc.)

## 🔧 Troubleshooting

### If models are missing:
```bash
cd backend
python train_system_fixed.py
```

### If dependencies are missing:
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### If you get import errors:
- Make sure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

## 📝 Notes

- The repository uses Git LFS for some large files (if configured)
- All source code and configuration is included
- Data files and models need to be generated or provided separately
- The system will work once models are trained or provided

## 🔗 Repository Information

- **Repository URL**: https://github.com/lance449/cyber-threat-v3.git
- **Branch**: main
- **Remote Name**: v3

