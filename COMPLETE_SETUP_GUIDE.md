# 🚀 Complete Setup Guide - Running System on Another Device

This comprehensive guide will help you set up and run the Cyber Threat Detection System on a new device from scratch.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Clone Repository](#step-1-clone-repository)
3. [Step 2: Install Dependencies](#step-2-install-dependencies)
4. [Step 3: Copy Required Files](#step-3-copy-required-files)
5. [Step 4: Verify Installation](#step-4-verify-installation)
6. [Step 5: Run the System](#step-5-run-the-system)
7. [Troubleshooting](#troubleshooting)

---

## 📦 Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed
- **Node.js 14+** and **npm** installed
- **Git** installed
- **Internet connection** (for cloning and installing dependencies)
- **Access to model files and processed data** (from main device or online drive)

---

## 📥 Step 1: Clone Repository

Open terminal/command prompt and run:

```bash
git clone https://github.com/lance449/cyber-threat-v3.git
cd cyber-threat-v3
```

This will download all the source code and configuration files.

---

## 🔧 Step 2: Install Dependencies

### 2.1 Backend Dependencies (Python)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2.2 Frontend Dependencies (Node.js)

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install Node.js packages
npm install
```

---

## 📂 Step 3: Copy Required Files

The repository does NOT contain model files or large datasets. You need to copy them from your main device.

### ⚠️ **CRITICAL: Files You MUST Copy**

These files are **REQUIRED** for the system to run:

#### **3.1 Copy Models Folder** (~834 MB)

**From main device:**
- Location: `backend/models/`
- Contains: All trained model files (.joblib, .pkl)

**To new device:**
- Copy entire `models/` folder
- Paste to: `backend/models/`

**What's inside:**
```
backend/models/
├── aca_rf_model.joblib
├── aca_rf_model.pkl
├── aca_scaler.joblib
├── aca_svm_model.joblib
├── fuzzy_rf_model.joblib
├── fuzzy_rf_scaler.joblib
├── fuzzy_rf_ensemble_model.joblib
├── intrudtree_model.joblib
├── intrudtree/
│   ├── binary_model.joblib
│   ├── multiclass_model.joblib
│   ├── scaler.joblib
│   ├── scaler_binary.joblib
│   ├── scaler_malicious.joblib
│   └── selected_features.joblib
├── orchestrator_state.joblib
├── random_forest.joblib
├── scaler.joblib
└── feature_names.txt
```

#### **3.2 Copy Processed Data Folder** (~95 MB)

**From main device:**
- Location: `backend/data/processed/`
- Contains: Processed training data and preprocessing objects

**To new device:**
- Copy entire `processed/` folder
- Paste to: `backend/data/processed/`

**What's inside:**
```
backend/data/processed/
├── X_train.csv (~61 MB)
├── X_test.csv (~23 MB)
├── y_train.csv (~0.23 MB)
├── y_test.csv (~0.09 MB)
├── metadata.csv (~10 MB)
├── scaler.joblib
├── imputer.joblib
├── label_encoder.joblib
├── class_weights.joblib
├── preprocessing_info.json
└── feature_names.txt
```

### 📊 **Optional: Copy CSV Dataset Files** (~327-392 MB)

**Only needed if you want to retrain models on the new device.**

**From main device:**
- Location: `backend/data/consolidated/`
- Files: `All_Network_Sample_Complete.csv`, `All_Network_Sample_Complete_100k.csv`

**To new device:**
- Upload to Google Drive / OneDrive / Dropbox
- Download on new device
- Place in: `backend/data/consolidated/`

**Note:** If you copied the models folder, you DON'T need the CSV files to run the system.

---

## 🔄 Transfer Methods

### **Method 1: USB Drive / External Hard Drive** (Recommended for large files)

1. On main device: Copy `backend/models/` and `backend/data/processed/` to USB
2. Connect USB to new device
3. Copy folders to correct locations in the cloned repository

### **Method 2: Cloud Storage** (Google Drive, OneDrive, Dropbox)

1. **On main device:**
   - Compress folders: Create ZIP files (`models.zip`, `processed.zip`)
   - Upload to cloud storage

2. **On new device:**
   - Download ZIP files
   - Extract to correct locations:
     - `models.zip` → Extract to `backend/models/`
     - `processed.zip` → Extract to `backend/data/processed/`

### **Method 3: Network Transfer** (If both devices on same network)

- Use shared network folder
- Use SCP/SFTP for secure transfer

### **Helper Script (Windows PowerShell)**

If you're on Windows, you can use the provided script to create ZIP files:

```powershell
.\package_for_transfer.ps1
```

This creates ready-to-upload ZIP files in `transfer_package/` folder.

---

## ✅ Step 4: Verify Installation

Before running, verify all required files exist:

### **Check Models:**
```bash
# From project root
ls backend/models/aca_rf_model.joblib
ls backend/models/fuzzy_rf_model.joblib
ls backend/models/intrudtree_model.joblib
ls backend/models/intrudtree/
```

### **Check Processed Data:**
```bash
# From project root
ls backend/data/processed/X_train.csv
ls backend/data/processed/X_test.csv
ls backend/data/processed/y_train.csv
ls backend/data/processed/y_test.csv
ls backend/data/processed/scaler.joblib
ls backend/data/processed/imputer.joblib
```

### **Quick Verification Checklist:**

- [ ] `backend/models/` folder exists with model files
- [ ] `backend/models/intrudtree/` subfolder exists
- [ ] `backend/data/processed/X_train.csv` exists
- [ ] `backend/data/processed/X_test.csv` exists
- [ ] `backend/data/processed/y_train.csv` exists
- [ ] `backend/data/processed/y_test.csv` exists
- [ ] `backend/data/processed/scaler.joblib` exists
- [ ] `backend/data/processed/imputer.joblib` exists
- [ ] `backend/data/processed/label_encoder.joblib` exists

---

## 🚀 Step 5: Run the System

### **Option A: Run Everything Together (Easiest)**

From project root:

```bash
python start_system.py
```

This starts both backend and frontend automatically.

### **Option B: Run Separately (For debugging)**

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows (or source venv/bin/activate on Linux/Mac)
python app_simple.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### **Access the System:**

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **API Status:** http://localhost:5000/api/status

---

## 🆘 Troubleshooting

### **Problem: "Models not found" error**

**Solution:**
1. Verify `backend/models/` folder exists
2. Check that model files have correct extensions (.joblib or .pkl)
3. Ensure folder structure is correct
4. Check file permissions (files shouldn't be read-only)

### **Problem: "Processed data not found" error**

**Solution:**
1. Verify `backend/data/processed/` folder exists
2. Check all required CSV files are present:
   - X_train.csv
   - X_test.csv
   - y_train.csv
   - y_test.csv
3. Verify preprocessing objects exist:
   - scaler.joblib
   - imputer.joblib
   - label_encoder.joblib

### **Problem: Import errors or missing modules**

**Solution:**
```bash
# Backend
cd backend
venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### **Problem: Port already in use**

**Solution:**
- Backend (port 5000): Change port in `backend/app_simple.py`
- Frontend (port 3000): Change port in `frontend/package.json` or use `PORT=3001 npm start`

### **Problem: System runs but shows errors in browser**

**Solution:**
1. Check browser console for errors (F12)
2. Verify backend is running: http://localhost:5000/api/status
3. Check CORS settings in `backend/app_simple.py`
4. Verify all API endpoints are accessible

### **Problem: Files corrupted during transfer**

**Solution:**
1. Re-copy files from main device
2. Verify file sizes match original files
3. Use ZIP files for transfer (better integrity)
4. Check file checksums if possible

---

## 📊 File Size Summary

| Item | Size | Required? |
|------|------|-----------|
| Repository (cloned) | ~50-100 MB | ✅ Yes |
| Models folder | ~834 MB | ✅ **YES** |
| Processed data folder | ~95 MB | ✅ **YES** |
| CSV dataset files | ~327-392 MB | ❌ Optional |
| **Total (minimum)** | **~980 MB** | |
| **Total (with CSV)** | **~1.3 GB** | |

---

## 📝 Quick Reference

### **Essential Commands:**

```bash
# Clone repository
git clone https://github.com/lance449/cyber-threat-v3.git
cd cyber-threat-v3

# Setup backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install

# Run system
cd ..
python start_system.py
```

### **File Locations:**

```
cyber-threat-v3/
├── backend/
│   ├── models/              ← COPY THIS FOLDER (~834 MB)
│   ├── data/
│   │   ├── processed/       ← COPY THIS FOLDER (~95 MB)
│   │   └── consolidated/    ← Optional: CSV files here
│   └── app_simple.py
├── frontend/
│   └── src/
└── start_system.py
```

---

## ✅ Success Checklist

After completing all steps, you should be able to:

- [ ] Access frontend at http://localhost:3000
- [ ] See backend API running at http://localhost:5000
- [ ] API status endpoint returns success: http://localhost:5000/api/status
- [ ] No errors in terminal/console
- [ ] System can detect threats and show results

---

## 🎉 You're Done!

If all steps completed successfully, your system is ready to use! 

**Next Steps:**
- Test threat detection functionality
- Explore the dashboard features
- Review detection reports

**Need Help?**
- Check the troubleshooting section above
- Review error messages carefully
- Verify all files are in correct locations

---

## 📚 Additional Resources

- **Repository:** https://github.com/lance449/cyber-threat-v3.git
- **Detailed File Guide:** See `FILES_TO_COPY_GUIDE.md`
- **Setup Instructions:** See `SETUP_INSTRUCTIONS.md`

---

**Last Updated:** 2025-01-27
**Repository Version:** v3.0

