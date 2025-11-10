# 📦 Files to Copy Guide

This guide tells you exactly what folders/files to copy from this device to another device to run the system smoothly without errors.

## 🎯 Quick Answer: What to Copy

### **Essential Folders (REQUIRED for system to run):**

1. **`backend/models/`** - All trained model files
2. **`backend/data/processed/`** - All preprocessing files and processed data

### **Optional (if you want to retrain models):**

3. **`backend/data/consolidated/*.csv`** - Large CSV data files (upload to online drive)

---

## 📋 Detailed Copy Instructions

### **Step 1: Copy Model Files**

**Location on this device:**
```
backend/models/
```

**What's inside:**
- `aca_rf_model.joblib` (or `.pkl`)
- `aca_scaler.joblib`
- `aca_svm_model.joblib`
- `fuzzy_rf_model.joblib`
- `fuzzy_rf_scaler.joblib`
- `fuzzy_rf_ensemble_model.joblib`
- `intrudtree_model.joblib`
- `intrudtree/` folder (contains binary_model.joblib, multiclass_model.joblib, scalers, etc.)
- `scaler.joblib`
- `random_forest.joblib`
- `orchestrator_state.joblib`
- `feature_names.txt`

**Copy entire folder:**
- Copy the entire `backend/models/` folder
- Paste it to `backend/models/` on the new device

---

### **Step 2: Copy Processed Data Files**

**Location on this device:**
```
backend/data/processed/
```

**What's inside:**
- `X_train.csv` (~61 MB)
- `X_test.csv` (~23 MB)
- `y_train.csv` (~0.23 MB)
- `y_test.csv` (~0.09 MB)
- `metadata.csv` (~10 MB)
- `scaler.joblib`
- `imputer.joblib`
- `label_encoder.joblib`
- `class_weights.joblib`
- `preprocessing_info.json`
- `feature_names.txt`

**Copy entire folder:**
- Copy the entire `backend/data/processed/` folder
- Paste it to `backend/data/processed/` on the new device

---

### **Step 3: Copy CSV Files (Optional - for retraining)**

**Location on this device:**
```
backend/data/consolidated/
```

**What's inside:**
- `All_Network_Sample_Complete.csv` (~327 MB)
- `All_Network_Sample_Complete_100k.csv` (~65 MB)

**Upload to online drive:**
- Upload these CSV files to Google Drive, OneDrive, Dropbox, etc.
- Download them on the new device
- Place them in `backend/data/consolidated/` on the new device

**Note:** You only need these if you want to retrain models. If you copy the models folder, you don't need the CSV files.

---

## 🚀 Setup on New Device

### **After copying files:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lance449/cyber-threat-v3.git
   cd cyber-threat-v3
   ```

2. **Copy the folders you prepared:**
   - Copy `models/` → `backend/models/`
   - Copy `processed/` → `backend/data/processed/`
   - (Optional) Copy CSV files → `backend/data/consolidated/`

3. **Install dependencies:**
   ```bash
   # Backend
   cd backend
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

4. **Run the system:**
   ```bash
   # From project root
   python start_system.py
   ```

---

## ✅ Verification Checklist

Before running on the new device, verify these files exist:

### **Models (backend/models/):**
- [ ] `aca_rf_model.joblib` or `aca_rf_model.pkl`
- [ ] `fuzzy_rf_model.joblib`
- [ ] `intrudtree_model.joblib`
- [ ] `intrudtree/` folder with sub-models

### **Processed Data (backend/data/processed/):**
- [ ] `X_train.csv`
- [ ] `X_test.csv`
- [ ] `y_train.csv`
- [ ] `y_test.csv`
- [ ] `metadata.csv`
- [ ] `scaler.joblib`
- [ ] `imputer.joblib`
- [ ] `label_encoder.joblib`

---

## 📊 File Sizes (Actual)

| Folder/File | Size | Files | Required? |
|------------|------|-------|-----------|
| `backend/models/` | ~834 MB | 18 files | ✅ **YES** |
| `backend/data/processed/` | ~95 MB | 11 files | ✅ **YES** |
| `backend/data/consolidated/*.csv` | ~327-392 MB | 2 files | ❌ Optional |

**Total required:** ~930 MB (without CSV files)
**Total with CSV:** ~1.3 GB

---

## 🔄 Transfer Methods

### **Method 1: USB Drive / External Hard Drive**
1. Copy `backend/models/` and `backend/data/processed/` to USB
2. Transfer to new device
3. Copy to correct locations

### **Method 2: Cloud Storage (Recommended)**
1. **Compress folders:**
   - Create ZIP files: `models.zip` and `processed.zip`
   - Upload to Google Drive / OneDrive / Dropbox

2. **Download on new device:**
   - Download ZIP files
   - Extract to correct locations

### **Method 3: Network Transfer**
- Use shared network folder
- Use SCP/SFTP if both devices are on same network

---

## ⚠️ Important Notes

1. **Don't modify file structure** - Keep folder names and paths exactly the same
2. **Preserve file permissions** - Make sure files aren't read-only on new device
3. **Check file integrity** - Verify all files copied successfully (check file sizes)
4. **CSV files are optional** - Only needed if you want to retrain models

---

## 🆘 Troubleshooting

### **If system says "models not found":**
- Check that `backend/models/` folder exists
- Verify model files have correct extensions (.joblib or .pkl)
- Check file paths are correct

### **If system says "processed data not found":**
- Check that `backend/data/processed/` folder exists
- Verify all CSV files (X_train.csv, X_test.csv, etc.) are present
- Check preprocessing objects (scaler.joblib, imputer.joblib) exist

### **If you get errors:**
- Make sure you copied the ENTIRE folders, not just some files
- Verify folder structure matches exactly
- Check that files aren't corrupted during transfer

---

## 📝 Summary

**Minimum required to run system:**
1. ✅ `backend/models/` folder
2. ✅ `backend/data/processed/` folder

**Optional (for retraining):**
3. ⭕ `backend/data/consolidated/*.csv` files (upload to online drive)

That's it! With just these 2 folders, your system will run smoothly on the new device! 🎉

