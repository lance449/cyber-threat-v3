# Cyber Threat Detection System v2.0

A comprehensive, modular cyber threat detection system that compares three advanced machine learning models for network traffic analysis using the BCCC-Mal-NetMem-2025 dataset.

## 🎯 System Overview

This system processes network traffic data and compares the detection performance of three specialized models:

1. **ACA + Random Forest (ACA_RF)**: Signature-based pattern detection using Aho-Corasick algorithm
2. **Fuzzy Logic + Random Forest (Fuzzy_RF)**: Behavior-based detection using fuzzy inference
3. **IntruDTree**: Interpretable decision-tree model tailored to cybersecurity

## 🏗️ System Architecture

```
CSP114/
├── backend/
│   ├── src/
│   │   ├── preprocessing/
│   │   │   └── data_preprocessor.py      # Unified data preprocessing
│   │   ├── models/
│   │   │   ├── aca_rf_model.py          # Model 1: ACA + RF
│   │   │   ├── fuzzy_rf_model.py        # Model 2: Fuzzy + RF
│   │   │   └── intrudtree_model.py      # Model 3: IntruDTree
│   │   ├── risk_analysis/
│   │   │   └── threat_analyzer.py       # Threat severity & risk analysis
│   │   ├── reporting/
│   │   │   └── comparison_reporter.py   # Model comparison & reporting
│   │   └── detection_orchestrator.py    # Main orchestrator
│   ├── app_new.py                       # New Flask API
│   ├── train_system.py                  # Training script
│   └── requirements.txt                 # Dependencies
├── frontend/                            # React.js frontend
└── data/
    └── raw/
        └── All_Network.csv              # BCCC-Mal-NetMem-2025 dataset
```

## 🚀 Key Features

### 🔍 **Unified Data Preprocessing**
- Handles the All_Network.csv dataset from BCCC-Mal-NetMem-2025
- Comprehensive feature engineering for cybersecurity
- Automatic handling of missing values, scaling, and encoding
- Compatible with all three models

### 🎯 **Model 1: ACA + Random Forest**
- **Purpose**: Signature-based pattern detection
- **Process**: 
  - Uses Aho-Corasick algorithm to match known malware patterns
  - Extracts pattern-based features from network flows
  - Combines with Random Forest for final classification
- **Output**: Detected threat label + ACA detection confirmation + severity + risk level

### 🧠 **Model 2: Fuzzy Logic + Random Forest**
- **Purpose**: Behavior-based detection using fuzzy inference
- **Process**:
  - Defines fuzzy rules for network behavior metrics
  - Computes fuzzy scores for threat likelihood
  - Adds fuzzy outputs as new features for Random Forest
- **Output**: Detected threat label + fuzzy score + severity + risk level

### 🌳 **Model 3: IntruDTree**
- **Purpose**: Interpretable decision-tree model for cybersecurity
- **Process**:
  - Specialized decision tree induction algorithm
  - Optimized for malware behavior indicators
  - Provides explainable detection paths
- **Output**: Detected threat label + branch path + severity + risk level

### 🛡️ **Threat Severity & Risk Analysis**
- **Severity Assignment**: Low, Medium, High based on threat type and behavior
- **Risk Analysis**: Calculates risk scores using multiple factors
- **Action Recommendations**: Provides actionable insights for cybersecurity professionals

### 📊 **Comparison & Reporting Module**
- **Side-by-side Model Comparison**: Accuracy, Precision, Recall, F1-score
- **Performance Metrics**: Detection time, false positives/negatives
- **Visual Reports**: Bar charts, performance graphs
- **Export Options**: PDF and CSV reports

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+ (for frontend)
- 8GB+ RAM (recommended for large dataset processing)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/lance449/cyber-threat-detection-using-machine-learning.git
cd cyber-threat-detection-using-machine-learning
```

2. **Create and activate virtual environment (recommended)**
```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare the dataset**
```bash
# Ensure All_Network.csv is in data/raw/
ls data/raw/All_Network.csv
```

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend
npm install
```

## 🚀 Usage

### Quick Start - Complete Pipeline

Run the complete training and evaluation pipeline:

```bash
cd backend
python train_system.py
```

This will:
1. Initialize the system
2. Preprocess the All_Network.csv dataset
3. Train all three models
4. Evaluate performance
5. Perform threat analysis
6. Compare models
7. Generate comprehensive reports

### API Usage

Start the Flask API:

```bash
cd backend
python app_new.py
```

The API provides the following endpoints:

- `GET /api/status` - System status
- `POST /api/initialize` - Initialize system
- `POST /api/preprocess` - Preprocess data
- `POST /api/train` - Train models
- `POST /api/evaluate` - Evaluate models
- `POST /api/compare` - Compare models
- `POST /api/detect` - Detect threats
- `POST /api/export-report` - Export reports
- `GET /api/results` - Get current results

### Frontend Usage

Start the React frontend:

```bash
cd frontend
npm start
```

Access the web interface at `http://localhost:3000`

## 📈 Model Performance

The system provides comprehensive performance metrics for each model:

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Detection Time**: Time taken for threat detection
- **False Positive Rate**: Rate of false alarms
- **False Negative Rate**: Rate of missed threats

### Risk Analysis
- **Risk Score**: 0-1 scale based on multiple factors
- **Risk Level**: Low, Medium, High, Critical
- **Impact Analysis**: Operational, Data, Financial, Reputation impacts
- **Action Recommendations**: Immediate, short-term, and long-term actions

## 📊 Reports & Visualizations

### Generated Reports
1. **Executive Summary**: High-level findings and recommendations
2. **Performance Comparison**: Side-by-side model metrics
3. **Statistical Analysis**: Detailed statistical comparisons
4. **Threat Analysis**: Risk assessment and severity analysis
5. **Technical Details**: Model specifications and configurations

### Export Formats
- **PDF Reports**: Professional formatted reports with charts
- **CSV Data**: Raw data for further analysis
- **JSON API**: Programmatic access to results

## 🔧 Configuration

### Model Configuration
```python
config = {
    'data_path': 'data/raw/All_Network.csv',
    'processed_data_path': 'data/processed',
    'models_path': 'models',
    'test_size': 0.3,
    'random_state': 42,
    'scaler_type': 'minmax',
    'model_configs': {
        'aca_rf': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'fuzzy_rf': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'intrudtree': {
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    }
}
```

## 📁 File Structure

### Backend Structure
```
backend/
├── src/
│   ├── preprocessing/
│   │   └── data_preprocessor.py      # Data preprocessing module
│   ├── models/
│   │   ├── aca_rf_model.py          # ACA + RF model
│   │   ├── fuzzy_rf_model.py        # Fuzzy + RF model
│   │   └── intrudtree_model.py      # IntruDTree model
│   ├── risk_analysis/
│   │   └── threat_analyzer.py       # Risk analysis module
│   ├── reporting/
│   │   └── comparison_reporter.py   # Reporting module
│   └── detection_orchestrator.py    # Main orchestrator
├── data/
│   ├── raw/
│   │   └── All_Network.csv          # Input dataset
│   └── processed/                   # Processed data files
├── models/                          # Trained model files
├── logs/                           # Training and system logs
├── reports/                        # Generated reports
├── app_new.py                      # Flask API
├── train_system.py                 # Training script
└── requirements.txt                # Python dependencies
```

## 🧪 Testing

### Unit Tests
```bash
cd backend
python -m pytest tests/
```

### Integration Tests
```bash
# Test complete pipeline
python train_system.py

# Test API endpoints
python -m pytest tests/test_api.py
```

## 📝 Logging

The system provides comprehensive logging:

- **Training Logs**: `logs/training.log`
- **API Logs**: Console and file logging
- **Error Tracking**: Detailed error messages and stack traces
- **Performance Metrics**: Model training and evaluation metrics

## 🔒 Security Considerations

- **Data Privacy**: All data processing is done locally
- **Model Security**: Trained models are saved securely
- **API Security**: CORS enabled for frontend integration
- **Input Validation**: Comprehensive input validation and sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BCCC-Mal-NetMem-2025 Dataset**: For providing the network traffic data
- **Scikit-learn**: For machine learning algorithms
- **Flask**: For the web API framework
- **React**: For the frontend framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for troubleshooting

## 🔄 Version History

- **v2.0.0**: Complete rebuild with modular architecture
- **v1.0.0**: Initial implementation

---

**Note**: This system is designed for research and educational purposes. For production use, additional security measures and validation should be implemented.
