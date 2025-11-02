#!/usr/bin/env python3
"""
Simplified Flask API Server for Cyber Threat Detection System
Provides endpoints for the React frontend
"""

import os
import sys
import logging
import traceback
import time
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the orchestrator
from detection_orchestrator import CyberThreatDetectionOrchestrator

app = Flask(__name__)
CORS(app)

# Global orchestrator instance
orchestrator = None

# In-memory stores for latest detection and manual verification
recent_threats = {}
manual_verifications = {}

def initialize_orchestrator():
    """Initialize the orchestrator with trained models"""
    global orchestrator
    
    dataset_path = os.environ.get('DATASET_PATH', 'data/consolidated/All_Network_Sample_Complete_100k.csv')
    config = {
        'data_path': dataset_path,  # Allow override via DATASET_PATH
        'processed_data_path': 'data/processed',
        'models_path': 'models',
        'test_size': 0.3,
        'random_state': 42,
        'scaler_type': 'minmax',
        'model_configs': {
            'fuzzy_rf': {
                'n_estimators': 200,
                'max_depth': 30,  # Increased from 25 to 30 for deeper trees
                'min_samples_split': 2,  # Reduced from 5 to 2 - more aggressive splits
                'min_samples_leaf': 1,  # Reduced from 2 to 1 - more aggressive splits
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'max_samples': 0.8,
                'min_impurity_decrease': 0.0,  # Reduced from 0.001 - allow more splits
                'criterion': 'gini',
                'ccp_alpha': 0.0,  # Reduced from 0.001 - less pruning
                'n_jobs': -1
            },
            'intrudtree': {
                'max_depth': 50,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'prune_threshold': 0.001,
                'n_important_features': 79
            },
            'aca_svm': {
                'svm_type': 'linear',
                'kernel': 'linear',
                'C': 5.0,
                'gamma': 'scale',
                'class_weight': 'balanced',
                'max_iter': 5000,
                'tol': 1e-4,
                'patterns_file': 'data/patterns/cyber_threat_patterns.txt'
            }
        }
    }
    
    try:
        orchestrator = CyberThreatDetectionOrchestrator(config)
        init_result = orchestrator.initialize_system()
        if not init_result['success']:
            logger.error(f"Failed to initialize orchestrator: {init_result['error']}")
            return False
        
        # Ensure processed data is present; prefer existing artifacts to avoid reprocessing full dataset
        processed_dir = orchestrator.config['processed_data_path']
        x_train = os.path.join(processed_dir, 'X_train.csv')
        x_test = os.path.join(processed_dir, 'X_test.csv')
        y_train = os.path.join(processed_dir, 'y_train.csv')
        y_test = os.path.join(processed_dir, 'y_test.csv')
        metadata = os.path.join(processed_dir, 'metadata.csv')

        if all(os.path.exists(p) for p in [x_train, x_test, y_train, y_test, metadata]):
            orchestrator.processed_data = {
                'X_train_path': x_train,
                'X_test_path': x_test,
                'y_train_path': y_train,
                'y_test_path': y_test,
                'metadata_path': metadata
            }
            load_result = orchestrator.load_processed_data()
            if not load_result['success']:
                logger.error(f"Failed to load existing processed data: {load_result['error']}")
                return False
        else:
            logger.warning("Processed data not found. Running preprocessing once on a sample for responsiveness...")
            # Use a manageable sample size to avoid OOM on startup
            orchestrator.config['sample_size'] = 100000
            pre = orchestrator.preprocess_data()
            if not pre['success']:
                logger.error(f"Preprocessing failed: {pre['error']}")
                return False
        load_result = orchestrator.load_processed_data()
        if not load_result['success']:
            logger.error(f"Failed to load processed data after preprocessing: {load_result['error']}")
            return False

        # Load models from disk if available, else train and save
        try:
            orchestrator.load_models()
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Training models now...")
            train_res = orchestrator.train_all_models()
            if not train_res['success']:
                logger.error(f"Training failed: {train_res['error']}")
                return False
            save_res = orchestrator.save_models()
            if not save_res['success']:
                logger.error(f"Saving models failed: {save_res['error']}")
                return False
        
        logger.info("Orchestrator initialized successfully with real data and models loaded")
        # Load trained models if they exist
        load_result = orchestrator.load_models()
        if load_result['success']:
            logger.info("✓ Trained models loaded successfully")
        else:
            logger.warning(f"⚠ Could not load trained models: {load_result.get('error', 'Unknown error')}")
            logger.info("Models will need to be trained before making predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing orchestrator: {str(e)}")
        return False

def check_required_files():
    """Check if all required model and data files exist"""
    required_files = [
        ('data/processed/X_train.csv', 'Training features'),
        ('data/processed/X_test.csv', 'Test features'),
        ('data/processed/y_train.csv', 'Training labels'),
        ('data/processed/y_test.csv', 'Test labels'),
        ('models/aca_rf_model.joblib', 'ACA RF model'),
        ('models/fuzzy_rf_model.joblib', 'Fuzzy RF model'),
        ('models/intrudtree_model.joblib', 'IntruDTree model')
    ]
    
    missing_files = []
    for file_path, description in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not os.path.exists(full_path):
            missing_files.append(f"{description} at {file_path}")
    
    return missing_files

@app.route('/api/status', methods=['GET'])
def status():
    """Check if the backend is running and models are available"""
    try:
        missing_files = check_required_files()
        
        # Check if we have real data or need to use simulation
        has_real_data = (orchestrator is not None and 
                        hasattr(orchestrator, 'X_test') and 
                        orchestrator.X_test is not None)
        
        ready = len(missing_files) == 0 and has_real_data
        
        return jsonify({
            "success": True,
            "status": "running",
            "missing_files": missing_files,
            "ready": ready,
            "models_loaded": orchestrator is not None,
            "data_mode": "real" if has_real_data else "simulated",
            "message": "System ready with real data" if has_real_data else "System running with simulated data for demonstration"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/detect', methods=['POST'])
def detect():
    """Perform threat detection using trained models or simulated data"""
    try:
        # Enforce real detection only
        if orchestrator is None or not hasattr(orchestrator, 'loaded_data'):
            return jsonify({
                "success": False,
                "error": "Orchestrator not ready with real data. Please run training/preprocessing first."
            }), 500

        # Use real data and models
        X_test = orchestrator.loaded_data['X_test']
        y_test = orchestrator.loaded_data['y_test']
        
        # Allow client to control how many to return (default 5 to show more flows)
        try:
            req_count = int(request.args.get('count', '5'))
        except Exception:
            req_count = 5
        sample_size = max(1, min(req_count, len(X_test)))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]

        start_time = time.time()
        detection_results = {}
        
        # Load label encoder for proper class names
        label_encoder = None
        try:
            label_encoder_path = os.path.join(orchestrator.config['processed_data_path'], 'label_encoder.joblib')
            if os.path.exists(label_encoder_path):
                import joblib
                label_encoder = joblib.load(label_encoder_path)
                logger.info(f"Loaded label encoder with classes: {label_encoder.classes_}")
        except Exception as e:
            logger.warning(f"Could not load label encoder: {e}")
        
        # Determine benign label index
        benign_idx = 0
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            if 'Benign' in list(label_encoder.classes_):
                benign_idx = list(label_encoder.classes_).index('Benign')
                logger.info(f"Benign label index: {benign_idx}")

        # Load metadata properly
        metadata_path = os.path.join(orchestrator.config['processed_data_path'], 'metadata.csv')
        meta_df = None
        if os.path.exists(metadata_path):
            try:
                meta_df = pd.read_csv(metadata_path)
                logger.info(f"Loaded metadata with {len(meta_df)} rows and columns: {list(meta_df.columns)}")
                # Ensure metadata has the same number of rows as test data
                if len(meta_df) != len(X_test):
                    logger.warning(f"Metadata rows ({len(meta_df)}) don't match test data ({len(X_test)})")
                    meta_df = None
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                meta_df = None

        for model_name, model in orchestrator.models.items():
            try:
                predictions, details = model.predict(X_sample)
                accuracy = float(np.mean(predictions == y_sample.values))
                threats_detected = int(np.sum(predictions != benign_idx))
                actual_threats = int(np.sum(y_sample.values != benign_idx))
                detection_results[model_name] = {
                    'predictions': predictions.tolist(),
                    'accuracy': accuracy,
                    'threats_detected': threats_detected,
                    'actual_threats': actual_threats,
                    'details': details
                }
            except Exception as e:
                detection_results[model_name] = {'error': str(e)}

        detection_time = time.time() - start_time
        
        # Generate individual threat records for the frontend
        threat_records = []
        timestamp = datetime.now().isoformat()
        
        # Build flow records from sample using model predictions
        # Try to align metadata if dimensions match the test set
        meta_src = None
        if 'metadata' in orchestrator.loaded_data:
            meta_df = orchestrator.loaded_data['metadata']
            if len(meta_df) == len(X_test):
                meta_src = meta_df.iloc[sample_indices]

        for i in range(sample_size):
            # Get metadata for this sample if available
            src_ip = "N/A"
            dst_ip = "N/A"
            if meta_df is not None and i < len(meta_df):
                try:
                    # Look for common IP column names
                    src_cols = ['Src IP', 'src_ip', 'Source IP', 'source_ip', 'srcip', 'sourceip']
                    dst_cols = ['Dst IP', 'dst_ip', 'Destination IP', 'destination_ip', 'dstip', 'destinationip']
                    
                    for col in src_cols:
                        if col in meta_df.columns:
                            src_ip = str(meta_df.iloc[sample_indices[i]][col])
                            break
                    
                    for col in dst_cols:
                        if col in meta_df.columns:
                            dst_ip = str(meta_df.iloc[sample_indices[i]][col])
                            break
                except Exception as e:
                    logger.warning(f"Failed to extract IPs from metadata row {i}: {e}")

            # Get per-model predictions - ensure we get the correct prediction for this specific sample
            aca_pred = detection_results.get('ACA_SVM', {}).get('predictions', [])
            fuzzy_pred = detection_results.get('Fuzzy_RF', {}).get('predictions', [])
            intrudtree_pred = detection_results.get('IntruDTree', {}).get('predictions', [])
            
            # Get the prediction for this specific sample index
            if len(aca_pred) > i:
                aca_binary = 1 if aca_pred[i] != benign_idx else 0
            else:
                # If no ACA predictions available, default to benign
                aca_binary = 0
            
            if len(fuzzy_pred) > i:
                fuzzy_binary = 1 if fuzzy_pred[i] != benign_idx else 0
            else:
                fuzzy_binary = 0
                
            if len(intrudtree_pred) > i:
                intrudtree_binary = 1 if intrudtree_pred[i] != benign_idx else 0
            else:
                intrudtree_binary = 0
            
            # Majority vote for ensemble
            votes = [aca_binary, fuzzy_binary, intrudtree_binary]
            ensemble_binary = 1 if sum(votes) >= 2 else 0
            
            # Get human-readable attack label from true label
            true_label_idx = y_sample.values[i]
            if label_encoder is not None and hasattr(label_encoder, 'classes_') and true_label_idx < len(label_encoder.classes_):
                attack_label = label_encoder.classes_[true_label_idx]
            else:
                # Map common attack types to readable names
                attack_mapping = {
                    0: 'Benign',
                    1: 'DoS',
                    2: 'DDoS', 
                    3: 'Port Scan',
                    4: 'Brute Force',
                    5: 'Infiltration',
                    6: 'Botnet',
                    7: 'Web Attack',
                    8: 'Exploit',
                    9: 'Backdoor',
                    10: 'Rootkit',
                    11: 'Trojan',
                    12: 'Virus',
                    13: 'Worm',
                    14: 'HackTool',
                    15: 'Hoax'
                }
                attack_label = attack_mapping.get(int(true_label_idx), f"Attack_{true_label_idx}")

            # Enhanced severity mapping based on attack type and model agreement
            severity = "Low"
            
            # High severity attacks (always high if detected by any model)
            high_severity_attacks = ['DoS', 'DDoS', 'Exploit', 'Rootkit', 'Trojan', 'Virus', 'Worm', 'Backdoor']
            medium_severity_attacks = ['Port Scan', 'HackTool', 'Hoax', 'Botnet', 'Infiltration', 'Web Attack']
            
            if attack_label in high_severity_attacks:
                if sum(votes) >= 2:  # Majority agreement
                    severity = "High"
                elif sum(votes) == 1:  # Single model detection
                    severity = "High"  # Still high for serious attacks
                else:
                    severity = "Medium"  # Even if no models detect, it's still concerning
            elif attack_label in medium_severity_attacks:
                if sum(votes) >= 2:
                    severity = "Medium"
                elif sum(votes) == 1:
                    severity = "Medium"
                else:
                    severity = "Low"
            elif attack_label == "Benign":
                severity = "Low"
            else:
                # Unknown attack types
                if sum(votes) >= 2:
                    severity = "Medium"
                elif sum(votes) == 1:
                    severity = "Low"
                else:
                    severity = "Low"

            # Determine risk level based on ensemble prediction
            if ensemble_binary == 1:
                risk_level = "High"
            elif sum(votes) >= 2:
                risk_level = "Medium"
            elif sum(votes) == 1:
                risk_level = "Low"
            else:
                risk_level = "None"

            # Get detection models that flagged this as threat
            detection_models = []
            if aca_binary == 1:
                detection_models.append("ACA_SVM")
            if fuzzy_binary == 1:
                detection_models.append("Fuzzy_RF")
            if intrudtree_binary == 1:
                detection_models.append("IntruDTree")
            
            # Debug logging for model predictions (only log occasionally to avoid spam)
            if i % 100 == 0:  # Log every 100th sample
                logger.info(f"Sample {i} - ACA: {aca_binary}, Fuzzy: {fuzzy_binary}, IntruDTree: {intrudtree_binary}, Benign_idx: {benign_idx}")
            
            # Enhanced risk analysis with model information
            threat_votes = sum(votes)
            total_models = len(votes)
            
            if attack_label == "Benign":
                risk_analysis = "🟢 Low risk: Normal traffic confirmed"
            elif threat_votes >= 2:
                # Majority agreement
                if severity == "High":
                    risk_analysis = f"🔥 High risk: {attack_label} confirmed by {threat_votes}/{total_models} models - immediate action required"
                elif severity == "Medium":
                    risk_analysis = f"⚠️ Medium risk: {attack_label} confirmed by {threat_votes}/{total_models} models - monitor closely"
                else:
                    risk_analysis = f"🟡 Low risk: {attack_label} confirmed by {threat_votes}/{total_models} models"
            elif threat_votes == 1:
                # Single model detection
                if severity == "High":
                    risk_analysis = f"🔥 High risk: {attack_label} detected by 1/{total_models} models - investigate immediately"
                elif severity == "Medium":
                    risk_analysis = f"⚠️ Medium risk: {attack_label} detected by 1/{total_models} models - investigate"
                else:
                    risk_analysis = f"🟡 Low risk: {attack_label} detected by 1/{total_models} models"
            else:
                # No model detection but attack type suggests threat
                if attack_label in high_severity_attacks:
                    risk_analysis = f"⚠️ Medium risk: {attack_label} detected (ground truth) - manual review recommended"
                else:
                    risk_analysis = f"🟡 Low risk: {attack_label} detected (ground truth) - routine monitoring"
            
            # Add model details
            if len(detection_models) > 0:
                risk_analysis += f" | Detected by: {', '.join(detection_models)}"
            
            # Build model_used string
            model_used = ", ".join(detection_models) if detection_models else "Unknown"
            
            # Helper function to convert numeric label to string attack label
            def get_attack_label_from_idx(label_idx):
                if label_encoder is not None and hasattr(label_encoder, 'classes_') and label_idx < len(label_encoder.classes_):
                    return label_encoder.classes_[label_idx]
                else:
                    attack_mapping = {
                        0: 'Benign', 1: 'DoS', 2: 'DDoS', 3: 'Brute Force', 4: 'Infiltration',
                        5: 'Botnet', 6: 'Web Attack', 7: 'Exploit', 8: 'Backdoor',
                        9: 'Rootkit', 10: 'Trojan', 11: 'Virus', 12: 'Worm',
                        13: 'HackTool', 14: 'Hoax'
                    }
                    return attack_mapping.get(int(label_idx), f"Attack_{label_idx}")
            
            # Store individual model predictions for accuracy calculation
            aca_label_idx = aca_pred[i] if len(aca_pred) > i else benign_idx
            fuzzy_label_idx = fuzzy_pred[i] if len(fuzzy_pred) > i else benign_idx
            intrudtree_label_idx = intrudtree_pred[i] if len(intrudtree_pred) > i else benign_idx
            
            model_predictions = {
                'ACA_SVM': {
                    'predicted_threat': aca_binary == 1,
                    'predicted_label': aca_label_idx,
                    'attack_label': get_attack_label_from_idx(aca_label_idx),  # Add attack label string
                    'is_correct': None  # Will be calculated based on true_label
                },
                'Fuzzy_RF': {
                    'predicted_threat': fuzzy_binary == 1,
                    'predicted_label': fuzzy_label_idx,
                    'attack_label': get_attack_label_from_idx(fuzzy_label_idx),  # Add attack label string
                    'is_correct': None
                },
                'IntruDTree': {
                    'predicted_threat': intrudtree_binary == 1,
                    'predicted_label': intrudtree_label_idx,
                    'attack_label': get_attack_label_from_idx(intrudtree_label_idx),  # Add attack label string
                    'is_correct': None
                }
            }
            
            # Calculate if each prediction was correct
            true_is_threat = (attack_label != "Benign")
            for model_name in model_predictions:
                predicted_is_threat = model_predictions[model_name]['predicted_threat']
                model_predictions[model_name]['is_correct'] = (predicted_is_threat == true_is_threat)
            
            # Display all flows (threats and benign) - every model always makes a prediction
            unique_flow_id = f"flow_{int(time.time()*1000)}_{i+1:04d}"
            threat_record = {
                'flow_id': unique_flow_id,
                'src_ip': str(src_ip),
                'dst_ip': str(dst_ip),
                'attack_type': attack_label,
                'attack_label': attack_label,
                'severity': severity,
                'model_used': model_used,
                'risk_analysis': risk_analysis,
                'timestamp': timestamp,
                'detection_models': detection_models,
                'ensemble': ensemble_binary,
                'true_is_threat': true_is_threat,
                'model_predictions': model_predictions  # Add this for accuracy calculation
            }
            
            threat_records.append(threat_record)
            # Store latest by flow_id for manual verification cross-reference
            recent_threats[threat_record['flow_id']] = threat_record
        
        # Create threat analysis
        threat_analysis = {}
        for model_name, results in detection_results.items():
            if 'error' not in results:
                risk_score = results['threats_detected'] / max(sample_size, 1)
                risk_level = 'High' if risk_score > 0.5 else 'Medium' if risk_score > 0.1 else 'Low'
                
                threat_analysis[model_name] = {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'severity': 'High' if results['threats_detected'] > 0 else 'Low'
                }
        
        return jsonify({
            "success": True,
            "data": {
                "detection_results": threat_records,
                "performance_metrics": {
                    "total_flows_analyzed": sample_size,
                    "threats_detected": len([t for t in threat_records if t['ensemble'] == "1"]),
                    "detection_rate": len([t for t in threat_records if t['ensemble'] == "1"]) / sample_size * 100,
                    "model_agreements": len([t for t in threat_records if sum([int(t['aca']), int(t['fuzzy']), int(t['intrudtree'])]) >= 2]),
                    "model_disagreements": len([t for t in threat_records if sum([int(t['aca']), int(t['fuzzy']), int(t['intrudtree'])]) == 1]),
                    "last_detection_time": timestamp
                },
                "summary": {
                    "total_flows": sample_size,
                    "high_risk_flows": len([t for t in threat_records if t['ensemble'] == "1"]),
                    "medium_risk_flows": len([t for t in threat_records if t['ensemble'] == "0" and sum([int(t['aca']), int(t['fuzzy']), int(t['intrudtree'])]) >= 2]),
                    "low_risk_flows": len([t for t in threat_records if t['ensemble'] == "0" and sum([int(t['aca']), int(t['fuzzy']), int(t['intrudtree'])]) == 1])
                }
            },
            "threat_analysis": threat_analysis,
            "detection_time": detection_time,
            "sample_size": sample_size,
            "timestamp": timestamp,
            "mode": "real"
        })
        
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }), 500

@app.route('/api/manual/verify', methods=['POST'])
def manual_verify():
    """Accept manual verification for detected flows."""
    try:
        payload = request.get_json(force=True) or {}
        verifications = payload.get('verifications', [])  # list of {flow_id, is_threat}
        updated = 0
        for item in verifications:
            flow_id = str(item.get('flow_id'))
            if not flow_id:
                continue
            is_threat = bool(item.get('is_threat'))
            manual_verifications[flow_id] = is_threat
            updated += 1
        return jsonify({
            "success": True,
            "updated": updated
        })
    except Exception as e:
        logger.error(f"Manual verify error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/manual/metrics', methods=['GET'])
def manual_metrics():
    """Compute per-model precision among verified positives based on manual verifications."""
    try:
        # Build arrays over verified flow_ids that still exist in recent_threats
        flow_ids = [fid for fid in manual_verifications.keys() if fid in recent_threats]
        if not flow_ids:
            return jsonify({
                "success": True,
                "metrics": {},
                "message": "No verified items yet"
            })

        def compute_precision_for_model(key_fn):
            predicted_positive = 0
            true_confirmed = 0
            for fid in flow_ids:
                rec = recent_threats[fid]
                pred = int(key_fn(rec))
                if pred == 1:
                    predicted_positive += 1
                    if manual_verifications[fid]:
                        true_confirmed += 1
            precision = (true_confirmed / predicted_positive) if predicted_positive > 0 else 0.0
            return {
                'predicted_positive': predicted_positive,
                'true_confirmed': true_confirmed,
                'precision': round(precision, 4)
            }

        metrics = {
            'ACA_RF': compute_precision_for_model(lambda r: r.get('aca', '0')),
            'Fuzzy_RF': compute_precision_for_model(lambda r: r.get('fuzzy', '0')),
            'IntruDTree': compute_precision_for_model(lambda r: r.get('intrudtree', '0')),
            'Ensemble': compute_precision_for_model(lambda r: r.get('ensemble', '0'))
        }

        return jsonify({
            "success": True,
            "metrics": metrics,
            "total_verified": len(flow_ids)
        })
    except Exception as e:
        logger.error(f"Manual metrics error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/realtime/status', methods=['GET'])
def realtime_status():
    """Get real-time detection status (simulated)"""
    try:
        return jsonify({
            "success": True,
            "is_capturing": False,  # Simulated
            "total_packets": 0,     # Simulated
            "active_flows": 0,      # Simulated
            "interface": "simulated"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/realtime/start', methods=['POST'])
def start_realtime_detection():
    """Start real-time detection (simulated)"""
    try:
        return jsonify({
            "success": True,
            "message": "Real-time detection started (simulated)"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/realtime/stop', methods=['POST'])
def stop_realtime_detection():
    """Stop real-time detection (simulated)"""
    try:
        return jsonify({
            "success": True,
            "message": "Real-time detection stopped (simulated)"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/models/performance', methods=['GET'])
def get_model_performance():
    """Get performance metrics for all models"""
    try:
        if orchestrator is None:
            return jsonify({
                "success": False,
                "error": "Orchestrator not initialized"
            }), 500
        
        # Evaluate all models
        eval_result = orchestrator.evaluate_all_models()
        
        if not eval_result['success']:
            return jsonify({
                "success": False,
                "error": eval_result['error']
            }), 500
        
        return jsonify({
            "success": True,
            "performance_metrics": eval_result['evaluation_results']
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate and return a report"""
    try:
        report_type = request.json.get('type', 'summary')
        
        if orchestrator is None:
            return jsonify({
                "success": False,
                "error": "Orchestrator not initialized"
            }), 500
        
        # Generate comparison report
        compare_result = orchestrator.compare_models()
        
        if not compare_result['success']:
            return jsonify({
                "success": False,
                "error": compare_result['error']
            }), 500
        
        return jsonify({
            "success": True,
            "report": compare_result['comparison_results'],
            "type": report_type,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/risk-analysis', methods=['GET'])
def get_risk_analysis_report():
    """Generate risk analysis summary report data"""
    try:
        if orchestrator is None:
            return jsonify({
                "success": True,
                "data": {
                    "severityAnalysis": {},
                    "recommendedActions": {},
                    "summary": {
                        "total_threats": 0,
                        "total_flows": 0,
                        "threat_rate": 0
                    }
                }
            })
        
        # Get recent threat data from the in-memory store
        if not recent_threats:
            return jsonify({
                "success": True,
                "data": {
                    "severityAnalysis": {},
                    "recommendedActions": {},
                    "summary": {
                        "total_threats": 0,
                        "total_flows": 0,
                        "threat_rate": 0
                    }
                }
            })
        
        # Filter threats to only include recent data (last 30 minutes)
        current_time = datetime.now()
        threats_to_analyze = {}
        
        for threat_id, threat_data in recent_threats.items():
            if 'timestamp' in threat_data:
                try:
                    threat_time = datetime.fromisoformat(threat_data['timestamp'])
                    if (current_time - threat_time).total_seconds() <= 1800:  # 30 minutes
                        threats_to_analyze[threat_id] = threat_data
                except ValueError:
                    continue
        
        # Initialize severity analysis structure
        severity_analysis = {
            "High": {"count": 0, "attacks": [], "top_attacks": []},
            "Medium": {"count": 0, "attacks": [], "top_attacks": []},
            "Low": {"count": 0, "attacks": [], "top_attacks": []}
        }
        
        total_threats = 0
        
        # Process filtered recent threats
        for threat_id, threat_data in threats_to_analyze.items():
            # Filter out threats not detected by any model
            model_used = threat_data.get('model_used', '').strip().lower()
            if not model_used or model_used == 'unknown' or model_used == 'simulated':
                continue
            
            # Filter out Unknown attack types
            attack_type = threat_data.get('attack_type', 'Unknown')
            if attack_type.lower() == 'unknown':
                continue
            
            severity = threat_data.get('severity', 'Low')
            
            if severity in severity_analysis:
                severity_analysis[severity]["count"] += 1
                severity_analysis[severity]["attacks"].append(attack_type)
                total_threats += 1
        
        # Calculate attack frequencies per severity
        for severity, data in severity_analysis.items():
            attack_counts = {}
            for attack in data["attacks"]:
                attack_counts[attack] = attack_counts.get(attack, 0) + 1
            
            # Sort by count and get top attacks
            top_attacks = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            data["top_attacks"] = top_attacks
        
        # Define recommended actions
        recommended_actions = {
            "High": {
                "action": "Immediate isolation and system scan",
                "description": "Isolate affected systems, terminate malicious processes, and initiate incident response.",
                "timeframe": "Immediate (0-15 minutes)"
            },
            "Medium": {
                "action": "Monitor and restrict access",
                "description": "Increase monitoring frequency, analyze patterns, prepare for escalation.",
                "timeframe": "Within 1 hour"
            },
            "Low": {
                "action": "Log and schedule review",
                "description": "Log the incident, conduct routine analysis, monitor for patterns.",
                "timeframe": "Within 24 hours"
            }
        }
        
        return jsonify({
            "success": True,
            "data": {
                "severityAnalysis": severity_analysis,
                "recommendedActions": recommended_actions,
                "summary": {
                    "total_threats": total_threats,
                    "total_flows": len(threats_to_analyze),
                    "threat_rate": round((total_threats / len(threats_to_analyze) * 100), 2) if threats_to_analyze else 0
                }
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating risk analysis report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/attack-frequency', methods=['GET'])
def get_attack_frequency_report():
    """Generate attack type frequency report data"""
    try:
        if not recent_threats:
            return jsonify({
                "success": True,
                "data": {
                    "attackFrequencyData": [],
                    "summary": {
                        "total_attacks": 0,
                        "unique_attack_types": 0,
                        "high_severity": 0,
                        "medium_severity": 0,
                        "low_severity": 0
                    }
                }
            })

        # Filter out benign traffic and count attack types
        attack_counts = {}
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        
        for threat_id, threat in recent_threats.items():
            # Filter out threats not detected by any model
            model_used = threat.get('model_used', '').strip().lower()
            if not model_used or model_used == 'unknown' or model_used == 'simulated':
                continue
            
            # Get attack type from either attack_type or attack_label field
            attack_type = threat.get('attack_type') or threat.get('attack_label', 'Unknown')
            
            # Filter out Unknown attack types and benign traffic
            if not attack_type or attack_type.lower() == 'benign' or attack_type.lower() == 'unknown':
                continue
                
            # Initialize attack type counter if not exists
            if attack_type not in attack_counts:
                attack_counts[attack_type] = {
                    'count': 0,
                    'severity_breakdown': {"High": 0, "Medium": 0, "Low": 0}
                }
            
            # Increment counters
            attack_counts[attack_type]['count'] += 1
            severity = threat.get('severity', 'Low')
            attack_counts[attack_type]['severity_breakdown'][severity] += 1
            severity_counts[severity] += 1

        # Calculate percentages and format output
        total_attacks = sum(item['count'] for item in attack_counts.values())
        attack_frequency_data = []

        for attack_type, data in attack_counts.items():
            percentage = (data['count'] / total_attacks * 100) if total_attacks > 0 else 0
            attack_frequency_data.append({
                "attackType": attack_type,  # Make sure attack type is included
                "count": data['count'],
                "percentage": round(percentage, 2),
                "severity_breakdown": data['severity_breakdown']
            })

        # Sort by count in descending order
        attack_frequency_data.sort(key=lambda x: x['count'], reverse=True)

        return jsonify({
            "success": True,
            "data": {
                "attackFrequencyData": attack_frequency_data,
                "summary": {
                    "total_attacks": total_attacks,
                    "unique_attack_types": len(attack_counts),
                    "severity_distribution": severity_counts,
                    "most_common": attack_frequency_data[0] if attack_frequency_data else None,
                    "least_common": attack_frequency_data[-1] if attack_frequency_data else None
                }
            }
        })

    except Exception as e:
        logger.error(f"Error generating attack frequency report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/export/<report_type>', methods=['POST'])
def export_report(report_type):
    """Export report data in various formats"""
    try:
        format_type = request.json.get('format', 'pdf')
        
        if report_type == 'risk-analysis':
            # Get risk analysis data
            risk_data = get_risk_analysis_report()
            if not risk_data[0].get('success'):
                return jsonify(risk_data[0]), risk_data[1]
            
            data = risk_data[0]['data']
            
            if format_type == 'csv':
                # Generate CSV content
                csv_content = "Severity Level,Count,Percentage,Top Attack Types,Recommended Action\n"
                
                for severity, analysis in data['severityAnalysis'].items():
                    top_attacks = "; ".join([f"{attack[0]} ({attack[1]})" for attack in analysis['top_attacks']])
                    action = data['recommendedActions'][severity]['action']
                    csv_content += f"{severity},{analysis['count']},{analysis['percentage']}%,{top_attacks},{action}\n"
                
                return jsonify({
                    "success": True,
                    "data": csv_content,
                    "filename": f"risk_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "content_type": "text/csv"
                })
            
            elif format_type == 'pdf':
                # For PDF, return structured data that frontend can format
                return jsonify({
                    "success": True,
                    "data": data,
                    "format": "pdf",
                    "filename": f"risk_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                })
        
        elif report_type == 'attack-frequency':
            # Get attack frequency data
            freq_data = get_attack_frequency_report()
            if not freq_data[0].get('success'):
                return jsonify(freq_data[0]), freq_data[1]
            
            data = freq_data[0]['data']
            
            if format_type == 'csv':
                # Generate CSV content
                csv_content = "Attack Type,Count,Percentage,High Severity,Medium Severity,Low Severity\n"
                
                for attack in data['attackFrequencyData']:
                    csv_content += f"{attack['attack_type']},{attack['count']},{attack['percentage']}%,{attack['severity_breakdown']['High']},{attack['severity_breakdown']['Medium']},{attack['severity_breakdown']['Low']}\n"
                
                return jsonify({
                    "success": True,
                    "data": csv_content,
                    "filename": f"attack_frequency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "content_type": "text/csv"
                })
            
            elif format_type == 'excel':
                # For Excel, return structured data that frontend can format
                return jsonify({
                    "success": True,
                    "data": data,
                    "format": "excel",
                    "filename": f"attack_frequency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                })
        
        return jsonify({
            "success": False,
            "error": f"Unsupported report type: {report_type}"
        }), 400
        
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# --- Streaming / real-time simulation state ---
stream_rows = []     # list of dicts representing dataset rows + labels
stream_index = 0
stream_lock = None

def _detect_for_row(X_row, true_label_idx=None, label_encoder=None, original_features=None):
    """Run available models on a single row (DataFrame with one row) and return detection summary."""
    try:
        # Convert original_features DataFrame to dict if needed (prevent JSON serialization issues)
        if original_features is not None and isinstance(original_features, pd.DataFrame):
            original_features = original_features.iloc[0].to_dict()
        
        benign_idx = 0
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            classes = list(label_encoder.classes_)
            if 'Benign' in classes:
                benign_idx = classes.index('Benign')
        results = {}
        votes = []
        detection_models = []
        for model_name, model in orchestrator.models.items():
            try:
                preds, details = model.predict(X_row)
                pred = int(preds[0])
                is_threat = 1 if pred != benign_idx else 0
                votes.append(is_threat)
                
                # Get individual model's attack label and severity
                model_attack_label = "Unknown"
                if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                    if 0 <= pred < len(label_encoder.classes_):
                        model_attack_label = label_encoder.classes_[pred]
                
                # Calculate model severity based on attack type characteristics and confidence
                # First, determine severity category for the attack type
                high_severity_attacks = ['DoS', 'DDoS', 'Exploit', 'Rootkit', 'Trojan', 'Virus', 'Worm', 'Backdoor']
                medium_severity_attacks = ['Port Scan', 'HackTool', 'Hoax', 'Botnet', 'Infiltration', 'Web Attack']
                
                # Base severity on attack type
                if model_attack_label in high_severity_attacks:
                    model_severity = "High"
                elif model_attack_label in medium_severity_attacks:
                    model_severity = "Medium"
                elif model_attack_label == "Benign":
                    model_severity = "Low"
                else:
                    # Unknown or other attack types - assess based on threat status
                    model_severity = "Medium" if is_threat == 1 else "Low"
                
                # Add model to detection_models if it detected a threat (for reporting which models detected ANY threat)
                if is_threat == 1:
                    detection_models.append(model_name)
                
                # Convert details to JSON-serializable format
                serializable_details = {}
                for key, value in details.items():
                    try:
                        if isinstance(value, np.ndarray):
                            serializable_details[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            serializable_details[key] = float(value)
                        else:
                            serializable_details[key] = value
                    except:
                        serializable_details[key] = str(value)
                
                results[model_name] = {
                    'pred': pred,
                    'is_threat': is_threat,
                    'details': serializable_details,
                    'attack_label': model_attack_label,
                    'severity': model_severity
                }
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                results[model_name] = {'error': str(e), 'is_threat': 0, 'attack_label': 'Error', 'severity': 'Low'}
                votes.append(0)
                # Don't add model if it didn't detect a threat
                # (model name excluded on error)
        
        # Get the final detected attack type (from true label or ensemble decision)
        final_attack_label = None
        if label_encoder is not None and true_label_idx is not None and 0 <= true_label_idx < len(label_encoder.classes_):
            final_attack_label = label_encoder.classes_[true_label_idx]
        elif true_label_idx is not None:
            # Fallback mapping
            attack_mapping = {
                0: 'Backdoor',
                1: 'Benign', 
                2: 'Exploit',
                3: 'HackTool',
                4: 'Hoax',
                5: 'Rootkit',
                6: 'Trojan',
                7: 'Virus',
                8: 'Worm'
            }
            final_attack_label = attack_mapping.get(true_label_idx, "Unknown")
        
        # Ensemble majority
        ensemble_binary = 1 if sum(votes) >= 2 else 0
        
        # Use the already-determined final_attack_label as attack_label
        attack_label = final_attack_label
        if attack_label is None:
            attack_label = "Unknown"
        
        # Filter detection_models to only include models that predicted the SAME attack type as the final detected type
        # This ensures "Detected by" only shows models that matched the final prediction
        matching_detection_models = []
        for model_name, model_data in results.items():
            model_predicted_label = model_data.get('attack_label', 'Unknown')
            if model_predicted_label == attack_label:
                matching_detection_models.append(model_name)
        
        detection_models = matching_detection_models

        # Dynamic severity mapping based on attack type characteristics and model agreement
        severity = "Low"
        
        # Categorize attacks by inherent severity (can be adjusted based on domain knowledge)
        # High severity: Attacks that directly compromise system integrity or data
        high_severity_attacks = ['DoS', 'DDoS', 'Exploit', 'Rootkit', 'Trojan', 'Virus', 'Worm', 'Backdoor']
        
        # Medium severity: Attacks that require attention but less immediate
        medium_severity_attacks = ['Port Scan', 'HackTool', 'Hoax', 'Botnet', 'Infiltration', 'Web Attack']
        
        # Use matching_models_count (models that predicted this specific attack type)
        # instead of votes (models that detected any threat)
        matching_models_count = len(detection_models)
        
        if attack_label in high_severity_attacks:
            # For high-severity attacks, severity depends on model confidence
            if matching_models_count >= 2:  # High confidence - multiple models agree
                severity = "High"
            elif matching_models_count == 1:  # Medium confidence - single model detection
                severity = "High"  # Still high for dangerous attacks even with single detection
            else:  # Low confidence - using ground truth label but no model predicted it
                severity = "Medium"  # Less certain but still concerning
        elif attack_label in medium_severity_attacks:
            # For medium-severity attacks, more conservative
            if matching_models_count >= 2:
                severity = "Medium"
            elif matching_models_count == 1:
                severity = "Medium"
            else:
                severity = "Low"  # Low confidence on medium-severity attack
        elif attack_label == "Benign":
            severity = "Low"
        else:
            # Unknown attack types - conservative approach
            if matching_models_count >= 2:
                severity = "Medium"
            elif matching_models_count == 1:
                severity = "Low"
            else:
                severity = "Low"

        # Enhanced risk analysis with model information
        # Use the matching_models_count calculated above
        total_models = len(results)
        
        if attack_label == "Benign":
            risk_analysis = "🟢 Low risk: Normal traffic confirmed"
        elif matching_models_count >= 2:
            # Majority agreement on this specific attack type
            if severity == "High":
                risk_analysis = f"🔥 High risk: {attack_label} confirmed by {matching_models_count}/{total_models} models - immediate action required"
            elif severity == "Medium":
                risk_analysis = f"⚠️ Medium risk: {attack_label} confirmed by {matching_models_count}/{total_models} models - monitor closely"
            else:
                risk_analysis = f"🟡 Low risk: {attack_label} confirmed by {matching_models_count}/{total_models} models"
        elif matching_models_count == 1:
            # Single model detected this specific attack type
            if severity == "High":
                risk_analysis = f"🔥 High risk: {attack_label} detected by 1/{total_models} models - investigate immediately"
            elif severity == "Medium":
                risk_analysis = f"⚠️ Medium risk: {attack_label} detected by 1/{total_models} models - investigate"
            else:
                risk_analysis = f"🟡 Low risk: {attack_label} detected by 1/{total_models} models"
        else:
            # No models detected this specific attack type (using ground truth)
            if severity == "High":
                risk_analysis = f"🔥 High risk: {attack_label} detected (ground truth) - manual review required"
            elif severity == "Medium":
                risk_analysis = f"⚠️ Medium risk: {attack_label} detected (ground truth) - manual review recommended"
            else:
                risk_analysis = f"🟡 Low risk: {attack_label} detected (ground truth) - routine monitoring"

        # Extract key features for manual verification
        key_features = {}
        # Use original_features if provided (for stream), otherwise use X_row
        if original_features is not None:
            # original_features is already a dict (converted at function entry)
            feature_dict = original_features
        elif X_row is not None and isinstance(X_row, pd.DataFrame):
            feature_dict = X_row.iloc[0].to_dict()
        else:
            feature_dict = {}
        
        if feature_dict:
            try:
                
                # Try important features first
                important_features = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 
                                     'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 
                                     'Flow Bytes/s', 'Flow Packets/s', 'FIN Flag Count',
                                     'SYN Flag Count', 'ACK Flag Count']
                
                for col in important_features:
                    if col in feature_dict:
                        try:
                            value = feature_dict[col]
                            key_features[col] = float(value) if isinstance(value, (int, float)) else str(value)
                        except:
                            pass
                
                # If we don't have enough features yet, add more from any columns
                if len(key_features) < 5:
                    for col, value in feature_dict.items():
                        if col not in key_features and len(key_features) < 10:
                            try:
                                key_features[col] = float(value) if isinstance(value, (int, float)) else str(value)
                            except:
                                pass
            except Exception as e:
                logger.warning(f"Error extracting features: {e}")
        
        # Convert key_features values to serializable types
        serializable_key_features = {}
        for key, value in key_features.items():
            try:
                if isinstance(value, (np.integer, np.floating)):
                    serializable_key_features[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_key_features[key] = value.tolist()
                else:
                    serializable_key_features[key] = value
            except:
                serializable_key_features[key] = str(value)
        
        return {
            'models': results,
            'detection_models': detection_models,
            'ensemble': ensemble_binary,
            'severity': severity,
            'attack_label': attack_label,
            'risk_analysis': risk_analysis,
            'key_features': serializable_key_features  # Add feature data for manual verification
        }
    except Exception as e:
        return {'error': str(e)}

def load_stream_data():
    """Populate stream_rows from orchestrator.loaded_data or from configured CSV file."""
    global stream_rows, stream_index
    stream_rows = []
    stream_index = 0
    try:
        # Prefer orchestrator.loaded_data
        if orchestrator is not None and hasattr(orchestrator, 'loaded_data') and orchestrator.loaded_data:
            ld = orchestrator.loaded_data
            X = ld.get('X_test')
            meta = ld.get('metadata') if 'metadata' in ld else None
            y = ld.get('y_test')
            if X is not None:
                # Use a larger sample and ensure IP diversity by avoiding repeated IP pairs
                sample_size = min(5000, len(X))  # Increased from 1000 to 5000 for more diversity
                
                # Create a more diverse sample with better spacing and IP diversity tracking
                if len(X) > sample_size:
                    # Use more aggressive sampling to get better IP diversity
                    # Take samples from different parts of the dataset
                    sample_indices = []
                    
                    # Sample from beginning, middle, and end of dataset
                    sections = [
                        (0, len(X) // 3),  # First third
                        (len(X) // 3, 2 * len(X) // 3),  # Middle third
                        (2 * len(X) // 3, len(X))  # Last third
                    ]
                    
                    samples_per_section = sample_size // 3
                    for start, end in sections:
                        if start < end:
                            section_indices = np.random.choice(range(start, end), min(samples_per_section, end - start), replace=False)
                            sample_indices.extend(section_indices)
                    
                    # Add some random indices for variety
                    remaining = sample_size - len(sample_indices)
                    if remaining > 0:
                        random_indices = np.random.choice(len(X), remaining, replace=False)
                        sample_indices.extend(random_indices)
                    
                    sample_indices = list(set(sample_indices))[:sample_size]  # Remove duplicates and limit size
                else:
                    sample_indices = list(range(len(X)))
                
                # Track used IP pairs to avoid repetition
                used_ip_pairs = set()
                final_entries = []
                
                for i in sample_indices:
                    row = X.iloc[i].to_dict()
                    # Attach label idx if available
                    label_idx = int(y.values[i]) if y is not None else None
                    
                    # Handle metadata alignment with better randomization
                    meta_row = {}
                    if meta is not None:
                        # Use the same index for metadata if available
                        if i < len(meta):
                            meta_row = meta.iloc[i].to_dict()
                        else:
                            # Use a different randomization strategy for metadata
                            meta_idx = (i * 7) % len(meta)  # Use prime number multiplication for better distribution
                            meta_row = meta.iloc[meta_idx].to_dict()
                    
                    # If still no metadata, try to extract from features
                    if not meta_row:
                        # Look for IP-like columns in features
                        for key, value in row.items():
                            if any(ip_key in key.lower() for ip_key in ['ip', 'source', 'destination', 'src', 'dst']):
                                if 'src' in key.lower() or 'source' in key.lower():
                                    meta_row['src_ip'] = value
                                elif 'dst' in key.lower() or 'destination' in key.lower():
                                    meta_row['dst_ip'] = value
                    
                    # Check if this IP pair has been used before
                    src_ip = meta_row.get('src_ip', 'N/A')
                    dst_ip = meta_row.get('dst_ip', 'N/A')
                    ip_pair = (src_ip, dst_ip)
                    
                    # STRICT IP diversity: Never allow the same IP pair to be used twice
                    if ip_pair not in used_ip_pairs:
                        used_ip_pairs.add(ip_pair)
                        
                        entry = {
                            'features': row,
                            'label_idx': label_idx,
                            'meta': meta_row
                        }
                        final_entries.append(entry)
                    
                    # Stop if we have enough diverse entries
                    if len(final_entries) >= sample_size:
                        break
                
                # Use the final entries with IP diversity
                stream_rows = final_entries
                
                # If we don't have enough entries, generate some with synthetic IPs to ensure diversity
                if len(stream_rows) < sample_size // 2:  # If we have less than half the desired sample
                    logger.warning(f"Only got {len(stream_rows)} unique IP pairs, generating synthetic ones for diversity")
                    
                    # Generate synthetic IP addresses for remaining samples
                    synthetic_count = min(sample_size - len(stream_rows), 1000)  # Limit synthetic IPs
                    for j in range(synthetic_count):
                        # Generate synthetic IP addresses
                        src_ip = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                        dst_ip = f"10.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
                        
                        # Ensure this synthetic pair is unique
                        while (src_ip, dst_ip) in used_ip_pairs:
                            src_ip = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                            dst_ip = f"10.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
                        
                        used_ip_pairs.add((src_ip, dst_ip))
                        
                        # Create synthetic entry with random features
                        synthetic_row = {}
                        for key in X.columns:
                            synthetic_row[key] = np.random.random()  # Random feature values
                        
                        synthetic_entry = {
                            'features': synthetic_row,
                            'label_idx': np.random.randint(0, 10),  # Random attack type
                            'meta': {'src_ip': src_ip, 'dst_ip': dst_ip}
                        }
                        stream_rows.append(synthetic_entry)
                
                return {'success': True, 'rows': len(stream_rows)}
        # Fallback: read configured data_path CSV (raw file)
        data_path = os.path.join(os.path.dirname(__file__), orchestrator.config.get('data_path')) if orchestrator and orchestrator.config else None
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # try to detect label column
            label_cols = [c for c in df.columns if c.lower() in ('label','attack','attack_type','class','result')]
            label_col = label_cols[0] if label_cols else None
            for _, r in df.iterrows():
                features = r.to_dict()
                label_val = None
                if label_col:
                    label_val = r[label_col]
                stream_rows.append({'features': features, 'label_value': label_val, 'meta': {}})
            return {'success': True, 'rows': len(stream_rows)}
        return {'success': False, 'error': 'No data available to stream'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/api/stream/reset', methods=['POST', 'GET'])
def stream_reset():
    """Initialize / reset the streaming dataset iterator."""
    try:
        res = load_stream_data()
        if not res.get('success', False):
            return jsonify({'success': False, 'error': res.get('error', 'failed to load data')}), 500
        
        # Validate IP diversity
        if 'stream_rows' in globals() and stream_rows:
            ip_pairs = set()
            unique_ips = set()
            for entry in stream_rows[:100]:  # Check first 100 entries
                meta = entry.get('meta', {})
                src_ip = meta.get('src_ip', 'N/A')
                dst_ip = meta.get('dst_ip', 'N/A')
                if src_ip != 'N/A' and dst_ip != 'N/A':
                    ip_pairs.add((src_ip, dst_ip))
                    unique_ips.add(src_ip)
                    unique_ips.add(dst_ip)
            
            logger.info(f"IP Diversity Check: {len(ip_pairs)} unique IP pairs, {len(unique_ips)} unique IPs in first 100 entries")
        
        return jsonify({'success': True, 'rows': res.get('rows', 0)})
    except Exception as e:
        logger.error(f"stream_reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stream/next', methods=['GET'])
def stream_next():
    """Return the next row (simulate detection). One call => one displayed detection."""
    global stream_index, stream_rows
    try:
        if not stream_rows:
            # Try to load automatically if not initialized
            load_res = load_stream_data()
            if not load_res.get('success', False):
                return jsonify({'success': False, 'error': 'No stream data available'}), 500

        if stream_index >= len(stream_rows):
            return jsonify({'success': False, 'finished': True, 'message': 'All rows streamed'}), 200

        entry = stream_rows[stream_index]
        stream_index += 1

        # Build minimal DataFrame for prediction using features
        X_row = None
        true_label_idx = entry.get('label_idx')
        label_encoder = None
        try:
            label_encoder_path = os.path.join(orchestrator.config['processed_data_path'], 'label_encoder.joblib') if orchestrator else None
            if label_encoder_path and os.path.exists(label_encoder_path):
                import joblib
                label_encoder = joblib.load(label_encoder_path)
        except Exception:
            label_encoder = None

        if 'features' in entry and isinstance(entry['features'], dict):
            X_row = pd.DataFrame([entry['features']])
        else:
            X_row = pd.DataFrame([entry.get('meta', {})])

        detection = {}
        # Convert X_row to dict for passing as original_features (prevents DataFrame serialization issues)
        original_features_dict = None
        if X_row is not None and isinstance(X_row, pd.DataFrame):
            try:
                original_features_dict = X_row.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Error converting X_row to dict: {e}")
                original_features_dict = None
        
        if orchestrator is not None and hasattr(orchestrator, 'models') and orchestrator.models:
            try:
                detection = _detect_for_row(X_row, true_label_idx=true_label_idx, label_encoder=label_encoder, original_features=original_features_dict)
            except Exception as e:
                logger.error(f"Error in _detect_for_row: {e}")
                # Fallback to simulated detection
                attack_label = entry.get('meta', {}).get('Attack', entry.get('label_value')) if entry.get('meta') else entry.get('label_value', 'Unknown')
                detection = {
                    'models': {},
                    'detection_models': ['Error'],
                    'ensemble': 1 if attack_label and attack_label != 'Benign' else 0,
                    'severity': 'High' if attack_label and attack_label not in (None, 'Benign') else 'Low',
                    'attack_label': attack_label,
                    'risk_analysis': f"Error: {str(e)}",
                    'key_features': {}
                }
        else:
            # Simulated detection: use label if present
            attack_label = entry.get('meta', {}).get('Attack', entry.get('label_value')) if entry.get('meta') else entry.get('label_value', 'Unknown')
            detection = {
                'models': {},
                'detection_models': ['Simulated'],
                'ensemble': 1 if attack_label and attack_label != 'Benign' else 0,
                'severity': 'High' if attack_label and attack_label not in (None, 'Benign') else 'Low',
                'attack_label': attack_label,
                'risk_analysis': f"Simulated: {attack_label}",
                'key_features': {}
            }

        # Extract common fields from metadata/features
        meta = entry.get('meta', {}) or {}
        features = entry.get('features', {}) or {}
        
        # Enhanced IP extraction with more column name variations
        def find_first(keys, source):
            for k in keys:
                if k in source:
                    val = source.get(k)
                    if val and str(val).strip() and str(val).lower() not in ['nan', 'none', '']:
                        return str(val).strip()
            return "N/A"
        
        # More comprehensive IP column name patterns
        src_ip_keys = ['Src IP', 'src_ip', 'source_ip', 'srcip', 'Source IP', 'sourceip', 'src', 
                       'Source_IP', 'source_ip_address', 'src_ip_addr', 'sourceipaddress']
        dst_ip_keys = ['Dst IP', 'dst_ip', 'destination_ip', 'dstip', 'Destination IP', 'destinationip', 'dst',
                       'Destination_IP', 'dest_ip_address', 'dst_ip_addr', 'destinationipaddress']
        
        # First try to get IPs from metadata (which we know has the correct columns)
        src_ip = find_first(src_ip_keys, meta)
        dst_ip = find_first(dst_ip_keys, meta)
        
        # If not found in metadata, try features as fallback
        if src_ip == "N/A":
            src_ip = find_first(src_ip_keys, features)
        if dst_ip == "N/A":
            dst_ip = find_first(dst_ip_keys, features)
        
        # Debug logging for IP extraction
        if src_ip != "N/A" or dst_ip != "N/A":
            logger.info(f"IP extraction successful - Src: {src_ip}, Dst: {dst_ip}")
        else:
            logger.warning(f"IP extraction failed - Meta keys: {list(meta.keys())}, Feature keys: {list(features.keys())[:10]}")

        timestamp = datetime.now().isoformat()
        model_used = ",".join(detection.get('detection_models', [])) if detection.get('detection_models') else "Unknown"
        severity = detection.get('severity', 'Low')
        risk_analysis = detection.get('risk_analysis', '')

        # Generate unique flow ID for this detection
        flow_id = f"flow_{int(time.time()*1000)}_{stream_index:04d}"
        
        # Calculate true_is_threat from attack_label
        true_is_threat = detection.get('attack_label', 'Unknown') != "Benign"
        
        # Convert models to JSON-serializable format
        serializable_models = {}
        model_predictions = {}
        if detection.get('models'):
            for model_name, model_data in detection.get('models', {}).items():
                serializable_models[model_name] = {
                    'pred': model_data.get('pred'),
                    'is_threat': model_data.get('is_threat'),
                    'attack_label': model_data.get('attack_label', 'Unknown'),
                    'severity': model_data.get('severity', 'Low'),
                    'details': {}  # Simplified - just store essential data
                }
                
                # Build model_predictions for accuracy calculation
                predicted_is_threat = model_data.get('is_threat', 0) == 1
                model_predictions[model_name] = {
                    'predicted_threat': predicted_is_threat,
                    'is_correct': (predicted_is_threat == true_is_threat),
                    'attack_label': model_data.get('attack_label', 'Unknown')  # Add attack label for type comparison
                }
        
        # Display all flows (threats and benign) - every model always makes a prediction
        detection_models_list = detection.get('detection_models', [])
        
        # Store in recent_threats for report access
        threat_record = {
            'flow_id': flow_id,
            'src_ip': str(src_ip),
            'dst_ip': str(dst_ip),
            'attack_type': detection.get('attack_label', 'Unknown'),
            'attack_label': detection.get('attack_label', 'Unknown'),  # Make sure both are set
            'severity': severity,
            'model_used': model_used,
            'risk_analysis': risk_analysis,
            'timestamp': timestamp,
            'detection_models': detection_models_list,
            'ensemble': detection.get('ensemble', 0),
            'true_is_threat': true_is_threat,
            'model_predictions': model_predictions
        }
        recent_threats[flow_id] = threat_record
        
        out = {
            'success': True,
            'row': {
                'flow_id': flow_id,
                'timestamp': timestamp,
                'src_ip': str(src_ip),
                'dst_ip': str(dst_ip),
                'attack_type': detection.get('attack_label', 'Unknown'),
                'model_used': model_used,
                'severity': severity,
                'risk_analysis': risk_analysis,
                'true_is_threat': true_is_threat,
                'model_predictions': model_predictions,
                'key_features': detection.get('key_features', {}),  # Add feature data for verification
                'models': serializable_models  # Add individual model predictions (JSON-safe)
            },
            'remaining': max(0, len(stream_rows) - stream_index)
        }
        return jsonify(out)
    except Exception as e:
        logger.error(f"stream_next error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stream/state', methods=['GET'])
def stream_state():
    try:
        total = len(stream_rows)
        remaining = max(0, total - stream_index)
        return jsonify({'success': True, 'total': total, 'streamed': stream_index, 'remaining': remaining})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug/columns', methods=['GET'])
def debug_columns():
    """Debug endpoint to see what columns are available in the data"""
    try:
        if orchestrator is not None and hasattr(orchestrator, 'loaded_data') and orchestrator.loaded_data:
            ld = orchestrator.loaded_data
            X = ld.get('X_test')
            meta = ld.get('metadata')
            
            info = {
                'X_test_columns': list(X.columns) if X is not None else [],
                'metadata_columns': list(meta.columns) if meta is not None else [],
                'X_test_shape': X.shape if X is not None else None,
                'metadata_shape': meta.shape if meta is not None else None,
                'sample_features': X.iloc[0].to_dict() if X is not None and len(X) > 0 else {},
                'sample_metadata': meta.iloc[0].to_dict() if meta is not None and len(meta) > 0 else {}
            }
            
            return jsonify({'success': True, 'debug_info': info})
        else:
            return jsonify({'success': False, 'error': 'No data loaded'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug/recent-threats', methods=['GET'])
def debug_recent_threats():
    """Debug endpoint to see what threat data is available"""
    try:
        return jsonify({
            'success': True,
            'recent_threats_count': len(recent_threats),
            'recent_threats_keys': list(recent_threats.keys())[:10],  # First 10 keys
            'sample_threat': list(recent_threats.values())[0] if recent_threats else None,
            'manual_verifications_count': len(manual_verifications)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reports/clear-data', methods=['POST'])
def clear_report_data():
    """Clear all cached threat data for reports"""
    try:
        global recent_threats, manual_verifications
        
        # Clear all cached data
        recent_threats.clear()
        manual_verifications.clear()
        
        logger.info("Cleared all cached threat data for reports")
        
        return jsonify({
            "success": True,
            "message": "All cached threat data cleared successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing report data: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/status', methods=['GET'])
def get_report_status():
    """Get current status of report data"""
    try:
        current_time = datetime.now()
        
        # Calculate data age
        data_age_minutes = 0
        if recent_threats:
            # Get the most recent timestamp
            timestamps = []
            for threat_data in recent_threats.values():
                if 'timestamp' in threat_data:
                    try:
                        threat_time = datetime.fromisoformat(threat_data['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(threat_time)
                    except:
                        continue
            
            if timestamps:
                latest_timestamp = max(timestamps)
                data_age_minutes = (current_time - latest_timestamp.replace(tzinfo=None)).total_seconds() / 60
        
        return jsonify({
            "success": True,
            "data": {
                "total_threats": len(recent_threats),
                "manual_verifications": len(manual_verifications),
                "data_age_minutes": round(data_age_minutes, 2),
                "is_stale": data_age_minutes > 30,  # Consider data stale after 30 minutes
                "latest_timestamp": max([t.get('timestamp', '') for t in recent_threats.values()]) if recent_threats else None
            },
            "timestamp": current_time.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting report status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/models/detailed-metrics', methods=['GET'])
def get_detailed_metrics():
    """Get comprehensive model performance metrics"""
    try:
        metrics = {}
        for model_name in ['ACA_RF', 'Fuzzy_RF', 'IntruDTree']:
            predictions = recent_threats.values()
            verified = manual_verifications
            
            true_positives = sum(1 for p in predictions 
                               if p['model'] == model_name and verified.get(p['flow_id']))
            total_detected = sum(1 for p in predictions if p['model'] == model_name)
            total_verified = len([v for v in verified.values() if v is True])
            
            metrics[model_name] = {
                'precision': (true_positives / total_detected * 100) if total_detected else 0,
                'coverage': (true_positives / total_verified * 100) if total_verified else 0,
                'reliability_score': calculate_reliability_score(precision, coverage),
                'efficiency_ratio': calculate_efficiency_ratio(model_name),
                'confidence_intervals': calculate_confidence_intervals(true_positives, total_detected)
            }
            
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Create required directories
    required_dirs = [
        'data/processed',
        'data/raw',
        'models',
        'logs'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Check for missing files on startup
    missing_files = check_required_files()
    if missing_files:
        logger.warning("Missing required files:")
        for file in missing_files:
            logger.warning(f"- {file}")
        logger.warning("Some features may not work properly")
    
    # Initialize orchestrator
    if initialize_orchestrator():
        logger.info("✅ Backend API server initialized successfully")
    else:
        logger.warning("⚠️ Backend API server initialized with limited functionality")
    
    # Start the server
    logger.info("🚀 Starting Flask API server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
