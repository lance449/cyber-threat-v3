"""
Model 1: Aho-Corasick + Random Forest (ACA + RF)
Signature-based pattern detection with Random Forest classification
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import re
import os

logger = logging.getLogger(__name__)

try:
    import pyahocorasick
    PYAHOCORASICK_AVAILABLE = True
    logger.info("pyahocorasick successfully imported")
except ImportError:
    PYAHOCORASICK_AVAILABLE = False
    logger.warning("pyahocorasick not available, using fallback pattern matching")

class AhoCorasickDetector:
    """
    Aho-Corasick algorithm implementation for signature-based pattern detection
    Falls back to simple pattern matching if pyahocorasick is not available
    """
    
    def __init__(self):
        """Initialize the Aho-Corasick detector"""
        self.patterns = []
        self.malware_signatures = self._load_malware_signatures()
        
        if PYAHOCORASICK_AVAILABLE:
            self.automaton = pyahocorasick.Automaton()
            self._build_automaton()
        else:
            logger.info("Using fallback pattern matching implementation")
    
    def _load_malware_signatures(self) -> List[str]:
        """
        Load malware signatures/patterns for detection
        
        Returns:
            List of malware signatures
        """
        # Define common malware signatures based on network behavior
        signatures = [
            # DDoS attack patterns
            "ddos", "flood", "syn_flood", "udp_flood", "icmp_flood",
            
            # Port scanning patterns
            "port_scan", "syn_scan", "fin_scan", "xmas_scan", "null_scan",
            
            # Malware communication patterns
            "c2_communication", "command_control", "botnet", "backdoor",
            
            # Exploit patterns
            "buffer_overflow", "sql_injection", "xss", "csrf",
            
            # Suspicious protocol patterns
            "unusual_protocol", "encrypted_traffic", "tunneled_traffic",
            
            # Anomalous behavior patterns
            "data_exfiltration", "lateral_movement", "privilege_escalation",
            
            # Specific malware families
            "trojan", "virus", "worm", "ransomware", "spyware", "adware",
            
            # Network reconnaissance
            "network_discovery", "service_enumeration", "vulnerability_scan",
            
            # Evasion techniques
            "packet_fragmentation", "timing_manipulation", "protocol_manipulation"
        ]
        
        return signatures
    
    def _build_automaton(self):
        """Build the Aho-Corasick automaton with malware signatures"""
        if PYAHOCORASICK_AVAILABLE:
            for idx, pattern in enumerate(self.malware_signatures):
                self.automaton.add_word(pattern.lower(), (idx, pattern))
                self.patterns.append(pattern)
            
            self.automaton.make_automaton()
            logger.info(f"Built Aho-Corasick automaton with {len(self.patterns)} patterns")
        else:
            self.patterns = self.malware_signatures
            logger.info(f"Using fallback pattern matching with {len(self.patterns)} patterns")
    
    def _fallback_pattern_match(self, text: str) -> List[Tuple]:
        """
        Fallback pattern matching using simple string search
        
        Args:
            text: Text to search for patterns
            
        Returns:
            List of matches as (index, pattern) tuples
        """
        matches = []
        text_lower = text.lower()
        
        for idx, pattern in enumerate(self.patterns):
            if pattern.lower() in text_lower:
                matches.append((idx, pattern))
        
        return matches
    
    def extract_pattern_features(self, flow_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pattern-based features from flow data
        
        Args:
            flow_data: Network flow data
            
        Returns:
            DataFrame with pattern features
        """
        pattern_features = pd.DataFrame(index=flow_data.index)
        
        # Convert flow data to text representation for pattern matching
        flow_texts = self._flow_to_text(flow_data)
        
        # Pattern matching scores
        pattern_scores = []
        pattern_matches = []
        
        for text in flow_texts:
            if PYAHOCORASICK_AVAILABLE:
                matches = list(self.automaton.iter(text.lower()))
            else:
                matches = self._fallback_pattern_match(text)
            
            score = len(matches)
            pattern_scores.append(score)
            pattern_matches.append([match[1] for match in matches])
        
        # Add pattern-based features
        pattern_features['pattern_score'] = pattern_scores
        pattern_features['pattern_match_count'] = pattern_scores
        pattern_features['has_pattern_match'] = (np.array(pattern_scores) > 0).astype(int)
        
        # Add individual pattern features
        for pattern in self.patterns:
            pattern_features[f'pattern_{pattern}'] = [
                1 if pattern in matches else 0 
                for matches in pattern_matches
            ]
        
        logger.info(f"Extracted {len(pattern_features.columns)} pattern features")
        return pattern_features
    
    def _flow_to_text(self, flow_data: pd.DataFrame) -> List[str]:
        """
        Convert flow data to text representation for pattern matching
        
        Args:
            flow_data: Network flow data
            
        Returns:
            List of text representations
        """
        texts = []
        
        for _, row in flow_data.iterrows():
            # Create text representation from flow characteristics
            text_parts = []
            
            # Protocol information
            if 'Protocol' in row:
                text_parts.append(f"protocol_{str(row['Protocol']).lower()}")
            
            # Port information
            if 'Src Port' in row and 'Dst Port' in row:
                text_parts.append(f"src_port_{row['Src Port']}")
                text_parts.append(f"dst_port_{row['Dst Port']}")
            
            # Flow characteristics
            if 'Flow Duration' in row:
                duration = row['Flow Duration']
                if duration < 1:
                    text_parts.append("short_flow")
                elif duration > 3600:
                    text_parts.append("long_flow")
            
            if 'Packet Length Mean' in row:
                pkt_len = row['Packet Length Mean']
                if pkt_len < 64:
                    text_parts.append("small_packets")
                elif pkt_len > 1400:
                    text_parts.append("large_packets")
            
            # Flag patterns
            flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
                        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
            for col in flag_cols:
                if col in row and row[col] > 0:
                    text_parts.append(f"flag_{col.split()[0].lower()}")
            
            # Combine all parts
            text = " ".join(text_parts)
            texts.append(text)
        
        return texts
    
    def detect_patterns(self, flow_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Detect patterns in flow data using Aho-Corasick algorithm
        
        Args:
            flow_data: Network flow data
            
        Returns:
            Tuple of (detection_scores, detection_details)
        """
        pattern_features = self.extract_pattern_features(flow_data)
        
        # Calculate detection scores
        detection_scores = pattern_features['pattern_score'].values
        
        # Normalize scores
        if detection_scores.max() > 0:
            detection_scores = detection_scores / detection_scores.max()
        
        # Create detection details
        detection_details = {
            'pattern_features': pattern_features,
            'total_patterns': len(self.patterns),
            'matched_patterns': pattern_features['pattern_match_count'].sum()
        }
        
        return detection_scores, detection_details

class ACARandomForestModel:
    """
    Aho-Corasick + Random Forest model for cyber threat detection
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', bootstrap: bool = True,
                 class_weight: str = None):
        """
        Initialize ACA + Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf nodes
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            class_weight: Class weight strategy
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        
        # Initialize Random Forest with enhanced parameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        # Initialize ACA detector
        self.aca_detector = AhoCorasickDetector()
        
        # Initialize feature names
        self.feature_names = []
        
        # Training status
        self.is_trained = False
        
        logger.info(f"ACA + Random Forest model initialized with {n_estimators} estimators, max_depth={max_depth}")
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features by combining original features with ACA pattern features
        
        Args:
            X: Original features
            
        Returns:
            Combined features DataFrame
        """
        # Extract pattern features using ACA
        pattern_features = self.aca_detector.extract_pattern_features(X)
        
        # Combine with original features
        combined_features = pd.concat([X, pattern_features], axis=1)
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        return combined_features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the ACA + RF model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ACA + Random Forest model...")
        
        # Prepare features
        X_combined = self.prepare_features(X_train)
        
        # Load class weights if available
        if class_weights is None:
            try:
                # Try to load from preprocessing directory
                weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         'data', 'processed', 'class_weights.joblib')
                if os.path.exists(weights_path):
                    class_weights = joblib.load(weights_path)
                    logger.info("Loaded class weights from preprocessing")
            except:
                logger.warning("Could not load class weights, using default")
        
        # Train Random Forest with class weights if available
        if class_weights:
            self.rf_classifier.class_weight = class_weights
            logger.info("Using class weights for training")
        
        self.rf_classifier.fit(X_combined, y_train)
        
        # Calculate training metrics
        y_pred_train = self.rf_classifier.predict(X_combined)
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        
        logger.info(f"ACA + RF model training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'ACA_RF',
            'training_metrics': training_metrics,
            'feature_count': len(self.feature_names),
            'pattern_count': len(self.aca_detector.patterns),
            'class_weights_used': bool(class_weights)
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict threats using the ACA + RF model
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_combined = self.prepare_features(X)
        
        # Get Random Forest predictions
        rf_predictions = self.rf_classifier.predict(X_combined)
        rf_probabilities = self.rf_classifier.predict_proba(X_combined)
        
        # Get ACA pattern detection scores
        aca_scores, aca_details = self.aca_detector.detect_patterns(X)
        
        # Combine predictions (retain RF labels; expose ACA influence via details only)
        combined_predictions = rf_predictions.copy()
        high_pattern_threshold = 0.7
        high_pattern_indices = aca_scores > high_pattern_threshold
        
        # Create prediction details
        prediction_details = {
            'rf_predictions': rf_predictions,
            'rf_probabilities': rf_probabilities,
            'aca_scores': aca_scores,
            'aca_details': aca_details,
            'combined_predictions': combined_predictions,
            'high_pattern_detections': np.sum(high_pattern_indices)
        }
        
        return combined_predictions, prediction_details
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions, details = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        # Add detailed metrics
        metrics.update({
            'rf_accuracy': accuracy_score(y_test, details['rf_predictions']),
            'aca_pattern_detections': details['aca_details']['matched_patterns'],
            'high_pattern_detections': details['high_pattern_detections']
        })
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'rf_classifier': self.rf_classifier,
            'feature_names': self.feature_names,
            'aca_patterns': self.aca_detector.patterns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"ACA + RF model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        # Try loading with better error handling for large models
        try:
            model_data = joblib.load(filepath)
        except MemoryError as me:
            logger.error(f"Memory error loading model: {me}")
            logger.warning("Attempting to free memory and retry...")
            import gc
            gc.collect()
            try:
                model_data = joblib.load(filepath)
            except MemoryError as me2:
                logger.error(f"Still unable to load model: {me2}")
                logger.error("Solutions: 1) Free memory, 2) Restart Python, 3) Retrain with smaller model")
                raise MemoryError(f"Unable to allocate memory for model. File: {filepath}. Error: {me2}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        self.rf_classifier = model_data['rf_classifier']
        self.feature_names = model_data['feature_names']
        self.aca_detector.patterns = model_data['aca_patterns']
        self.aca_detector._build_automaton()
        
        self.is_trained = True
        logger.info(f"ACA + RF model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_classifier.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)
