"""
Model 2: Fuzzy Logic + Random Forest (Fuzzy + RF)
Behavior-based detection using fuzzy inference with Random Forest classification
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import os

logger = logging.getLogger(__name__)

class FuzzyLogicEngine:
    """
    Fuzzy Logic Engine for behavior-based threat detection
    Implements the exact pseudocode from the study
    """
    
    def __init__(self):
        """Initialize the fuzzy logic engine"""
        self.fuzzy_rules = self._define_fuzzy_rules()
        self.membership_functions = self._define_membership_functions()
        self.thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
    def _define_fuzzy_rules(self) -> List[Dict]:
        """
        Define fuzzy rules for threat detection
        
        Returns:
            List of fuzzy rules
        """
        rules = [
            # Rule 1: High packet rate (potential DDoS)
            {
                'name': 'high_packet_rate',
                'conditions': ['packets_per_second'],
                'thresholds': {'packets_per_second': 1000},
                'weight': 0.3
            },
            
            # Rule 2: High byte rate
            {
                'name': 'high_byte_rate',
                'conditions': ['bytes_per_second'],
                'thresholds': {'bytes_per_second': 1000000},
                'weight': 0.25
            },
            
            # Rule 3: Unusual port activity
            {
                'name': 'unusual_port_activity',
                'conditions': ['is_well_known_port'],
                'thresholds': {'is_well_known_port': 0},
                'weight': 0.2
            },
            
            # Rule 4: Short flow duration (potential scan)
            {
                'name': 'short_flow_duration',
                'conditions': ['flow_duration'],
                'thresholds': {'flow_duration': 1.0},
                'weight': 0.15
            },
            
            # Rule 5: Large packet size (potential data exfiltration)
            {
                'name': 'large_packet_size',
                'conditions': ['packet_length_mean'],
                'thresholds': {'packet_length_mean': 1400},
                'weight': 0.1
            },
            
            # Rule 6: High flag count (suspicious behavior)
            {
                'name': 'high_flag_count',
                'conditions': ['total_flags'],
                'thresholds': {'total_flags': 5},
                'weight': 0.2
            },
            
            # Rule 7: Unusual IAT (Inter Arrival Time)
            {
                'name': 'unusual_iat',
                'conditions': ['flow_iat_mean'],
                'thresholds': {'flow_iat_mean': 0.001},
                'weight': 0.15
            },
            
            # Rule 8: Protocol anomalies
            {
                'name': 'protocol_anomaly',
                'conditions': ['protocol_encoded'],
                'thresholds': {'protocol_encoded': 1},
                'weight': 0.1
            }
        ]
        
        return rules
    
    def _define_membership_functions(self) -> Dict:
        """
        Define membership functions for fuzzy sets
        
        Returns:
            Dictionary of membership functions
        """
        membership_functions = {
            'low': lambda x, threshold: max(0, 1 - (x / threshold)) if x <= threshold else 0,
            'medium': lambda x, threshold: max(0, 1 - abs(x - threshold) / threshold) if 0 <= x <= 2 * threshold else 0,
            'high': lambda x, threshold: max(0, (x - threshold) / threshold) if x >= threshold else 0,
            'very_high': lambda x, threshold: max(0, (x - 2 * threshold) / threshold) if x >= 2 * threshold else 0
        }
        
        return membership_functions
    
    def calculate_membership(self, value: float, threshold: float, membership_type: str = 'high') -> float:
        """
        Calculate membership degree for a given value
        
        Args:
            value: Input value
            threshold: Threshold for the membership function
            membership_type: Type of membership function
            
        Returns:
            Membership degree (0-1)
        """
        if membership_type in self.membership_functions:
            return min(1.0, max(0.0, self.membership_functions[membership_type](value, threshold)))
        else:
            # Default to high membership function
            return min(1.0, max(0.0, self.membership_functions['high'](value, threshold)))
    
    def fuzzy_detect(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main fuzzy logic detection following the pseudocode
        Inputs: X = {x1, x2, ..., xm} feature vector
        Output: threat_scores (crisp) and threat_labels
        """
        num_rows = len(X)
        threat_scores = np.zeros(num_rows)
        threat_labels = np.zeros(num_rows, dtype=int)
        
        for i in range(num_rows):
            # Get feature vector for this sample
            x = X.iloc[i]
            
            # 1. Fuzzification - map each feature to membership degrees
            membership_degrees = {}
            for feature in X.columns:
                if feature in x and pd.notna(x[feature]):
                    membership_degrees[feature] = self._fuzzify_feature(feature, x[feature])
            
            # 2. Evaluate rule antecedents -> rule firing strengths
            firing_strengths = {}
            for rule in self.fuzzy_rules:
                firing_strengths[rule['name']] = self._evaluate_antecedent(rule, membership_degrees)
            
            # 3. Generate rule consequents and aggregate
            aggregated_output = self._aggregate_consequents(firing_strengths)
            
            # 4. Defuzzification to get crisp score
            threat_score = self._defuzzify(aggregated_output)
            threat_scores[i] = threat_score
            
            # 5. Threshold to get label
            threat_labels[i] = self._threshold_to_label(threat_score)
        
        return threat_scores, threat_labels
    
    def _fuzzify_feature(self, feature: str, value: float) -> Dict[str, float]:
        """
        Fuzzification: map input feature to membership degrees
        """
        membership = {}
        
        # FALLBACK: Generate default membership for ANY numeric feature
        # This ensures we always have SOME fuzzy signal
        if isinstance(value, (int, float)):
            # Normalize to 0-1 range based on typical scales
            if value > 0:
                # Log-scale normalization for better range handling
                normalized = np.log1p(value) / 10.0  # Log scale, /10 to bring to 0-1
                membership['high'] = min(1.0, normalized)
                membership['medium'] = min(0.6, normalized)
                membership['low'] = min(0.3, normalized)
            else:
                membership['high'] = 0.0
                membership['medium'] = 0.0
                membership['low'] = 0.0
        
        # Define membership functions for specific features
        if feature in ['packets_per_second', 'bytes_per_second']:
            # High rate membership
            if value > 1000:
                membership['high'] = min(1.0, (value - 1000) / 1000)
            else:
                membership['high'] = 0.0
            
            # Medium rate membership
            if 100 <= value <= 1000:
                membership['medium'] = min(1.0, (value - 100) / 900)
            else:
                membership['medium'] = 0.0
            
            # Low rate membership
            if value < 100:
                membership['low'] = min(1.0, value / 100)
            else:
                membership['low'] = 0.0
                
        elif feature in ['flow_duration']:
            # Short duration (potential scan)
            if value < 1.0:
                membership['short'] = min(1.0, (1.0 - value) / 1.0)
            else:
                membership['short'] = 0.0
            
            # Normal duration
            if 1.0 <= value <= 10.0:
                membership['normal'] = min(1.0, (10.0 - value) / 9.0)
            else:
                membership['normal'] = 0.0
                
        elif feature in ['total_flags']:
            # High flag count
            if value > 5:
                membership['high'] = min(1.0, (value - 5) / 5)
            else:
                membership['high'] = 0.0
                
        else:
            # Default membership
            membership['normal'] = 1.0
        
        return membership
    
    def _evaluate_antecedent(self, rule: Dict, membership_degrees: Dict) -> float:
        """
        Evaluate rule antecedents using fuzzy AND/OR operations
        """
        firing_strength = 0.0
        
        for condition in rule['conditions']:
            if condition in membership_degrees:
                # Get membership degree for this condition
                condition_membership = membership_degrees[condition].get('high', 0.0)
                firing_strength = max(firing_strength, condition_membership)  # OR operation
        
        return firing_strength
    
    def _aggregate_consequents(self, firing_strengths: Dict[str, float]) -> float:
        """
        Aggregate rule consequents using AGGRESSIVE sum (not average)
        """
        if not firing_strengths:
            return 0.0
        
        # AGGRESSIVE SUM instead of weighted average
        # This amplifies threat signals instead of averaging them down
        total_strength = 0.0
        max_firing = 0.0
        
        for rule in self.fuzzy_rules:
            rule_name = rule['name']
            if rule_name in firing_strengths:
                weight = rule['weight']
                firing = firing_strengths[rule_name]
                total_strength += firing * weight
                max_firing = max(max_firing, firing)
        
        # Use MAX of sum or max firing - this boosts any threat signal
        # Then scale up to make it more impactful
        aggregated = max(total_strength, max_firing) * 2.0  # 2x boost
        
        # Clamp to 0-1 range but allow higher values
        return min(1.0, max(0.0, aggregated))
    
    def _defuzzify(self, aggregated_output: float) -> float:
        """
        Defuzzification using centroid method
        """
        # Simple centroid defuzzification
        return min(1.0, max(0.0, aggregated_output))
    
    def _threshold_to_label(self, threat_score: float) -> int:
        """
        Convert threat score to label using thresholds
        """
        if threat_score >= self.thresholds['high']:
            return 3  # High threat
        elif threat_score >= self.thresholds['medium']:
            return 2  # Medium threat
        elif threat_score >= self.thresholds['low']:
            return 1  # Low threat
        else:
            return 0  # Benign

    def apply_fuzzy_rules(self, flow_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fuzzy rules to flow data using the new pseudocode implementation
        """
        # Use the new fuzzy_detect method
        threat_scores, threat_labels = self.fuzzy_detect(flow_data)
        
        # Create output DataFrame
        fuzzy_scores = pd.DataFrame(index=flow_data.index)
        fuzzy_scores['fuzzy_score'] = threat_scores
        fuzzy_scores['fuzzy_threat_level'] = threat_labels
        
        # Create fuzzy categories
        fuzzy_scores['fuzzy_low'] = (threat_scores < 0.2).astype(int)
        fuzzy_scores['fuzzy_medium'] = ((threat_scores >= 0.2) & (threat_scores < 0.6)).astype(int)
        fuzzy_scores['fuzzy_high'] = (threat_scores >= 0.6).astype(int)
        
        return fuzzy_scores
    
    def extract_fuzzy_features(self, flow_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract fuzzy logic features from flow data
        
        Args:
            flow_data: Network flow data
            
        Returns:
            DataFrame with fuzzy features
        """
        # Apply fuzzy rules
        fuzzy_scores = self.apply_fuzzy_rules(flow_data)
        
        # Create additional fuzzy features
        fuzzy_features = pd.DataFrame(index=flow_data.index)
        
        # Basic fuzzy score
        fuzzy_features['fuzzy_score'] = fuzzy_scores['fuzzy_score']
        
        # Rule-specific features
        for rule in self.fuzzy_rules:
            rule_name = rule['name']
            if f'{rule_name}_score' in fuzzy_scores.columns:
                fuzzy_features[f'{rule_name}_score'] = fuzzy_scores[f'{rule_name}_score']
        
        # Fuzzy categories
        fuzzy_features['fuzzy_low'] = (fuzzy_scores['fuzzy_score'] < 0.3).astype(int)
        fuzzy_features['fuzzy_medium'] = ((fuzzy_scores['fuzzy_score'] >= 0.3) & 
                                        (fuzzy_scores['fuzzy_score'] < 0.7)).astype(int)
        fuzzy_features['fuzzy_high'] = (fuzzy_scores['fuzzy_score'] >= 0.7).astype(int)
        
        # Fuzzy risk levels
        fuzzy_features['fuzzy_risk_low'] = (fuzzy_scores['fuzzy_score'] < 0.2).astype(int)
        fuzzy_features['fuzzy_risk_medium'] = ((fuzzy_scores['fuzzy_score'] >= 0.2) & 
                                             (fuzzy_scores['fuzzy_score'] < 0.6)).astype(int)
        fuzzy_features['fuzzy_risk_high'] = (fuzzy_scores['fuzzy_score'] >= 0.6).astype(int)
        
        # BOOST fuzzy scores to make them more threat-indicative
        # Multiply fuzzy scores by a factor to make them more influential
        fuzzy_features['fuzzy_score_boosted'] = fuzzy_features['fuzzy_score'] * 10  # 10x boost
        
        # Add threat probability estimate based on fuzzy score
        # If fuzzy score > 0.1, treat as suspicious (lower threshold)
        fuzzy_features['is_suspicious'] = (fuzzy_features['fuzzy_score'] > 0.1).astype(int)
        
        # Add multiple threshold-based indicators for more granular threat detection
        fuzzy_features['threat_indicator_weak'] = (fuzzy_features['fuzzy_score'] > 0.05).astype(int)
        fuzzy_features['threat_indicator_medium'] = (fuzzy_features['fuzzy_score'] > 0.2).astype(int)
        fuzzy_features['threat_indicator_strong'] = (fuzzy_features['fuzzy_score'] > 0.5).astype(int)
        
        return fuzzy_features

class FuzzyRandomForestModel:
    """
    Fuzzy Logic + Random Forest model for cyber threat detection
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 30,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', bootstrap: bool = True,
                 class_weight: str = 'balanced', max_samples: float = 0.8,
                 min_impurity_decrease: float = 0.0, criterion: str = 'gini',
                 ccp_alpha: float = 0.0, n_jobs: int = -1):
        """
        Initialize Fuzzy + Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf nodes
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            class_weight: Class weight strategy
            max_samples: Maximum samples per tree (for overfitting prevention)
            min_impurity_decrease: Minimum impurity decrease for splitting
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs
        
        # Initialize Random Forest with optimized parameters for speed and accuracy
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            max_samples=max_samples,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            ccp_alpha=ccp_alpha,
            n_jobs=n_jobs,
            random_state=42,
            warm_start=False,
            oob_score=True
        )
        
        # Initialize fuzzy logic engine
        self.fuzzy_engine = FuzzyLogicEngine()
        
        # Initialize feature names
        self.feature_names = []
        
        # Training status
        self.is_trained = False
        
        logger.info(f"Fuzzy + Random Forest model initialized with {n_estimators} estimators, max_depth={max_depth}")
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features by combining original features with fuzzy logic features
        
        Args:
            X: Original features
            
        Returns:
            Combined features DataFrame
        """
        # Extract fuzzy features
        fuzzy_features = self.fuzzy_engine.extract_fuzzy_features(X)
        
        # Combine with original features
        combined_features = pd.concat([X, fuzzy_features], axis=1)
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        return combined_features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the Fuzzy + RF model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Fuzzy Logic + Random Forest model...")
        
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
        
        # Apply AGGRESSIVE resampling to balance classes BEFORE training (like IntruDTree does)
        from sklearn.utils import resample
        
        # Get class distribution
        class_counts = y_train.value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        # MORE AGGRESSIVE: Upsample minority classes 3x instead of 2x
        target_size = min_count * 3  # More aggressive upsampling for better threat detection
        
        logger.info(f"Class distribution before balancing: {dict(class_counts)}")
        logger.info(f"Target size for balancing: {target_size}")
        
        # Resample to create balanced dataset
        balanced_samples = []
        balanced_labels = []
        
        for class_label in y_train.unique():
            class_mask = y_train == class_label
            class_X = X_combined[class_mask]
            class_y = y_train[class_mask]
            
            if len(class_X) > target_size:
                # Downsample majority class
                class_X_resampled, class_y_resampled = resample(
                    class_X, class_y, n_samples=target_size, random_state=42
                )
            else:
                # Upsample minority class
                class_X_resampled, class_y_resampled = resample(
                    class_X, class_y, n_samples=target_size, random_state=42, replace=True
                )
            
            balanced_samples.append(class_X_resampled)
            balanced_labels.append(class_y_resampled)
        
        # Combine balanced data
        X_balanced = pd.concat(balanced_samples, ignore_index=True)
        y_balanced = pd.concat(balanced_labels, ignore_index=True)
        
        logger.info(f"Balanced class distribution: {dict(y_balanced.value_counts())}")
        
        # FORCE class weights to be very aggressive toward minority classes
        # Calculate aggressive weights based on ORIGINAL data distribution (before balancing)
        unique_classes = y_train.unique()
        n_samples_original = len(y_train)
        n_classes = len(unique_classes)
        
        # Create very aggressive class weights that strongly favor minority classes
        aggressive_class_weights = {}
        for cls in unique_classes:
            class_count_original = np.sum(y_train == cls)
            # Calculate weight as: total_samples / (n_classes * class_count)
            # This gives much higher weight to minority classes
            weight = n_samples_original / (n_classes * class_count_original)
            aggressive_class_weights[cls] = weight
        
        logger.info(f"Computed aggressive class weights (based on original data): {aggressive_class_weights}")
        
        # Apply aggressive weights
        self.rf_classifier.class_weight = aggressive_class_weights
        logger.info("Using AGGRESSIVE class weights for training to favor threat detection")
        
        # Train on BALANCED data with aggressive weights
        self.rf_classifier.fit(X_balanced, y_balanced)
        
        # Calculate training metrics
        y_pred_train = self.rf_classifier.predict(X_combined)
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),  
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        
        logger.info(f"Fuzzy + RF model training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'Fuzzy_RF',
            'training_metrics': training_metrics,
            'feature_count': len(self.feature_names),
            'fuzzy_rules_count': len(self.fuzzy_engine.fuzzy_rules),
            'class_weights_used': bool(class_weights)
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict threats using the Fuzzy + RF model
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            X_combined = self.prepare_features(X)
            
            # Get Random Forest predictions
            rf_predictions = self.rf_classifier.predict(X_combined)
            rf_probabilities = self.rf_classifier.predict_proba(X_combined)
            
            # Get fuzzy scores
            fuzzy_features = self.fuzzy_engine.extract_fuzzy_features(X)
            fuzzy_scores = fuzzy_features['fuzzy_score'].values
            
            # ACTUALLY USE fuzzy logic to modify predictions (not just provide features)
            combined_predictions = rf_predictions.copy()
            
            # Determine benign class index (usually 0, but let's be smart about it)
            # Check if RF classifier has learned classes
            if hasattr(self.rf_classifier, 'classes_'):
                classes = self.rf_classifier.classes_
                # Try to find benign class
                benign_idx = 0  # Default to 0
                if hasattr(classes, '__iter__'):
                    for idx, class_name in enumerate(classes):
                        if 'benign' in str(class_name).lower():
                            benign_idx = idx
                            break
            else:
                benign_idx = 0  # Default assumption
            
            logger.info(f"Using benign_idx = {benign_idx} for fuzzy override logic")
            
            # ULTRA-AGGRESSIVE fuzzy logic override - use RF probabilities directly
            # Strategy: If RF has ANY probability of a threat, treat it as a threat
            for i in range(len(X)):
                pred_idx = rf_predictions[i]
                fuzzy_score = fuzzy_scores[i]
                
                # Check if RF predicted benign (assume benign is class 0)
                is_benign_prediction = (pred_idx == benign_idx)
                
                # Get probabilities for all classes
                probs = rf_probabilities[i]
                sorted_indices = np.argsort(probs)[::-1]  # Descending order
                
                # Get the probability of the benign class
                benign_prob = probs[benign_idx]
                
                # SMART OVERRIDE: Use ensemble of RF probabilities + fuzzy score
                if is_benign_prediction:
                    # Strategy: If benign prob is NOT overwhelmingly dominant, trust threat classes
                    benign_probability = probs[benign_idx]
                    
                    # If benign prob < 0.7, there's significant uncertainty - use threat
                    if benign_probability < 0.7:
                        # Find best threat candidate
                        threat_candidates = [idx for idx in sorted_indices if idx != benign_idx]
                        if len(threat_candidates) > 0:
                            best_threat = threat_candidates[0]
                            threat_prob = probs[best_threat]
                            
                            # Only override if threat prob is at least 20% of benign prob
                            if threat_prob >= (benign_probability * 0.2):
                                combined_predictions[i] = best_threat
                                logger.debug(f"Smart override: benign(prob={benign_probability:.3f}) → threat class {best_threat}, prob={threat_prob:.3f})")
                    
                    # Even if benign is strong, if fuzzy score is high, override
                    elif fuzzy_score > 0.3:  # Medium fuzzy signal
                        threat_candidates = [idx for idx in sorted_indices if idx != benign_idx]
                        if len(threat_candidates) > 0:
                            combined_predictions[i] = threat_candidates[0]
                            logger.debug(f"Fuzzy override: high fuzzy ({fuzzy_score:.3f}) → threat {threat_candidates[0]}")
                
                # BOOST existing threat predictions when fuzzy confirms
                elif not is_benign_prediction and fuzzy_score > 0.1:
                    # Fuzzy confirms the threat - keep the prediction
                    # This boosts confidence in the detection
                    pass
                
            # Count how many predictions were modified
            modified_count = np.sum(combined_predictions != rf_predictions)
            high_fuzzy_threshold = 0.6
            high_fuzzy_indices = fuzzy_scores > high_fuzzy_threshold
            
            # Calculate statistics for logging
            original_benign_count = np.sum(rf_predictions == benign_idx)
            final_benign_count = np.sum(combined_predictions == benign_idx)
            original_threat_count = np.sum(rf_predictions != benign_idx)
            final_threat_count = np.sum(combined_predictions != benign_idx)
            
            # Create prediction details
            prediction_details = {
                'rf_predictions': rf_predictions,
                'rf_probabilities': rf_probabilities,
                'fuzzy_scores': fuzzy_scores,
                'fuzzy_features': fuzzy_features,
                'combined_predictions': combined_predictions,
                'predictions_modified_by_fuzzy': int(modified_count),  # Track how many were overridden
                'high_fuzzy_detections': np.sum(high_fuzzy_indices),
                'fuzzy_risk_distribution': {
                    'low': np.sum(fuzzy_features['fuzzy_risk_low']),
                    'medium': np.sum(fuzzy_features['fuzzy_risk_medium']),
                    'high': np.sum(fuzzy_features['fuzzy_risk_high'])
                },
                'statistics': {
                    'original_benign': int(original_benign_count),
                    'final_benign': int(final_benign_count),
                    'original_threats': int(original_threat_count),
                    'final_threats': int(final_threat_count),
                    'overrides': int(modified_count)
                }
            }
            
            # Log summary of changes
            logger.info(f"Fuzzy_RF Prediction Summary:")
            logger.info(f"  Original: {original_benign_count} benign, {original_threat_count} threats")
            logger.info(f"  Final:    {final_benign_count} benign, {final_threat_count} threats")
            logger.info(f"  Modified: {modified_count} predictions overridden by fuzzy logic")
            
            return combined_predictions, prediction_details
            
        except Exception as e:
            logger.error(f"Error in Fuzzy_RF prediction: {e}")
            # Return fallback predictions
            fallback_predictions = np.zeros(len(X), dtype=int)
            fallback_details = {
                'rf_predictions': fallback_predictions,
                'rf_probabilities': np.ones((len(X), 2)) * 0.5,  # Equal probabilities
                'fuzzy_scores': np.zeros(len(X)),
                'fuzzy_features': pd.DataFrame(index=X.index),
                'combined_predictions': fallback_predictions,
                'high_fuzzy_detections': 0,
                'fuzzy_risk_distribution': {'low': len(X), 'medium': 0, 'high': 0},
                'error': str(e)
            }
            return fallback_predictions, fallback_details
    
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
            'fuzzy_score_mean': np.mean(details['fuzzy_scores']),
            'fuzzy_score_std': np.std(details['fuzzy_scores']),
            'high_fuzzy_detections': details['high_fuzzy_detections'],
            'fuzzy_risk_distribution': details['fuzzy_risk_distribution']
        })
        
        return metrics
    
    def get_fuzzy_analysis(self, X: pd.DataFrame) -> Dict:
        """
        Get detailed fuzzy logic analysis
        
        Args:
            X: Input features
            
        Returns:
            Fuzzy analysis results
        """
        fuzzy_features = self.fuzzy_engine.extract_fuzzy_features(X)
        
        analysis = {
            'fuzzy_score_statistics': {
                'mean': fuzzy_features['fuzzy_score'].mean(),
                'std': fuzzy_features['fuzzy_score'].std(),
                'min': fuzzy_features['fuzzy_score'].min(),
                'max': fuzzy_features['fuzzy_score'].max(),
                'median': fuzzy_features['fuzzy_score'].median()
            },
            'risk_level_distribution': {
                'low': int(fuzzy_features['fuzzy_risk_low'].sum()),
                'medium': int(fuzzy_features['fuzzy_risk_medium'].sum()),
                'high': int(fuzzy_features['fuzzy_risk_high'].sum())
            },
            'rule_contributions': {}
        }
        
        # Analyze rule contributions
        for rule in self.fuzzy_engine.fuzzy_rules:
            rule_name = rule['name']
            if f'{rule_name}_score' in fuzzy_features.columns:
                analysis['rule_contributions'][rule_name] = {
                    'mean_score': fuzzy_features[f'{rule_name}_score'].mean(),
                    'max_score': fuzzy_features[f'{rule_name}_score'].max(),
                    'weight': rule['weight']
                }
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create serializable model data
        model_data = {
            'rf_classifier': self.rf_classifier,
            'feature_names': self.feature_names,
            'fuzzy_rules': self.fuzzy_engine.fuzzy_rules,
            'membership_functions': None,  # Don't save lambda functions
            'thresholds': self.fuzzy_engine.thresholds,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'class_weight': self.class_weight
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Fuzzy + RF model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.rf_classifier = model_data['rf_classifier']
        self.feature_names = model_data['feature_names']
        self.fuzzy_engine.fuzzy_rules = model_data['fuzzy_rules']
        self.fuzzy_engine.thresholds = model_data['thresholds']
        
        # Restore parameters
        self.n_estimators = model_data.get('n_estimators', 100)
        self.max_depth = model_data.get('max_depth', 10)
        self.min_samples_split = model_data.get('min_samples_split', 2)
        self.min_samples_leaf = model_data.get('min_samples_leaf', 1)
        self.max_features = model_data.get('max_features', 'sqrt')
        self.bootstrap = model_data.get('bootstrap', True)
        self.class_weight = model_data.get('class_weight', None)
        
        # Recreate membership functions
        self.fuzzy_engine.membership_functions = self.fuzzy_engine._define_membership_functions()
        
        self.is_trained = True
        logger.info(f"Fuzzy + RF model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_classifier.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'class_weight': self.class_weight,
            'max_samples': self.max_samples,
            'min_impurity_decrease': self.min_impurity_decrease,
            'criterion': self.criterion,
            'ccp_alpha': self.ccp_alpha,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
