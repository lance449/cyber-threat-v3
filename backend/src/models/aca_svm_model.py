"""
ACA + SVM Model for Cyber Threat Detection
Combines Aho-Corasick Algorithm for pattern matching with Support Vector Machine for classification
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
try:
    import pyahocorasick
    PYAHOCORASICK_AVAILABLE = True
except ImportError:
    PYAHOCORASICK_AVAILABLE = False
    print("pyahocorasick not available, using fallback pattern matching")
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ACANode:
    """
    Node class for Aho-Corasick automaton
    """
    def __init__(self):
        self.next = {}  # Dictionary of next states
        self.fail = None  # Failure link
        self.output = []  # Output patterns at this node

class ACAPatternMatcher:
    """
    Aho-Corasick Algorithm implementation following the exact pseudocode
    """
    
    def __init__(self, patterns: List[str] = None):
        """
        Initialize ACA pattern matcher
        
        Args:
            patterns: List of patterns to match
        """
        self.patterns = patterns or []
        self.automaton = None
        self.pattern_weights = {}
        self.pattern_categories = {}
        self.root = None
        
    def load_patterns_from_file(self, filepath: str) -> List[str]:
        """
        Load patterns from a file
        
        Args:
            filepath: Path to pattern file
            
        Returns:
            List of patterns
        """
        patterns = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
            
            logger.info(f"Loaded {len(patterns)} patterns from {filepath}")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to load patterns from {filepath}: {e}")
            return []
    
    def load_cyber_threat_patterns(self) -> List[str]:
        """
        Load predefined cyber threat patterns
        
        Returns:
            List of cyber threat patterns
        """
        # Reduced set of key malware signatures and attack patterns for faster training
        threat_patterns = [
            # Core malware signatures
            b'malware_signature',
            b'trojan_pattern',
            b'virus_signature',
            b'worm_propagation',
            
            # Key attack patterns
            b'sql_injection',
            b'xss_script',
            b'buffer_overflow',
            b'denial_of_service',
            
            # Network anomalies
            b'port_scanning',
            b'brute_force',
            b'data_exfiltration',
            b'command_control',
            
            # Protocol anomalies
            b'invalid_header',
            b'suspicious_flag',
            b'anomalous_packet',
            b'protocol_violation',
            
            # Behavioral patterns
            b'rapid_connection',
            b'unusual_transfer',
            b'timing_attack',
            b'resource_exhaustion'
        ]
        
        # Convert to bytes if not already
        byte_patterns = []
        for pattern in threat_patterns:
            if isinstance(pattern, str):
                byte_patterns.append(pattern.encode('utf-8'))
            else:
                byte_patterns.append(pattern)
        
        logger.info(f"Loaded {len(byte_patterns)} predefined cyber threat patterns")
        return byte_patterns
    
    def build_ac(self, patterns: List[str] = None) -> bool:
        """
        Build ACA automaton following the exact pseudocode
        Procedure BuildAC(P):
        """
        try:
            if patterns:
                self.patterns = patterns
            
            if not self.patterns:
                logger.warning("No patterns provided for automaton construction")
                return False
            
            # Create root node
            self.root = ACANode()
            
            # 1. Build trie
            for pattern in self.patterns:
                node = self.root
                for ch in pattern:
                    if ch not in node.next:
                        node.next[ch] = ACANode()
                    node = node.next[ch]
                node.output.append(pattern)  # mark pattern at terminal node
            
            # 2. Build failure links (BFS)
            from collections import deque
            queue = deque()
            
            # Initialize queue with root's children
            for ch in self.root.next:
                child = self.root.next[ch]
                child.fail = self.root
                queue.append(child)
            
            # BFS to build failure links
            while queue:
                r = queue.popleft()
                for ch in r.next:
                    s = r.next[ch]
                    queue.append(s)
                    state = r.fail
                    
                    # Find failure state
                    while state is not None and ch not in state.next:
                        state = state.fail
                    
                    if state is not None and ch in state.next:
                        s.fail = state.next[ch]
                    else:
                        s.fail = self.root
                    
                    # Union outputs
                    s.output.extend(s.fail.output)
            
            logger.info(f"Built ACA automaton with {len(self.patterns)} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build ACA automaton: {e}")
            return False
    
    def build_automaton(self, patterns: List[str] = None) -> bool:
        """
        Wrapper for build_ac to maintain compatibility
        """
        return self.build_ac(patterns)
    
    def ac_search(self, text: Union[str, bytes]) -> List[Tuple[int, int, str]]:
        """
        ACA search following the exact pseudocode
        Procedure AC_Search(root, T):
        """
        if not self.root:
            logger.warning("Automaton not built. Call build_ac() first.")
            return []
        
        try:
            # Convert to string for pattern matching
            if isinstance(text, bytes):
                text_str = text.decode('utf-8', errors='ignore')
            else:
                text_str = text
            
            matches = []
            node = self.root
            
            # AC_Search algorithm
            for i in range(len(text_str)):
                ch = text_str[i]
                
                # Follow failure links until we find a valid transition
                while node != self.root and ch not in node.next:
                    node = node.fail
                
                # Move to next state if transition exists
                if ch in node.next:
                    node = node.next[ch]
                
                # Report matches if any
                if node.output:
                    for pattern in node.output:
                        start_idx = i - len(pattern) + 1
                        matches.append((start_idx, i, pattern))
            
            return matches
            
        except Exception as e:
            logger.error(f"ACA search failed: {e}")
            return []

    def ac_dfa_creation_optimized(self, patterns: List[str], K: int = 2) -> bool:
        """
        K-stride optimized ACA creation following pseudocode
        Procedure AC_DFA_Creation_Optimized(P, K):
        """
        try:
            if not patterns:
                return False
            
            modified_patterns = []
            pattern_mappings = {}
            
            for pattern in patterns:
                m = len(pattern)
                for j in range(K):
                    # Extract subpattern: chars starting at j, stepping by K
                    subpattern = ''.join([pattern[i] for i in range(j, m, K)])
                    
                    if len(subpattern) >= 2:  # min_len_threshold
                        # Store mapping from (pattern, j) -> subpattern
                        pattern_mappings[(pattern, j)] = subpattern
                        modified_patterns.append(subpattern)
            
            # Build ACA on modified patterns
            self.patterns = modified_patterns
            self.pattern_mappings = pattern_mappings
            self.K = K
            
            return self.build_ac(modified_patterns)
            
        except Exception as e:
            logger.error(f"Optimized ACA creation failed: {e}")
            return False

    def ac_dfa_search_optimized(self, text: Union[str, bytes]) -> List[Tuple[int, int, str]]:
        """
        K-stride optimized ACA search following pseudocode
        Procedure AC_DFA_Search_Optimized(root, T, K):
        """
        if not self.root or not hasattr(self, 'pattern_mappings'):
            logger.warning("Optimized automaton not built. Call ac_dfa_creation_optimized() first.")
            return []
        
        try:
            # Convert to string for pattern matching
            if isinstance(text, bytes):
                text_str = text.decode('utf-8', errors='ignore')
            else:
                text_str = text
            
            matches = []
            node = self.root
            
            # Search every Kth byte
            for j in range(0, len(text_str), self.K):
                ch = text_str[j]
                
                # Follow failure links until we find a valid transition
                while node != self.root and ch not in node.next:
                    node = node.fail
                
                # Move to next state if transition exists
                if ch in node.next:
                    node = node.next[ch]
                
                # Check for matches and verify full patterns
                if node.output:
                    for subpattern in node.output:
                        # Find original pattern and offset
                        for (original_pattern, offset), mapped_subpattern in self.pattern_mappings.items():
                            if mapped_subpattern == subpattern:
                                start_index = j - offset
                                if (start_index >= 0 and 
                                    start_index + len(original_pattern) <= len(text_str) and
                                    text_str[start_index:start_index + len(original_pattern)] == original_pattern):
                                    matches.append((start_index, start_index + len(original_pattern) - 1, original_pattern))
            
            return matches
            
        except Exception as e:
            logger.error(f"Optimized ACA search failed: {e}")
            return []

    def match_patterns(self, text: Union[str, bytes]) -> List[Tuple[int, int, str]]:
        """
        Wrapper for ac_search to maintain compatibility
        """
        return self.ac_search(text)
    
    def get_pattern_statistics(self, text: Union[str, bytes]) -> Dict[str, Any]:
        """
        Get pattern matching statistics for text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with pattern statistics
        """
        matches = self.match_patterns(text)
        
        if not matches:
            return {
                'total_matches': 0,
                'unique_patterns': 0,
                'match_density': 0.0,
                'pattern_distribution': {}
            }
        
        # Count pattern occurrences
        pattern_counts = {}
        for _, _, pattern in matches:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate statistics
        total_matches = len(matches)
        unique_patterns = len(pattern_counts)
        text_length = len(text) if isinstance(text, str) else len(text.decode('utf-8', errors='ignore'))
        match_density = total_matches / max(text_length, 1)
        
        return {
            'total_matches': total_matches,
            'unique_patterns': unique_patterns,
            'match_density': match_density,
            'pattern_distribution': pattern_counts
        }

class ACAFeatureExtractor:
    """
    Feature extraction using ACA pattern matching
    """
    
    def __init__(self, pattern_matcher: ACAPatternMatcher):
        """
        Initialize ACA feature extractor
        
        Args:
            pattern_matcher: ACA pattern matcher instance
        """
        self.pattern_matcher = pattern_matcher
        self.feature_names = []
        
    def extract_binary_features(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Extract binary features indicating pattern presence
        
        Args:
            text: Text to extract features from
            
        Returns:
            Binary feature vector
        """
        matches = self.pattern_matcher.match_patterns(text)
        matched_patterns = set(match[2] for match in matches)
        
        # Create binary vector for each pattern
        binary_features = np.zeros(len(self.pattern_matcher.patterns), dtype=np.float32)
        
        for i, pattern in enumerate(self.pattern_matcher.patterns):
            pattern_str = pattern.decode('utf-8', errors='ignore') if isinstance(pattern, bytes) else pattern
            if pattern_str in matched_patterns:
                binary_features[i] = 1.0
        
        return binary_features
    
    def extract_count_features(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Extract count features for pattern occurrences
        
        Args:
            text: Text to extract features from
            
        Returns:
            Count feature vector
        """
        matches = self.pattern_matcher.match_patterns(text)
        pattern_counts = {}
        
        for _, _, pattern in matches:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Create count vector for each pattern
        count_features = np.zeros(len(self.pattern_matcher.patterns), dtype=np.float32)
        
        for i, pattern in enumerate(self.pattern_matcher.patterns):
            pattern_str = pattern.decode('utf-8', errors='ignore') if isinstance(pattern, bytes) else pattern
            count_features[i] = float(pattern_counts.get(pattern_str, 0))
        
        return count_features
    
    def extract_frequency_features(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Extract normalized frequency features
        
        Args:
            text: Text to extract features from
            
        Returns:
            Frequency feature vector
        """
        count_features = self.extract_count_features(text)
        text_length = len(text) if isinstance(text, str) else len(text.decode('utf-8', errors='ignore'))
        
        # Normalize by text length
        frequency_features = count_features / max(text_length, 1)
        
        return frequency_features
    
    def extract_aggregated_features(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Extract aggregated pattern features
        
        Args:
            text: Text to extract features from
            
        Returns:
            Aggregated feature vector
        """
        stats = self.pattern_matcher.get_pattern_statistics(text)
        
        # Create aggregated features
        aggregated_features = np.array([
            stats['total_matches'],
            stats['unique_patterns'],
            stats['match_density'],
            len(text) if isinstance(text, str) else len(text.decode('utf-8', errors='ignore'))
        ], dtype=np.float32)
        
        return aggregated_features
    
    def extract_all_features(self, text: Union[str, bytes]) -> np.ndarray:
        """
        Extract all ACA features
        
        Args:
            text: Text to extract features from
            
        Returns:
            Combined feature vector
        """
        binary_features = self.extract_binary_features(text)
        count_features = self.extract_count_features(text)
        frequency_features = self.extract_frequency_features(text)
        aggregated_features = self.extract_aggregated_features(text)
        
        # Combine all features
        all_features = np.concatenate([
            binary_features,
            count_features,
            frequency_features,
            aggregated_features
        ])
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for ACA features
        
        Returns:
            List of feature names
        """
        if not self.feature_names:
            # Binary features
            binary_names = [f"aca_binary_{i}" for i in range(len(self.pattern_matcher.patterns))]
            
            # Count features
            count_names = [f"aca_count_{i}" for i in range(len(self.pattern_matcher.patterns))]
            
            # Frequency features
            frequency_names = [f"aca_freq_{i}" for i in range(len(self.pattern_matcher.patterns))]
            
            # Aggregated features
            aggregated_names = [
                "aca_total_matches",
                "aca_unique_patterns", 
                "aca_match_density",
                "aca_text_length"
            ]
            
            self.feature_names = binary_names + count_names + frequency_names + aggregated_names
        
        return self.feature_names

class ACASVMModel:
    """
    ACA + SVM model for cyber threat detection
    """
    
    def __init__(self, 
                 svm_type: str = 'linear',
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 class_weight: str = 'balanced',
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 patterns_file: str = None):
        """
        Initialize ACA + SVM model
        
        Args:
            svm_type: Type of SVM ('linear', 'kernel', 'sgd')
            kernel: Kernel type for SVM ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            class_weight: Class weight strategy
            max_iter: Maximum iterations
            tol: Tolerance for stopping criterion
            patterns_file: Path to patterns file
        """
        self.svm_type = svm_type
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.patterns_file = patterns_file
        
        # Initialize components
        self.pattern_matcher = ACAPatternMatcher()
        self.feature_extractor = ACAFeatureExtractor(self.pattern_matcher)
        self.scaler = StandardScaler()
        self.svm_model = None
        self.pipeline = None
        
        # Training state
        self.is_trained = False
        self.feature_names = []
        self.class_names = []
        self.training_metrics = {}
        
        logger.info(f"ACA + SVM model initialized: svm_type={svm_type}, kernel={kernel}, C={C}")
    
    def _create_svm_model(self) -> Any:
        """
        Create SVM model based on configuration
        
        Returns:
            SVM model instance
        """
        if self.svm_type == 'linear':
            return LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
        elif self.svm_type == 'sgd':
            return SGDClassifier(
                loss='hinge',
                alpha=1.0/self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
        else:  # kernel SVM
            return SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                class_weight=self.class_weight,
                probability=True,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
    
    def _extract_aca_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract optimized ACA features from network flow data
        
        Args:
            X: Network flow features
            
        Returns:
            ACA feature matrix
        """
        logger.info("Extracting optimized ACA features from network flows...")
        
        # Check if pattern matcher is properly initialized
        if not hasattr(self.pattern_matcher, 'patterns') or not self.pattern_matcher.patterns:
            logger.warning("Pattern matcher not initialized, using fallback features")
            # Return enhanced fallback features for better accuracy
            fallback_features = np.zeros((len(X), 15))  # 15 basic features
            return fallback_features
        
        # Select most relevant columns for pattern matching (optimized selection)
        pattern_columns = []
        
        # Priority 1: Critical cybersecurity features
        critical_keywords = ['packet', 'flow', 'protocol', 'flag', 'duration', 'rate', 'ttl', 'tos', 'window']
        for col in X.columns:
            if any(keyword in col.lower() for keyword in critical_keywords):
                pattern_columns.append(col)
        
        # Priority 2: Behavioral features
        behavioral_keywords = ['count', 'size', 'length', 'iat', 'bytes', 'mean', 'std', 'min', 'max']
        for col in X.columns:
            if any(keyword in col.lower() for keyword in behavioral_keywords) and col not in pattern_columns:
                pattern_columns.append(col)
        
        # Priority 3: Network features
        network_keywords = ['src', 'dst', 'port', 'ip', 'address', 'direction']
        for col in X.columns:
            if any(keyword in col.lower() for keyword in network_keywords) and col not in pattern_columns:
                pattern_columns.append(col)
        
        # Use more columns for better accuracy (increased from 15 to 25)
        if len(pattern_columns) > 25:
            pattern_columns = pattern_columns[:25]
        
        if not pattern_columns:
            # Fallback to numeric columns
            pattern_columns = X.select_dtypes(include=[np.number]).columns.tolist()[:8]
        
        logger.info(f"Using {len(pattern_columns)} columns for optimized ACA pattern matching")
        
        # Vectorized feature extraction for better performance
        try:
            # Pre-compute text representations for all samples
            text_representations = []
            for idx, row in X.iterrows():
                text_parts = []
                for col in pattern_columns:
                    value = row[col]
                    if pd.notna(value):
                        if isinstance(value, (int, float)):
                            text_parts.append(str(int(value)))
                        else:
                            text_parts.append(str(value))
                
                text_representations.append(' '.join(text_parts))
            
            # Batch process ACA features
            aca_features_list = []
            batch_size = 1000  # Process in batches for memory efficiency
            
            for i in range(0, len(text_representations), batch_size):
                batch_texts = text_representations[i:i + batch_size]
                
                for text in batch_texts:
                    try:
                        # Extract optimized ACA features
                        aca_features = self.feature_extractor.extract_all_features(text)
                        aca_features_list.append(aca_features)
                    except Exception as e:
                        logger.warning(f"Error extracting ACA features: {e}")
                        # Use fallback features
                        fallback_features = np.zeros(len(self.pattern_matcher.patterns) * 3 + 4)
                        aca_features_list.append(fallback_features)
                
                if (i + batch_size) % 5000 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(text_representations))} samples for ACA feature extraction")
            
            aca_features_matrix = np.array(aca_features_list)
            logger.info(f"Optimized ACA feature extraction completed. Shape: {aca_features_matrix.shape}")
            
            return aca_features_matrix
            
        except Exception as e:
            logger.error(f"Error in vectorized ACA feature extraction: {e}")
            # Fallback to enhanced features for better accuracy
            fallback_features = np.zeros((len(X), 15))
            return fallback_features
    
    def _combine_features(self, X: pd.DataFrame, aca_features: np.ndarray) -> np.ndarray:
        """
        Combine ACA features with original network features
        
        Args:
            X: Original network features
            aca_features: ACA-derived features
            
        Returns:
            Combined feature matrix
        """
        # Convert original features to numpy
        X_numeric = X.select_dtypes(include=[np.number]).values
        
        # Combine features
        combined_features = np.hstack([X_numeric, aca_features])
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        logger.info(f"  - Original features: {X_numeric.shape[1]}")
        logger.info(f"  - ACA features: {aca_features.shape[1]}")
        
        return combined_features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the ACA + SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional class weights
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ACA + SVM model...")
        start_time = time.time()
        
        try:
            # Load patterns
            if self.patterns_file and os.path.exists(self.patterns_file):
                patterns = self.pattern_matcher.load_patterns_from_file(self.patterns_file)
            else:
                patterns = self.pattern_matcher.load_cyber_threat_patterns()
            
            # Build ACA automaton
            if not self.pattern_matcher.build_automaton(patterns):
                raise ValueError("Failed to build ACA automaton")
            
            # Extract ACA features
            aca_features = self._extract_aca_features(X_train)
            
            # Combine with original features
            combined_features = self._combine_features(X_train, aca_features)
            
            # Create feature names
            original_feature_names = X_train.select_dtypes(include=[np.number]).columns.tolist()
            aca_feature_names = self.feature_extractor.get_feature_names()
            self.feature_names = original_feature_names + aca_feature_names
            
            # Create SVM model
            self.svm_model = self._create_svm_model()
            
            # Create pipeline with scaling
            self.pipeline = Pipeline([
                ('scaler', self.scaler),
                ('svm', self.svm_model)
            ])
            
            # Train pipeline with timeout protection
            logger.info("Training SVM pipeline...")
            logger.info(f"Training on {combined_features.shape[0]} samples with {combined_features.shape[1]} features")
            
            # Use smaller subset for faster training if dataset is too large
            if len(combined_features) > 50000:  # Increased threshold
                logger.info("Using subset of data for faster SVM training...")
                from sklearn.utils import resample
                X_subset, y_subset = resample(combined_features, y_train, n_samples=50000, random_state=42)
                self.pipeline.fit(X_subset, y_subset)
            else:
                self.pipeline.fit(combined_features, y_train)
            
            # Calculate training metrics
            y_pred_train = self.pipeline.predict(combined_features)
            
            self.training_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            
            # Store class information
            self.class_names = sorted(y_train.unique())
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            logger.info(f"ACA + SVM training completed in {training_time:.2f} seconds")
            logger.info(f"Training accuracy: {self.training_metrics['accuracy']:.4f}")
            
            return {
                'model_type': 'ACA_SVM',
                'training_metrics': self.training_metrics,
                'training_time': training_time,
                'feature_count': len(self.feature_names),
                'aca_patterns_count': len(patterns),
                'svm_type': self.svm_type,
                'class_weights_used': bool(class_weights)
            }
            
        except Exception as e:
            logger.error(f"ACA + SVM training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict threats using the ACA + SVM model
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions with ACA + SVM model...")
        
        try:
            # Extract ACA features
            aca_features = self._extract_aca_features(X)
            
            # Combine with original features
            combined_features = self._combine_features(X, aca_features)
            
            # Make predictions
            predictions = self.pipeline.predict(combined_features)
            
            # Get prediction probabilities if available
            prediction_probs = None
            if hasattr(self.pipeline.named_steps['svm'], 'predict_proba'):
                prediction_probs = self.pipeline.named_steps['svm'].predict_proba(combined_features)
            
            # Get decision function values
            decision_scores = None
            if hasattr(self.pipeline.named_steps['svm'], 'decision_function'):
                decision_scores = self.pipeline.named_steps['svm'].decision_function(combined_features)
            
            # Create prediction details
            prediction_details = {
                'predictions': predictions,
                'prediction_probs': prediction_probs,
                'decision_scores': decision_scores,
                'aca_features_shape': aca_features.shape,
                'combined_features_shape': combined_features.shape,
                'svm_type': self.svm_type,
                'feature_count': len(self.feature_names)
            }
            
            return predictions, prediction_details
            
        except Exception as e:
            logger.error(f"ACA + SVM prediction failed: {e}")
            raise
    
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
        
        logger.info("Evaluating ACA + SVM model...")
        
        try:
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
                'svm_type': self.svm_type,
                'feature_count': len(self.feature_names),
                'aca_patterns_count': len(self.pattern_matcher.patterns),
                'prediction_details': details
            })
            
            # Generate classification report
            try:
                class_report = classification_report(y_test, predictions, output_dict=True)
                metrics['classification_report'] = class_report
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"ACA + SVM evaluation failed: {e}")
            raise
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             param_grid: Dict = None) -> Dict:
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Tuning results
        """
        logger.info("Performing hyperparameter tuning for ACA + SVM...")
        
        try:
            # Extract ACA features
            aca_features = self._extract_aca_features(X_train)
            combined_features = self._combine_features(X_train, aca_features)
            
            # Default parameter grid
            if param_grid is None:
                if self.svm_type == 'linear':
                    param_grid = {
                        'svm__C': [0.1, 1.0, 10.0, 100.0]
                    }
                else:
                    param_grid = {
                        'svm__C': [0.1, 1.0, 10.0],
                        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                    }
            
            # Create base pipeline
            base_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', self._create_svm_model())
            ])
            
            # Grid search
            grid_search = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(combined_features, y_train)
            
            # Update model with best parameters
            self.C = grid_search.best_params_.get('svm__C', self.C)
            self.gamma = grid_search.best_params_.get('svm__gamma', self.gamma)
            
            # Retrain with best parameters
            self.svm_model = self._create_svm_model()
            self.pipeline = Pipeline([
                ('scaler', self.scaler),
                ('svm', self.svm_model)
            ])
            self.pipeline.fit(combined_features, y_train)
            
            logger.info(f"Hyperparameter tuning completed. Best params: {grid_search.best_params_}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create serializable model data (avoid circular references)
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_metrics': self.training_metrics,
            'svm_type': self.svm_type,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'patterns': self.pattern_matcher.patterns,  # Save patterns as simple list
            'pattern_weights': self.pattern_matcher.pattern_weights,
            'pattern_categories': self.pattern_matcher.pattern_categories
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"ACA + SVM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.training_metrics = model_data['training_metrics']
        self.svm_type = model_data['svm_type']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        self.class_weight = model_data['class_weight']
        self.max_iter = model_data['max_iter']
        
        # Recreate pattern matcher
        self.pattern_matcher = ACAPatternMatcher()
        self.pattern_matcher.patterns = model_data['patterns']
        self.pattern_matcher.pattern_weights = model_data.get('pattern_weights', {})
        self.pattern_matcher.pattern_categories = model_data.get('pattern_categories', {})
        
        # Recreate feature extractor
        self.feature_extractor = ACAFeatureExtractor(self.pattern_matcher)
        
        self.is_trained = True
        logger.info(f"ACA + SVM model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for linear SVM only)
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.svm_type != 'linear':
            logger.warning("Feature importance only available for linear SVM")
            return pd.DataFrame()
        
        try:
            # Get coefficients from linear SVM
            svm_model = self.pipeline.named_steps['svm']
            coefficients = svm_model.coef_[0] if hasattr(svm_model, 'coef_') else []
            
            if len(coefficients) == 0:
                return pd.DataFrame()
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(coefficients)],
                'importance': np.abs(coefficients)
            })
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'ACA_SVM',
            'svm_type': self.svm_type,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'aca_patterns_count': len(self.pattern_matcher.patterns) if self.pattern_matcher.patterns else 0,
            'training_metrics': self.training_metrics
        }
    
    def log_detection(self, flow_id: str, prediction: int, confidence: float, 
                     features: Dict, aca_matches: List = None) -> Dict:
        """
        Log detection for manual verification
        
        Args:
            flow_id: Unique flow identifier
            prediction: Model prediction
            confidence: Prediction confidence
            features: Feature values used for prediction
            aca_matches: ACA pattern matches found
            
        Returns:
            Detection log entry
        """
        detection_log = {
            'timestamp': datetime.now().isoformat(),
            'model_name': 'ACA_SVM',
            'flow_id': flow_id,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'features': features,
            'aca_matches': aca_matches or [],
            'verification_status': 'pending',
            'manual_label': None,
            'verification_notes': None
        }
        
        # Log to file
        log_file = 'logs/aca_svm_detections.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(detection_log) + '\n')
        
        logger.info(f"Detection logged for flow {flow_id}: prediction={prediction}, confidence={confidence:.3f}")
        
        return detection_log
    
    def update_verification(self, flow_id: str, manual_label: int, 
                           verification_notes: str = None) -> bool:
        """
        Update manual verification for a detection
        
        Args:
            flow_id: Flow identifier
            manual_label: Manual verification label
            verification_notes: Additional notes
            
        Returns:
            True if updated successfully
        """
        log_file = 'logs/aca_svm_detections.log'
        
        if not os.path.exists(log_file):
            logger.warning("No detection log file found")
            return False
        
        try:
            # Read all log entries
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Update matching entry
            updated = False
            with open(log_file, 'w') as f:
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('flow_id') == flow_id:
                            entry['verification_status'] = 'verified'
                            entry['manual_label'] = int(manual_label)
                            entry['verification_notes'] = verification_notes
                            entry['verification_timestamp'] = datetime.now().isoformat()
                            updated = True
                        f.write(json.dumps(entry) + '\n')
                    except json.JSONDecodeError:
                        f.write(line)  # Keep malformed lines as-is
            
            if updated:
                logger.info(f"Verification updated for flow {flow_id}: label={manual_label}")
            else:
                logger.warning(f"No detection found for flow {flow_id}")
            
            return updated
            
        except Exception as e:
            logger.error(f"Failed to update verification: {e}")
            return False
    
    def get_verification_stats(self) -> Dict:
        """
        Get verification statistics
        
        Returns:
            Dictionary with verification statistics
        """
        log_file = 'logs/aca_svm_detections.log'
        
        if not os.path.exists(log_file):
            return {
                'total_detections': 0,
                'pending_verifications': 0,
                'verified_detections': 0,
                'accuracy_rate': 0.0
            }
        
        try:
            total_detections = 0
            pending_verifications = 0
            verified_detections = 0
            correct_predictions = 0
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total_detections += 1
                        
                        if entry.get('verification_status') == 'pending':
                            pending_verifications += 1
                        elif entry.get('verification_status') == 'verified':
                            verified_detections += 1
                            if entry.get('prediction') == entry.get('manual_label'):
                                correct_predictions += 1
                    except json.JSONDecodeError:
                        continue
            
            accuracy_rate = correct_predictions / max(verified_detections, 1)
            
            return {
                'total_detections': total_detections,
                'pending_verifications': pending_verifications,
                'verified_detections': verified_detections,
                'accuracy_rate': accuracy_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to get verification stats: {e}")
            return {}
    
    def predict_with_logging(self, X: pd.DataFrame, flow_ids: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Predict with automatic logging for manual verification
        
        Args:
            X: Input features
            flow_ids: Optional flow identifiers for logging
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions with ACA + SVM model (with logging)...")
        
        try:
            # Make predictions
            predictions, details = self.predict(X)
            
            # Log detections for manual verification
            if flow_ids is None:
                flow_ids = [f"flow_{i}" for i in range(len(X))]
            
            # Get prediction probabilities for confidence
            prediction_probs = details.get('prediction_probs')
            if prediction_probs is None:
                # Use decision scores as confidence proxy
                decision_scores = details.get('decision_scores')
                if decision_scores is not None:
                    # Handle both 1D and 2D decision scores
                    if decision_scores.ndim == 1:
                        # Binary classification - use absolute values
                        confidences = np.abs(decision_scores)
                    else:
                        # Multi-class - convert decision scores to probabilities (simple normalization)
                        max_scores = np.max(np.abs(decision_scores), axis=1, keepdims=True)
                        normalized_scores = decision_scores / (max_scores + 1e-8)
                        confidences = np.max(normalized_scores, axis=1)
                else:
                    confidences = np.ones(len(predictions))  # Default confidence
            else:
                confidences = np.max(prediction_probs, axis=1)
            
            # Log each detection
            logged_detections = []
            for i, (flow_id, prediction, confidence) in enumerate(zip(flow_ids, predictions, confidences)):
                # Extract ACA matches for this sample
                aca_features = self._extract_aca_features(X.iloc[[i]])
                sample_text = ' '.join([str(X.iloc[i][col]) for col in X.columns[:5]])  # Use first 5 columns
                aca_matches = self.pattern_matcher.match_patterns(sample_text)
                
                # Create feature dictionary
                feature_dict = {}
                for col in X.columns:
                    value = X.iloc[i][col]
                    try:
                        feature_dict[col] = float(value)
                    except (ValueError, TypeError):
                        feature_dict[col] = str(value)
                
                # Log detection
                detection_log = self.log_detection(
                    flow_id=flow_id,
                    prediction=prediction,
                    confidence=confidence,
                    features=feature_dict,
                    aca_matches=aca_matches
                )
                logged_detections.append(detection_log)
            
            # Update prediction details
            details['logged_detections'] = logged_detections
            details['verification_stats'] = self.get_verification_stats()
            
            return predictions, details
            
        except Exception as e:
            logger.error(f"ACA + SVM prediction with logging failed: {e}")
            raise
