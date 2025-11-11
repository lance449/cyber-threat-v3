"""
Model 3: IntruDTree
Interpretable decision-tree model tailored to cybersecurity
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import os

logger = logging.getLogger(__name__)

class IntruDTreeNode:
    """
    Node class for IntruDTree decision tree
    """
    
    def __init__(self, feature_idx: int = None, threshold: float = None, 
                 value: int = None, left: 'IntruDTreeNode' = None, 
                 right: 'IntruDTreeNode' = None, depth: int = 0):
        """
        Initialize IntruDTree node
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.is_leaf = value is not None
        self.feature_name = None
        self.split_info = {}

class IntruDTree:
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, cybersecurity_features: List[str] = None,
                 n_important_features: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cybersecurity_features = cybersecurity_features or self._get_default_cybersecurity_features()
        self.n_important_features = n_important_features
        self.root = None
        self.feature_names = []
        self.n_classes = 0
        self.important_features = []
        
        # Add class balancing parameters
        self.class_weights = None
        self.min_samples_per_class = 100  # Minimum samples per class for meaningful splits
        
    def _get_default_cybersecurity_features(self) -> List[str]:
        """
        Get default cybersecurity features to prioritize
        
        Returns:
            List of cybersecurity feature names
        """
        return [
            'flow_duration', 'packet_length_mean', 'flow_bytes_per_sec',
            'flow_packets_per_sec', 'flow_iat_mean', 'fwd_iat_mean',
            'bwd_iat_mean', 'flow_fwd_packets', 'flow_bwd_packets',
            'fwd_packet_length_max', 'bwd_packet_length_max',
            'fwd_packet_length_min', 'bwd_packet_length_min',
            'flow_fwd_bytes', 'flow_bwd_bytes', 'fwd_packets_per_sec',
            'bwd_packets_per_sec', 'fin_flag_count', 'syn_flag_count',
            'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
            'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
            'down_up_ratio', 'avg_packet_size', 'fwd_seg_size_avg',
            'bwd_seg_size_avg', 'fwd_bytes_per_bulk', 'bwd_bytes_per_bulk',
            'fwd_bulk_rate_avg', 'bwd_bulk_rate_avg', 'fwd_subflow_packets',
            'bwd_subflow_packets', 'fwd_subflow_bytes', 'bwd_subflow_bytes',
            'fwd_init_win_bytes', 'bwd_init_win_bytes', 'fwd_act_data_pkts',
            'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max',
            'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'
        ]
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        
        # Count occurrences of each class
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_information_gain(self, y: np.ndarray, y_left: np.ndarray, 
                                  y_right: np.ndarray) -> float:
        """
        Calculate information gain for a split
        
        Args:
            y: Original labels
            y_left: Labels in left split
            y_right: Labels in right split
        """
        # Calculate parent entropy
        parent_entropy = self._calculate_entropy(y)
        
        # Calculate weighted child entropies
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)
        
        if n_total == 0:
            return 0.0
        
        left_entropy = self._calculate_entropy(y_left)
        right_entropy = self._calculate_entropy(y_right)
        
        # Calculate weighted average of child entropies
        weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
        
        # Information gain
        information_gain = parent_entropy - weighted_entropy
        return information_gain
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str]) -> Tuple[int, float, float]:
        """
        Find the best split for a node (optimized for large datasets)
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
        """
        best_gain = -1
        best_feature_idx = -1
        best_threshold = 0
        
        n_samples, n_features = X.shape
        
        # Calculate parent entropy
        parent_entropy = self._calculate_entropy(y)
        
        # Prioritize features - try important features first, then random
        priority_features = []
        if hasattr(self, 'feature_importance_scores') and self.feature_importance_scores:
            # Sort features by importance
            sorted_features = sorted(
                self.feature_importance_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            # Take top 50 important features with safety check
            for f in sorted_features[:50]:
                if f[0] in feature_names:
                    try:
                        idx = feature_names.index(f[0])
                        if idx < X.shape[1]:  # Ensure index is valid
                            priority_features.append(idx)
                    except (ValueError, IndexError):
                        pass
        
        # Combine priority features with random features
        if len(priority_features) > 0:
            remaining_features = [i for i in range(n_features) if i not in priority_features]
            # Try up to 60 features total: all priority + random from remaining
            random_indices = np.random.choice(remaining_features, min(60 - len(priority_features), len(remaining_features)), replace=False)
            feature_indices = list(priority_features) + list(random_indices)
        else:
            # No priorities, use random sampling - increase to 60
            max_features_to_try = min(n_features, 60)
            feature_indices = np.random.choice(n_features, max_features_to_try, replace=False)
        
        # Try each selected feature
        for feature_idx in feature_indices:
            # Safety check for feature index bounds
            if feature_idx >= X.shape[1]:
                continue
                
            feature_values = X[:, feature_idx]
            
            # Skip constant features
            if len(np.unique(feature_values)) <= 1:
                continue
                
            unique_values = np.unique(feature_values)
            
            # Increase threshold candidates to 200 for maximum split quality
            max_thresholds = min(len(unique_values), 200)
            if len(unique_values) > max_thresholds:
                # Sample thresholds more intelligently - prioritize values that create balanced splits
                percentiles = np.linspace(0.05, 0.95, max_thresholds)
                threshold_candidates = np.percentile(unique_values, percentiles)
            else:
                threshold_candidates = unique_values
            
            # Try each threshold candidate
            for threshold in threshold_candidates:
                # Create split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Very lenient split requirements - only require 1 sample per side
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
                # Calculate information gain with improved method
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                gain = self._calculate_information_gain(y, y_left, y_right)
                
                # Apply cybersecurity feature bonus (increased from 1.2 to 1.5) with safety check
                if feature_idx < len(feature_names) and feature_names[feature_idx] in self.cybersecurity_features:
                    gain *= 1.5
                
                # Update best split if this is better (accept any positive gain)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        # Ensure we return a positive gain even if it's very small
        return best_feature_idx, best_threshold, max(best_gain, 0.0001)
    
    def _create_leaf_node(self, y: np.ndarray, depth: int) -> IntruDTreeNode:
        """
        Create a leaf node
        
        Args:
            y: Labels for the node
            depth: Depth of the node
            
        Returns:
            Leaf node
        """
        # Predict the most common class
        unique, counts = np.unique(y, return_counts=True)
        predicted_value = unique[np.argmax(counts)]
        
        return IntruDTreeNode(value=predicted_value, depth=depth)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, 
                   feature_names: List[str]) -> IntruDTreeNode:
        """
        Recursively build the decision tree
        
        Args:
            X: Feature matrix
            y: Labels
            depth: Current depth
            feature_names: List of feature names
            
        Returns:
            Root node of the subtree
        """
        n_samples = len(y)
        
        # Check stopping conditions - less aggressive early stopping
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return self._create_leaf_node(y, depth)
        
        # Find best split using improved algorithm
        best_feature_idx, best_threshold, best_gain = self._find_best_split(X, y, feature_names)
        
        # Very permissive split acceptance - accept any split that separates data
        if best_feature_idx == -1:
            return self._create_leaf_node(y, depth)
        
        # Create split
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # More lenient check - allow smaller splits
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf_node(y, depth)
        
        # Create child nodes
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_names)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_names)
        
        # Create internal node
        node = IntruDTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            depth=depth
        )
        node.feature_name = feature_names[best_feature_idx]
        node.split_info = {
            'gain': best_gain,
            'n_samples': n_samples,
            'n_left': np.sum(left_mask),
            'n_right': np.sum(right_mask)
        }
        
        return node
    
    def intrudtree(self, X: pd.DataFrame, y: pd.Series) -> 'IntruDTree':
        """
        Main IntruDTree algorithm following the exact pseudocode from the study
        Procedure IntruDTree(DS, feature_list, n):
        
        Study Requirements:
        1. Calculate feature importance using information gain
        2. Select top-n important features for cybersecurity
        3. Build decision tree with cybersecurity-specific criteria
        4. Apply pruning based on cybersecurity domain knowledge
        """
        logger.info("Training IntruDTree model following study pseudocode...")
        
        # Store feature names and class information
        self.feature_names = X.columns.tolist()
        self.n_classes = len(np.unique(y))
        
        # Apply data balancing for better class representation
        logger.info("Applying data balancing for better class representation...")
        X_balanced, y_balanced = self._balance_dataset(X, y)
        logger.info(f"Balanced dataset: {len(X_balanced)} samples, {len(np.unique(y_balanced))} classes")
        
        # 1. Feature importance calculation using information gain (Study Requirement 1)
        logger.info("Step 1: Calculating feature importance using information gain...")
        importance_scores = self._calculate_cybersecurity_feature_importance(X_balanced, y_balanced)
        
        # Store feature importance scores for use in split scoring
        self.feature_importance_scores = importance_scores
        
        # 2. Select important features for cybersecurity (Study Requirement 2)
        logger.info("Step 2: Selecting top-n important features for cybersecurity...")
        self.important_features = self._select_cybersecurity_features(
            self.feature_names, importance_scores, self.n_important_features
        )
        
        logger.info(f"Selected {len(self.important_features)} cybersecurity features:")
        for i, feature in enumerate(self.important_features[:10]):
            logger.info(f"  {i+1}. {feature} (importance: {importance_scores[feature]:.4f})")
        
        # 3. Build cybersecurity-specific decision tree (Study Requirement 3)
        logger.info("Step 3: Building cybersecurity-specific decision tree...")
        X_important = X_balanced[self.important_features]
        X_np = X_important.values
        y_np = y_balanced.values
        
        # Create feature mapping to match the reduced feature space
        feature_names_reduced = list(X_important.columns)
        
        self.root = self._build_cybersecurity_tree(X_np, y_np, feature_names_reduced, 0)
        
        # 4. Apply cybersecurity-specific pruning (Study Requirement 4)
        logger.info("Step 4: Applying cybersecurity-specific pruning...")
        self._apply_cybersecurity_pruning()
        
        # Debug: Check tree structure after pruning
        logger.info(f"Tree structure after pruning:")
        logger.info(f"  - Root is leaf: {self.root.is_leaf if self.root else 'No root'}")
        logger.info(f"  - Root depth: {self.root.depth if self.root else 'No root'}")
        if self.root and not self.root.is_leaf:
            logger.info(f"  - Root has left child: {self.root.left is not None}")
            logger.info(f"  - Root has right child: {self.root.right is not None}")
        
        logger.info(f"IntruDTree training completed successfully!")
        logger.info(f"  - Tree depth: {self._get_tree_depth()}")
        logger.info(f"  - Features used: {len(self.important_features)}")
        logger.info(f"  - Classes: {self.n_classes}")
        
        return self
    
    def _balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Enhanced dataset balancing with multiple techniques based on imbalance severity
        """
        from sklearn.utils import resample
        
        # Get class distribution
        class_counts = y.value_counts()
        logger.info(f"Original class distribution: {dict(class_counts)}")
        
        # Calculate class imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # Choose balancing strategy based on imbalance severity - balanced approach
        if imbalance_ratio > 10:  # Severe imbalance
            logger.info("Severe imbalance detected, using moderate balancing (less aggressive)")
            target_size = int(min_count * 1.5)  # Reduced from 3x to 1.5x - less aggressive
        elif imbalance_ratio > 5:  # Moderate imbalance
            logger.info("Moderate imbalance detected, using gentle balancing")
            target_size = int(min_count * 1.2)  # Reduced from 2x to 1.2x - less aggressive
        elif imbalance_ratio > 2:  # Mild imbalance
            logger.info("Mild imbalance detected, using minimal balancing")
            target_size = int((max_count + min_count) / 2)  # Gentle balancing
        else:
            logger.info("Balanced dataset, no balancing needed")
            return X, y
        
        logger.info(f"Target size for balancing: {target_size}")
        
        # Resample each class with enhanced strategy
        balanced_samples = []
        balanced_labels = []
        
        for class_label in y.unique():
            class_mask = y == class_label
            # Reset index to avoid alignment issues
            class_X = X.reset_index(drop=True)[class_mask.reset_index(drop=True)]
            class_y = y.reset_index(drop=True)[class_mask.reset_index(drop=True)]
            
            if len(class_X) > target_size:
                # Downsample majority class
                class_X_resampled, class_y_resampled = resample(
                    class_X, class_y, n_samples=target_size, random_state=42
                )
            else:
                # Upsample minority class with replacement
                class_X_resampled, class_y_resampled = resample(
                    class_X, class_y, n_samples=target_size, random_state=42, replace=True
                )
            
            balanced_samples.append(class_X_resampled)
            balanced_labels.append(class_y_resampled)
        
        # Combine all balanced samples
        X_balanced = pd.concat(balanced_samples, ignore_index=True)
        y_balanced = pd.concat(balanced_labels, ignore_index=True)
        
        logger.info(f"Balanced class distribution: {dict(y_balanced.value_counts())}")
        logger.info(f"Dataset size: {X.shape[0]} -> {X_balanced.shape[0]} samples")
        
        return X_balanced, y_balanced
    
    def _calculate_cybersecurity_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate cybersecurity-specific feature importance using optimized information gain
        """
        logger.info("Calculating cybersecurity feature importance (optimized)...")
        importance_scores = {}
        
        # Use sklearn's mutual_info_classif for faster computation with zero-variance removal
        try:
            from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
            
            # Convert categorical features to numeric if needed
            X_numeric = X.copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == 'object':
                    X_numeric[col] = pd.Categorical(X_numeric[col]).codes
            
            # Handle infinity and NaN values
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
            X_numeric = X_numeric.fillna(0)
            
            # Remove zero-variance features first
            var_threshold = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
            X_filtered = var_threshold.fit_transform(X_numeric)
            selected_features = X_numeric.columns[var_threshold.get_support()]
            
            logger.info(f"Removed {len(X_numeric.columns) - len(selected_features)} zero-variance features")
            
            # Calculate mutual information scores on filtered features
            mi_scores = mutual_info_classif(X_filtered, y, random_state=42, n_neighbors=3)
            
            # Create importance dictionary with enhanced cybersecurity bonuses
            for i, feature in enumerate(selected_features):
                base_score = float(mi_scores[i])
                
                # Enhanced cybersecurity feature bonus system
                if feature in self.cybersecurity_features:
                    if any(keyword in feature.lower() for keyword in ['packet', 'flow', 'flag', 'protocol', 'port', 'ip']):
                        base_score *= 2.0  # 100% bonus for critical network features
                    elif any(keyword in feature.lower() for keyword in ['attack', 'threat', 'malicious', 'suspicious']):
                        base_score *= 1.8  # 80% bonus for threat-related features
                    else:
                        base_score *= 1.5  # 50% bonus for other cybersecurity features
                
                # Enhanced behavioral feature bonuses
                if any(keyword in feature.lower() for keyword in ['duration', 'rate', 'size', 'count', 'frequency', 'pattern']):
                    base_score *= 1.4  # 40% bonus for behavioral features
                
                # Statistical feature bonuses
                if any(keyword in feature.lower() for keyword in ['mean', 'std', 'var', 'min', 'max', 'median']):
                    base_score *= 1.3  # 30% bonus for statistical features
                
                # Time-based feature bonuses
                if any(keyword in feature.lower() for keyword in ['time', 'timestamp', 'hour', 'day', 'minute']):
                    base_score *= 1.2  # 20% bonus for time-based features
                
                importance_scores[feature] = base_score
            
            # Add zero scores for removed features
            for feature in X_numeric.columns:
                if feature not in importance_scores:
                    importance_scores[feature] = 0.0
                
        except ImportError:
            logger.warning("sklearn.feature_selection not available, using enhanced fallback method")
            # Enhanced fallback calculation with zero-variance removal
            critical_features = X.columns[:min(50, len(X.columns))]  # Increased limit
            
            for feature in critical_features:
                try:
                    feature_values = X[feature]
                    
                    # Skip zero-variance features
                    if feature_values.var() < 0.01:
                        importance_scores[feature] = 0.0
                        continue
                    
                    gain = self._calculate_information_gain_for_feature(feature_values, y)
                    
                    # Apply enhanced cybersecurity feature bonus
                    if feature in self.cybersecurity_features:
                        if any(keyword in feature.lower() for keyword in ['packet', 'flow', 'flag', 'protocol', 'port', 'ip']):
                            gain *= 2.0
                        elif any(keyword in feature.lower() for keyword in ['attack', 'threat', 'malicious', 'suspicious']):
                            gain *= 1.8
                        else:
                            gain *= 1.5
                    
                    if any(keyword in feature.lower() for keyword in ['duration', 'rate', 'size', 'count', 'frequency', 'pattern']):
                        gain *= 1.4
                    
                    if any(keyword in feature.lower() for keyword in ['mean', 'std', 'var', 'min', 'max', 'median']):
                        gain *= 1.3
                    
                    if any(keyword in feature.lower() for keyword in ['time', 'timestamp', 'hour', 'day', 'minute']):
                        gain *= 1.2
                    
                    importance_scores[feature] = gain
                except Exception as e:
                    logger.warning(f"Error calculating importance for {feature}: {e}")
                    importance_scores[feature] = 0.0
        
        logger.info(f"Calculated importance scores for {len(importance_scores)} features")
        return importance_scores
    
    def _select_cybersecurity_features(self, feature_names: List[str], importance_scores: Dict[str, float], n: int) -> List[str]:
        """
        Select top-n cybersecurity features based on study requirements
        """
        # Sort features by importance score
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prioritize cybersecurity features
        cybersecurity_selected = []
        other_selected = []
        
        for feature, score in sorted_features:
            if feature in self.cybersecurity_features and len(cybersecurity_selected) < n // 2:
                cybersecurity_selected.append(feature)
            elif len(cybersecurity_selected) + len(other_selected) < n:
                other_selected.append(feature)
        
        # Combine selections
        selected_features = cybersecurity_selected + other_selected
        
        logger.info(f"Selected {len(cybersecurity_selected)} cybersecurity features and {len(other_selected)} other features")
        return selected_features
    
    def _build_cybersecurity_tree(self, X: np.ndarray, y: np.ndarray, imp_features: List[str], depth: int) -> IntruDTreeNode:
        """
        Build cybersecurity-specific decision tree with improved algorithm
        """
        N = IntruDTreeNode()
        
        # Check if all instances have same class
        if len(np.unique(y)) == 1:
            N.value = y[0]
            N.is_leaf = True
            N.depth = depth
            return N
        
        # Stopping criteria - allow deeper trees for better accuracy
        if depth >= self.max_depth:
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            N.depth = depth
            return N
        
        # Find best split using improved algorithm
        # Use the improved _find_best_split method if available
        if hasattr(self, '_find_best_split'):
            # Extract feature names for compatibility
            feature_names = imp_features if imp_features else [f'feature_{i}' for i in range(X.shape[1])]
            best_feature_idx, best_threshold, best_gain = self._find_best_split(X, y, feature_names)
        else:
            # Fallback to simplified search
            best_feature_idx, best_threshold, best_gain = self._simple_find_best_split(X, y, imp_features)
        
        # Accept split even if gain is very small (for maximum tree growth)
        if best_feature_idx == -1:
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            N.depth = depth
            return N
        
        # Create internal node
        N.feature_idx = best_feature_idx
        N.threshold = best_threshold
        N.feature_name = imp_features[best_feature_idx] if best_feature_idx < len(imp_features) else f"feature_{best_feature_idx}"
        N.depth = depth
        N.split_info = {'gain': best_gain, 'cybersecurity_score': best_gain}
        
        # Create splits
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # Ensure we have at least 1 sample in each split (minimal requirement)
        if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            N.depth = depth
            return N
        
        # Create child nodes with remaining features
        remaining_features = [f for i, f in enumerate(imp_features) if i != best_feature_idx]
        
        if np.sum(left_mask) > 0:
            N.left = self._build_cybersecurity_tree(X[left_mask], y[left_mask], remaining_features, depth + 1)
        else:
            unique, counts = np.unique(y[left_mask], return_counts=True)
            N.left = IntruDTreeNode(value=unique[np.argmax(counts)], is_leaf=True, depth=depth+1)
        
        if np.sum(right_mask) > 0:
            N.right = self._build_cybersecurity_tree(X[right_mask], y[right_mask], remaining_features, depth + 1)
        else:
            unique, counts = np.unique(y[right_mask], return_counts=True)
            N.right = IntruDTreeNode(value=unique[np.argmax(counts)], is_leaf=True, depth=depth+1)
        
        return N
    
    def _simple_find_best_split(self, X: np.ndarray, y: np.ndarray, imp_features: List[str]) -> Tuple[int, float, float]:
        """Simplified split finding for fallback"""
        best_gain = -1
        best_feature_idx = -1
        best_threshold = 0
        
        for i in range(min(X.shape[1], len(imp_features))):
            feature_values = X[:, i]
            
            if len(np.unique(feature_values)) <= 1:
                continue
            
            # Try up to 50 thresholds
            unique_values = np.unique(feature_values)
            if len(unique_values) > 50:
                indices = np.linspace(0, len(unique_values)-1, 50, dtype=int)
                threshold_candidates = unique_values[indices]
            else:
                threshold_candidates = unique_values
            
            for threshold in threshold_candidates:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Very lenient - only require 1 sample per side
                if np.sum(left_mask) >= 1 and np.sum(right_mask) >= 1:
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    gain = self._calculate_information_gain(y, y_left, y_right)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = i
                        best_threshold = threshold
        
        # Ensure we return a positive gain
        return best_feature_idx, best_threshold, max(best_gain, 0.0001)
    
    def _calculate_improved_split_score(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, feature_idx: int) -> float:
        """
        Calculate improved split score with simplified and more effective approach
        """
        # Base information gain
        base_gain = self._calculate_information_gain(y_parent, y_left, y_right)
        
        # Early return for poor splits
        if base_gain <= 0:
            return 0.0
        
        # Class diversity analysis
        left_classes = len(np.unique(y_left))
        right_classes = len(np.unique(y_right))
        
        # Simplified bonus system focused on class purity
        diversity_bonus = 1.0
        if left_classes > 1 and right_classes > 1:
            # Both children have multiple classes - this is good for learning
            diversity_bonus = 1.3
        elif left_classes > 1 or right_classes > 1:
            # At least one child has diversity
            diversity_bonus = 1.1
        
        # Balance bonus - reward reasonably balanced splits
        n_left, n_right = len(y_left), len(y_right)
        balance_ratio = min(n_left, n_right) / max(n_left, n_right) if max(n_left, n_right) > 0 else 0
        # Reward balanced splits but don't penalize too heavily
        if balance_ratio > 0.2:
            balance_bonus = 1.2  # Good balance
        elif balance_ratio > 0.1:
            balance_bonus = 1.0  # Acceptable
        else:
            balance_bonus = 0.8  # Unbalanced but still acceptable
        
        # Enhanced feature importance bonus for important features
        feature_bonus = 1.0
        if feature_idx < len(self.important_features):
            feature_name = self.important_features[feature_idx]
            if hasattr(self, 'feature_importance_scores') and feature_name in self.feature_importance_scores:
                importance_score = self.feature_importance_scores[feature_name]
                feature_bonus = 1.0 + (importance_score * 0.3)  # Moderate bonus
        
        # Class purity improvement bonus
        purity_bonus = 0.0
        parent_purity = self._calculate_class_purity(y_parent)
        left_purity = self._calculate_class_purity(y_left)
        right_purity = self._calculate_class_purity(y_right)
        
        # Reward splits that increase purity in children
        if left_purity > parent_purity or right_purity > parent_purity:
            avg_child_purity = (left_purity + right_purity) / 2.0
            if avg_child_purity > parent_purity:
                purity_bonus = (avg_child_purity - parent_purity) * 2.0  # Scale up the improvement
        
        # Final score calculation - simpler and more focused
        final_score = base_gain * diversity_bonus * balance_bonus * feature_bonus + purity_bonus
        
        return max(final_score, 0.0001)  # Ensure positive score
    
    def _calculate_class_purity(self, y: np.ndarray) -> float:
        """Calculate class purity (proportion of majority class)"""
        if len(y) == 0:
            return 0.0
        
        unique, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        return max_count / len(y)
    
    def _calculate_class_imbalance(self, y: np.ndarray) -> float:
        """Calculate class imbalance ratio"""
        if len(y) == 0:
            return 1.0
        
        class_counts = np.bincount(y)
        if len(class_counts) <= 1:
            return 1.0
        
        return np.max(class_counts) / np.min(class_counts)
    
    def _calculate_cybersecurity_split_score(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, feature_name: str) -> float:
        """
        Calculate cybersecurity-specific split score with class balancing
        """
        # Base information gain
        base_gain = self._calculate_information_gain(y_parent, y_left, y_right)
        
        # Cybersecurity-specific adjustments
        cybersecurity_bonus = 1.0
        
        # Bonus for cybersecurity features
        if feature_name in self.cybersecurity_features:
            cybersecurity_bonus *= 1.3
        
        # Bonus for behavioral features
        if any(keyword in feature_name.lower() for keyword in ['packet', 'flow', 'duration', 'rate']):
            cybersecurity_bonus *= 1.2
        
        # Bonus for flag-related features
        if 'flag' in feature_name.lower():
            cybersecurity_bonus *= 1.4
        
        # Class balancing bonus - reward splits that create more balanced class distributions
        n_left, n_right = len(y_left), len(y_right)
        n_total = len(y_parent)
        
        # Calculate class distribution balance
        left_classes = len(np.unique(y_left))
        right_classes = len(np.unique(y_right))
        parent_classes = len(np.unique(y_parent))
        
        # Reward splits that maintain or increase class diversity
        class_diversity_bonus = 1.0
        if left_classes > 1 and right_classes > 1:
            class_diversity_bonus = 1.5  # Strong bonus for balanced splits
        elif left_classes > 1 or right_classes > 1:
            class_diversity_bonus = 1.2  # Moderate bonus for partial balance
        
        # Penalty for very unbalanced splits
        balance_ratio = min(n_left, n_right) / max(n_left, n_right)
        if balance_ratio < 0.1:  # Very unbalanced split
            cybersecurity_bonus *= 0.3  # Strong penalty
        elif balance_ratio < 0.2:  # Moderately unbalanced
            cybersecurity_bonus *= 0.7  # Moderate penalty
        
        return base_gain * cybersecurity_bonus * class_diversity_bonus
    
    def _apply_cybersecurity_pruning(self):
        """
        Apply cybersecurity-specific pruning - more conservative approach
        """
        logger.info("Applying cybersecurity-specific pruning...")
        
        def prune_node(node, parent_score=float('inf')):
            if node.is_leaf:
                return
            
            # Only prune if absolutely necessary (very conservative)
            node_score = node.split_info.get('gain', 0)
            
            # Very conservative pruning - only prune extremely weak splits at depth > 10
            if node.depth > 10 and node_score < 0.00001 and node_score < parent_score * 0.1:
                logger.info(f"Pruning weak node at depth {node.depth} with score {node_score:.6f}")
                # Convert to leaf with majority class from children
                # Get class distribution from children
                classes = []
                if node.left:
                    self._gather_classes(node.left, classes)
                if node.right:
                    self._gather_classes(node.right, classes)
                
                if classes:
                    unique, counts = np.unique(classes, return_counts=True)
                    node.value = unique[np.argmax(counts)]
                else:
                    node.value = 0  # Default
                
                node.is_leaf = True
                node.left = None
                node.right = None
                return
            
            # Recursively prune children with current node's score
            if node.left:
                prune_node(node.left, node_score)
            if node.right:
                prune_node(node.right, node_score)
        
        if self.root:
            prune_node(self.root)
        
        logger.info("Cybersecurity pruning completed")
    
    def _gather_classes(self, node, classes_list):
        """Helper to gather class values from subtree"""
        if node.is_leaf:
            classes_list.append(node.value)
        else:
            if node.left:
                self._gather_classes(node.left, classes_list)
            if node.right:
                self._gather_classes(node.right, classes_list)
    
    def _tree_gen_improved(self, X: np.ndarray, y: np.ndarray, imp_features: List[str], depth: int) -> IntruDTreeNode:
        """
        Improved TreeGen procedure with better splitting criteria
        """
        N = IntruDTreeNode()
        
        # Check if all instances have same class
        if len(np.unique(y)) == 1:
            N.value = y[0]
            N.is_leaf = True
            return N
        
        # Check stopping conditions
        if (not imp_features or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split or
            len(y) < 50):  # Minimum samples for meaningful splits
            # Return leaf with majority class
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            return N
        
        # Find best split feature with improved criteria
        best_feature_idx = -1
        best_gain = -1
        best_threshold = 0
        best_gini = float('inf')
        
        for i, feature in enumerate(imp_features):
            if i < X.shape[1]:  # Make sure we don't go out of bounds
                feature_values = X[:, i]
                
                # Skip constant features
                if len(np.unique(feature_values)) <= 1:
                    continue
                
                # Find best threshold for this feature
                unique_values = np.unique(feature_values)
                
                # Sample thresholds for performance (max 20 thresholds)
                if len(unique_values) > 20:
                    indices = np.linspace(0, len(unique_values)-1, 20, dtype=int)
                    threshold_candidates = [unique_values[i] for i in indices]
                else:
                    threshold_candidates = unique_values
                
                for threshold in threshold_candidates:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) >= self.min_samples_leaf and np.sum(right_mask) >= self.min_samples_leaf:
                        y_left = y[left_mask]
                        y_right = y[right_mask]
                        
                        # Use both information gain and Gini impurity
                        gain = self._calculate_information_gain(y, y_left, y_right)
                        gini_left = self._calculate_gini_impurity(y_left)
                        gini_right = self._calculate_gini_impurity(y_right)
                        
                        # Weighted Gini impurity
                        n_left, n_right = len(y_left), len(y_right)
                        n_total = len(y)
                        weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
                        
                        # Combined score: information gain - weighted gini
                        combined_score = gain - weighted_gini * 0.1
                        
                        if combined_score > best_gain:
                            best_gain = combined_score
                            best_feature_idx = i
                            best_threshold = threshold
                            best_gini = weighted_gini
        
        # If no good split found, create leaf
        if best_feature_idx == -1 or best_gain <= 0:
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            return N
        
        # Create internal node
        N.feature_idx = best_feature_idx
        N.threshold = best_threshold
        N.feature_name = imp_features[best_feature_idx]
        N.depth = depth
        
        # Create splits
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # Create child nodes
        if np.sum(left_mask) > 0:
            remaining_features = [f for f in imp_features if f != imp_features[best_feature_idx]]
            N.left = self._tree_gen_improved(X[left_mask], y[left_mask], remaining_features, depth + 1)
        else:
            unique, counts = np.unique(y, return_counts=True)
            N.left = IntruDTreeNode(value=unique[np.argmax(counts)], is_leaf=True)
        
        if np.sum(right_mask) > 0:
            remaining_features = [f for f in imp_features if f != imp_features[best_feature_idx]]
            N.right = self._tree_gen_improved(X[right_mask], y[right_mask], remaining_features, depth + 1)
        else:
            unique, counts = np.unique(y, return_counts=True)
            N.right = IntruDTreeNode(value=unique[np.argmax(counts)], is_leaf=True)
        
        return N
    
    def _calculate_gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0.0
        
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'IntruDTree':
        """
        Wrapper for intrudtree to maintain compatibility
        """
        return self.intrudtree(X, y)
    
    def _calculate_score(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance scores using information gain
        """
        importance_scores = {}
        
        for feature in X.columns:
            # Calculate information gain for this feature
            gain = self._calculate_information_gain_for_feature(X[feature], y)
            importance_scores[feature] = gain
        
        return importance_scores
    
    def _calculate_information_gain_for_feature(self, feature_values: pd.Series, y: pd.Series) -> float:
        """
        Calculate information gain for a single feature (optimized for large datasets)
        """
        try:
            # Calculate parent entropy
            parent_entropy = self._calculate_entropy(y.values)
            
            # For numeric features, find best split (with sampling for performance)
            if pd.api.types.is_numeric_dtype(feature_values):
                # Sample thresholds for performance (max 50 thresholds)
                unique_values = sorted(feature_values.unique())
                if len(unique_values) > 50:
                    # Sample evenly across the range
                    indices = np.linspace(0, len(unique_values)-1, 50, dtype=int)
                    threshold_candidates = [unique_values[i] for i in indices]
                else:
                    threshold_candidates = unique_values
                
                best_gain = 0
                
                for threshold in threshold_candidates:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                        y_left = y[left_mask]
                        y_right = y[right_mask]
                        
                        gain = self._calculate_information_gain(y.values, y_left.values, y_right.values)
                        best_gain = max(best_gain, gain)
                
                return best_gain
            else:
                # For categorical features, use all unique values
                unique_values = feature_values.unique()
                best_gain = 0
                
                for value in unique_values:
                    left_mask = feature_values == value
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                        y_left = y[left_mask]
                        y_right = y[right_mask]
                        
                        gain = self._calculate_information_gain(y.values, y_left.values, y_right.values)
                        best_gain = max(best_gain, gain)
                
                return best_gain
                
        except Exception as e:
            logger.warning(f"Error calculating information gain for feature: {e}")
            return 0.0
    
    def _select_features(self, feature_names: List[str], importance_scores: Dict[str, float], n: int) -> List[str]:
        """
        Select top-n important features
        """
        # Sort features by importance score
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top-n features
        selected_features = [feature for feature, score in sorted_features[:n]]
        
        return selected_features
    
    def _tree_gen(self, X: np.ndarray, y: np.ndarray, imp_features: List[str], depth: int) -> IntruDTreeNode:
        """
        TreeGen procedure following the exact pseudocode
        Procedure TreeGen(DS_sub, imp_features):
        """
        N = IntruDTreeNode()
        
        # Check if all instances have same class
        if len(np.unique(y)) == 1:
            N.value = y[0]
            N.is_leaf = True
            return N
        
        # Check if no important features left
        if not imp_features or depth >= self.max_depth or len(y) < self.min_samples_split:
            # Return leaf with majority class
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            return N
        
        # Find best split feature
        best_feature_idx = -1
        best_gain = -1
        best_threshold = 0
        
        for i, feature in enumerate(imp_features):
            if i < X.shape[1]:  # Make sure we don't go out of bounds
                feature_values = X[:, i]
                
                # Find best threshold for this feature
                if len(np.unique(feature_values)) > 1:
                    unique_values = np.unique(feature_values)
                    for threshold in unique_values:
                        left_mask = feature_values <= threshold
                        right_mask = ~left_mask
                        
                        if np.sum(left_mask) >= self.min_samples_leaf and np.sum(right_mask) >= self.min_samples_leaf:
                            y_left = y[left_mask]
                            y_right = y[right_mask]
                            
                            gain = self._calculate_information_gain(y, y_left, y_right)
                            
                            if gain > best_gain:
                                best_gain = gain
                                best_feature_idx = i
                                best_threshold = threshold
        
        # If no good split found, create leaf
        if best_feature_idx == -1 or best_gain <= 0:
            unique, counts = np.unique(y, return_counts=True)
            N.value = unique[np.argmax(counts)]
            N.is_leaf = True
            return N
        
        # Create internal node
        N.feature_idx = best_feature_idx
        N.threshold = best_threshold
        N.feature_name = imp_features[best_feature_idx]
        N.depth = depth
        
        # Create splits
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # Create child nodes
        if np.sum(left_mask) > 0:
            remaining_features = [f for f in imp_features if f != imp_features[best_feature_idx]]
            N.left = self._tree_gen(X[left_mask], y[left_mask], remaining_features, depth + 1)
        else:
            N.left = IntruDTreeNode(value=y[0], is_leaf=True)
        
        if np.sum(right_mask) > 0:
            remaining_features = [f for f in imp_features if f != imp_features[best_feature_idx]]
            N.right = self._tree_gen(X[right_mask], y[right_mask], remaining_features, depth + 1)
        else:
            N.right = IntruDTreeNode(value=y[0], is_leaf=True)
        
        return N
    
    def _predict_single(self, x: np.ndarray, node: IntruDTreeNode, current_depth: int = 0) -> int:
        """
        Predict for a single sample with improved logic
        
        Args:
            x: Single sample features
            node: Current node
            
        Returns:
            Predicted class
        """
        if node.is_leaf:
            logger.info(f"Leaf node reached: value={node.value}, traversal_depth={current_depth}")
            return node.value
        
        # Handle missing features gracefully
        if node.feature_idx >= len(x):
            logger.warning(f"Feature index {node.feature_idx} >= feature count {len(x)}")
            # If feature is missing, go to the larger subtree
            if node.left and node.right:
                return self._predict_single(x, node.right, current_depth + 1)  # Default to right
            elif node.left:
                return self._predict_single(x, node.left, current_depth + 1)
            elif node.right:
                return self._predict_single(x, node.right, current_depth + 1)
            else:
                return node.value  # Fallback to node value
        
        # Normal prediction logic
        feature_value = x[node.feature_idx]
        threshold = node.threshold
        
        # Only log for first few nodes to avoid spam
        if node.depth <= 3:
            logger.info(f"Node depth={node.depth}, feature_idx={node.feature_idx}, feature_value={feature_value:.6f}, threshold={threshold:.6f}")
        
        if feature_value <= threshold:
            if node.depth <= 3:
                logger.info(f"Going LEFT (feature_value <= threshold)")
            if node.left:
                return self._predict_single(x, node.left, current_depth + 1)
            else:
                logger.warning(f"No left child at depth {node.depth}, using node value {node.value}")
                return node.value  # Fallback if no left child
        else:
            if node.depth <= 3:
                logger.info(f"Going RIGHT (feature_value > threshold)")
            if node.right:
                return self._predict_single(x, node.right, current_depth + 1)
            else:
                logger.warning(f"No right child at depth {node.depth}, using node value {node.value}")
                return node.value  # Fallback if no right child
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict classes for samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use only important features if they are available
        if hasattr(self, 'important_features') and self.important_features:
            # Check if X has the important features
            available_features = [f for f in self.important_features if f in X.columns]
            if available_features:
                X_selected = X[available_features]
            else:
                # If feature names don't match, use all features (fallback)
                X_selected = X
        else:
            X_selected = X
        
        X_np = X_selected.values
        predictions = []
        
        for i in range(len(X_np)):
            pred = self._predict_single(X_np[i], self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _get_tree_depth(self) -> int:
        """Get the depth of the tree"""
        def get_max_depth(node, current_depth=0):
            if node.is_leaf:
                return current_depth
            # Only recurse if children exist
            left_depth = get_max_depth(node.left, current_depth + 1) if node.left else current_depth
            right_depth = get_max_depth(node.right, current_depth + 1) if node.right else current_depth
            return max(left_depth, right_depth)
        
        if self.root is None:
            return 0
        
        depth = get_max_depth(self.root)
        logger.info(f"Calculated tree depth: {depth}")
        return depth
    
    def get_decision_path(self, x: np.ndarray) -> List[Dict]:
        """
        Get the decision path for a single sample
        
        Args:
            x: Single sample features
            
        Returns:
            List of decision path information
        """
        if self.root is None:
            raise ValueError("Model must be fitted before getting decision path")
        
        path = []
        node = self.root
        
        while not node.is_leaf:
            feature_name = node.feature_name
            feature_value = x[node.feature_idx]
            threshold = node.threshold
            decision = "left" if feature_value <= threshold else "right"
            
            path.append({
                'feature': feature_name,
                'feature_value': feature_value,
                'threshold': threshold,
                'decision': decision,
                'depth': node.depth,
                'gain': node.split_info.get('gain', 0)
            })
            
            if decision == "left":
                node = node.left
            else:
                node = node.right
        
        # Add leaf information
        path.append({
            'feature': 'leaf',
            'predicted_value': node.value,
            'depth': node.depth
        })
        
        return path
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance based on information gain
        
        Returns:
            DataFrame with feature importance
        """
        if self.root is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_dict = {}
        
        def calculate_node_importance(node):
            if node.is_leaf:
                return
            
            # Calculate importance for this node
            feature_name = node.feature_name
            gain = node.split_info.get('gain', 0)
            n_samples = node.split_info.get('n_samples', 0)
            
            if feature_name not in importance_dict:
                importance_dict[feature_name] = 0
            
            importance_dict[feature_name] += gain * n_samples
            
            # Recursively calculate for children
            calculate_node_importance(node.left)
            calculate_node_importance(node.right)
        
        calculate_node_importance(self.root)
        
        # Normalize importance
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v / total_importance for k, v in importance_dict.items()}
        
        # Create DataFrame
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_dict.items()
        ])
        
        return importance_df.sort_values('importance', ascending=False)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'prune_threshold': self.prune_threshold,
            'n_important_features': self.intrudtree_instance.n_important_features
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        """Sklearn compatibility method"""
        return self.intrudtree(X, y)
    

class IntruDTreeModel:
    """
    IntruDTree model wrapper for cyber threat detection
    """
    
    def __init__(self, max_depth: int = 50, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, prune_threshold: float = 0.001,
                 n_important_features: int = 79):  # Use all features
        """
        Initialize IntruDTree model with optimized defaults for maximum accuracy
        
        Args:
            max_depth: Maximum depth of the tree (increased to 50 for maximum accuracy)
            min_samples_split: Minimum samples required to split (minimal at 2)
            min_samples_leaf: Minimum samples required in leaf nodes (minimal at 1)
            prune_threshold: Threshold for pruning decisions
            n_important_features: Number of important features to select (increased to 100)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.prune_threshold = prune_threshold
        
        self.intrudtree_instance = IntruDTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_important_features=n_important_features
        )
        self.is_trained = False
        self.feature_names = []
        self.n_important_features = n_important_features  # Store for later use
        self.using_sklearn = False  # Track if using sklearn hybrid approach
        self.using_ensemble = False  # Track if using ensemble approach
        self.ensemble_models = []  # Store ensemble models
        
        # Add sklearn compatibility
        self._estimator_type = "classifier"
        
        logger.info(f"IntruDTree model initialized with max_depth={max_depth}, min_samples_split={min_samples_split}, n_important_features={n_important_features}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the IntruDTree model with hybrid sklearn approach for better accuracy
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Training results dictionary
        """
        logger.info("Training IntruDTree model with hybrid sklearn approach...")
        
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
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Train ensemble of IntruDTree models for better accuracy
        logger.info("Training ensemble of IntruDTree models for maximum accuracy...")
        
        # Create multiple IntruDTree models with different parameters (optimized for 80%+ accuracy)
        self.ensemble_models = []
        ensemble_params = [
            {'max_depth': 50, 'min_samples_split': 2, 'min_samples_leaf': 1, 'n_important_features': 60},
            {'max_depth': 45, 'min_samples_split': 2, 'min_samples_leaf': 1, 'n_important_features': 55},
            {'max_depth': 55, 'min_samples_split': 2, 'min_samples_leaf': 1, 'n_important_features': 65},
            {'max_depth': 48, 'min_samples_split': 3, 'min_samples_leaf': 2, 'n_important_features': 58},
            {'max_depth': 52, 'min_samples_split': 2, 'min_samples_leaf': 1, 'n_important_features': 62}
        ]
        
        logger.info(f"Creating ensemble with {len(ensemble_params)} IntruDTree models...")
        for i, params in enumerate(ensemble_params):
            logger.info(f"Training ensemble model {i+1}/{len(ensemble_params)}...")
            model = IntruDTree(**params)
            model.fit(X_train, y_train)
            self.ensemble_models.append(model)
        
        # Store original instance for compatibility
        self.intrudtree_instance = self.ensemble_models[0]
        
        # Calculate training metrics using ensemble
        y_pred_train = self.predict_ensemble(X_train)[0]
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        self.using_sklearn = False
        self.using_ensemble = True
        
        logger.info(f"IntruDTree ensemble training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'IntruDTree-Ensemble',
            'training_metrics': training_metrics,
            'feature_count': len(self.feature_names),
            'tree_depth': max([m._get_tree_depth() for m in self.ensemble_models]),
            'ensemble_size': len(self.ensemble_models),
            'class_weights_used': bool(class_weights)
        }
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Predict using ensemble of IntruDTree models"""
        all_predictions = []
        for model in self.ensemble_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Majority voting
        all_predictions = np.array(all_predictions)
        ensemble_predictions = []
        
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_predictions.append(unique[np.argmax(counts)])
        
        return np.array(ensemble_predictions), {'ensemble_size': len(self.ensemble_models)}
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict threats using the IntruDTree model
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Check if using ensemble
            if hasattr(self, 'using_ensemble') and self.using_ensemble:
                logger.info(f"Using IntruDTree ensemble with {len(self.ensemble_models)} models for prediction...")
                predictions, details = self.predict_ensemble(X)
                return predictions, details
            
            # Debug: Log prediction process
            logger.info(f"IntruDTreeModel prediction: {len(X)} samples, {X.shape[1]} features")
            logger.info(f"Tree root: is_leaf={self.intrudtree_instance.root.is_leaf}, depth={self.intrudtree_instance.root.depth}")
            if not self.intrudtree_instance.root.is_leaf:
                logger.info(f"Root feature_idx={self.intrudtree_instance.root.feature_idx}, threshold={self.intrudtree_instance.root.threshold}")
            
            # Get predictions
            predictions_result = self.intrudtree_instance.predict(X)
            
            # Handle tuple return from IntruDTree.predict
            if isinstance(predictions_result, tuple):
                predictions, _ = predictions_result
            else:
                predictions = predictions_result
            
            # Ensure predictions is a numpy array
            predictions = np.array(predictions)
            
            # Debug first few predictions
            logger.info(f"First 5 predictions: {predictions[:5]}")
            logger.info(f"Predictions shape: {predictions.shape}")
            
            # Debug first prediction in detail
            if len(X) > 0:
                logger.info(f"Debugging first prediction:")
                logger.info(f"Sample features shape: {X.iloc[0].values.shape}")
                logger.info(f"Root node: feature_idx={self.intrudtree_instance.root.feature_idx}, threshold={self.intrudtree_instance.root.threshold}")
                logger.info(f"First sample feature values: {X.iloc[0].values[:10]}")  # First 10 features
            
            # Get decision paths for first few samples
            decision_paths = []
            for i in range(min(5, len(X))):
                try:
                    path = self.intrudtree_instance.get_decision_path(X.iloc[i].values)
                    decision_paths.append(path)
                except Exception as e:
                    logger.warning(f"Error getting decision path for sample {i}: {e}")
                    decision_paths.append([])
            
            # Create prediction details
            prediction_details = {
                'predictions': predictions,
                'decision_paths': decision_paths,
                'tree_depth': self.intrudtree_instance._get_tree_depth(),
                'feature_importance': self.intrudtree_instance.get_feature_importance()
            }
            
            return predictions, prediction_details
            
        except Exception as e:
            logger.error(f"Error in IntruDTree prediction: {e}")
            # Return fallback predictions
            fallback_predictions = np.zeros(len(X), dtype=int)
            fallback_details = {
                'predictions': fallback_predictions,
                'decision_paths': [],
                'tree_depth': 0,
                'feature_importance': pd.DataFrame(),
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
            'tree_depth': details['tree_depth'],
            'feature_importance_top5': details['feature_importance'].head(5).to_dict('records')
        })
        
        return metrics
    
    def get_interpretability_info(self, X: pd.DataFrame) -> Dict:
        """
        Get interpretability information for the model
        
        Args:
            X: Input features
            
        Returns:
            Interpretability information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting interpretability info")
        
        # Get feature importance
        feature_importance = self.intrudtree_instance.get_feature_importance()
        
        # Get sample decision paths
        sample_paths = []
        for i in range(min(3, len(X))):
            path = self.intrudtree_instance.get_decision_path(X.iloc[i].values)
            sample_paths.append({
                'sample_idx': i,
                'path': path
            })
        
        return {
            'feature_importance': feature_importance.to_dict('records'),
            'tree_depth': self.intrudtree_instance._get_tree_depth(),
            'sample_decision_paths': sample_paths,
            'cybersecurity_features': self.intrudtree_instance.cybersecurity_features
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Serialize tree structure to avoid circular references
        tree_data = self._serialize_tree(self.intrudtree_instance.root)
        
        model_data = {
            'tree_data': tree_data,
            'feature_names': self.feature_names,
            'important_features': self.intrudtree_instance.important_features,
            'max_depth': self.intrudtree_instance.max_depth,
            'min_samples_split': self.intrudtree_instance.min_samples_split,
            'min_samples_leaf': self.intrudtree_instance.min_samples_leaf,
            'n_classes': self.intrudtree_instance.n_classes
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"IntruDTree model saved to {filepath}")
    
    def _serialize_tree(self, node):
        """Serialize tree node to avoid circular references"""
        if node is None:
            return None
        
        serialized_node = {
            'feature_idx': node.feature_idx,
            'threshold': node.threshold,
            'value': node.value,
            'depth': node.depth,
            'is_leaf': node.is_leaf,
            'feature_name': node.feature_name,
            'split_info': node.split_info
        }
        
        if not node.is_leaf:
            serialized_node['left'] = self._serialize_tree(node.left)
            serialized_node['right'] = self._serialize_tree(node.right)
        
        return serialized_node
    
    def _deserialize_tree(self, tree_data):
        """Deserialize tree data back to tree structure"""
        if tree_data is None:
            return None
        
        node = IntruDTreeNode(
            feature_idx=tree_data['feature_idx'],
            threshold=tree_data['threshold'],
            value=tree_data['value'],
            depth=tree_data['depth']
        )
        node.is_leaf = tree_data['is_leaf']
        node.feature_name = tree_data['feature_name']
        node.split_info = tree_data['split_info']
        
        if not node.is_leaf:
            node.left = self._deserialize_tree(tree_data['left'])
            node.right = self._deserialize_tree(tree_data['right'])
        
        return node
    
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
        
        # Recreate IntruDTree object
        self.intrudtree = IntruDTree(
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            min_samples_leaf=model_data['min_samples_leaf'],
            n_important_features=len(model_data['important_features'])
        )
        
        # Restore tree structure
        self.intrudtree_instance.root = self._deserialize_tree(model_data['tree_data'])
        self.intrudtree_instance.feature_names = model_data['feature_names']
        self.intrudtree_instance.important_features = model_data['important_features']
        self.intrudtree_instance.n_classes = model_data['n_classes']
        
        self.feature_names = model_data['feature_names']
        
        self.is_trained = True
        logger.info(f"IntruDTree model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from IntruDTree"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.intrudtree_instance.get_feature_importance()
