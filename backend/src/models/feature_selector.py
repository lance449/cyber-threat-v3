"""
Advanced Feature Selection for Cyber Threat Detection
Optimizes feature selection for each model individually
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """
    Advanced feature selection optimized for cybersecurity
    """
    
    def __init__(self, n_features: int = 50):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of features to select
        """
        self.n_features = n_features
        self.selected_features = {}
        self.feature_scores = {}
        self.selector_methods = {}
        
        logger.info(f"Advanced feature selector initialized for {n_features} features")
    
    def select_features_for_model(self, X: pd.DataFrame, y: pd.Series, 
                                 model_name: str, model_type: str) -> List[str]:
        """
        Select optimal features for a specific model
        
        Args:
            X: Feature matrix
            y: Labels
            model_name: Name of the model
            model_type: Type of model ('tree', 'linear', 'ensemble')
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features for {model_name} ({model_type})")
        
        # Use different selection methods based on model type
        if model_type == 'tree':
            selected_features = self._select_for_tree_model(X, y)
        elif model_type == 'linear':
            selected_features = self._select_for_linear_model(X, y)
        elif model_type == 'ensemble':
            selected_features = self._select_for_ensemble_model(X, y)
        else:
            selected_features = self._select_generic_features(X, y)
        
        self.selected_features[model_name] = selected_features
        logger.info(f"Selected {len(selected_features)} features for {model_name}")
        
        return selected_features
    
    def _select_for_tree_model(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Feature selection optimized for tree-based models
        """
        logger.info("Using tree-optimized feature selection")
        
        # Method 1: Mutual Information (good for non-linear relationships)
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
        X_mi = mi_selector.fit_transform(X, y)
        mi_features = X.columns[mi_selector.get_support()].tolist()
        mi_scores = mi_selector.scores_
        
        # Method 2: Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Method 3: F-score (statistical significance)
        f_selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X.shape[1]))
        f_selector.fit(X, y)
        f_features = X.columns[f_selector.get_support()].tolist()
        f_scores = f_selector.scores_
        
        # Combine scores
        feature_scores = {}
        for i, feature in enumerate(X.columns):
            score = 0
            if feature in mi_features:
                score += mi_scores[i] / np.max(mi_scores)  # Normalize
            score += rf_importance[i]  # Already normalized
            if feature in f_features:
                score += f_scores[i] / np.max(f_scores)  # Normalize
            feature_scores[feature] = score
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:self.n_features]]
        
        self.feature_scores['tree'] = feature_scores
        self.selector_methods['tree'] = 'mutual_info + rf_importance + f_score'
        
        return selected_features
    
    def _select_for_linear_model(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Feature selection optimized for linear models
        """
        logger.info("Using linear-optimized feature selection")
        
        # Method 1: F-score (statistical significance)
        f_selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X.shape[1]))
        f_selector.fit(X, y)
        f_features = X.columns[f_selector.get_support()].tolist()
        f_scores = f_selector.scores_
        
        # Method 2: Recursive Feature Elimination with Logistic Regression
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            rfe_selector = RFE(estimator=lr, n_features_to_select=min(self.n_features, X.shape[1]))
            rfe_selector.fit(X, y)
            rfe_features = X.columns[rfe_selector.get_support()].tolist()
        except Exception as e:
            logger.warning(f"RFE failed: {e}")
            rfe_features = []
        
        # Method 3: Correlation-based selection
        corr_features = self._select_correlation_features(X, y)
        
        # Combine features
        all_features = set(f_features + rfe_features + corr_features)
        
        # Score features
        feature_scores = {}
        for feature in all_features:
            score = 0
            if feature in f_features:
                score += 1.0
            if feature in rfe_features:
                score += 1.0
            if feature in corr_features:
                score += 0.5
            feature_scores[feature] = score
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:self.n_features]]
        
        self.feature_scores['linear'] = feature_scores
        self.selector_methods['linear'] = 'f_score + rfe + correlation'
        
        return selected_features
    
    def _select_for_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Feature selection optimized for ensemble models
        """
        logger.info("Using ensemble-optimized feature selection")
        
        # Combine tree and linear methods
        tree_features = self._select_for_tree_model(X, y)
        linear_features = self._select_for_linear_model(X, y)
        
        # Combine and deduplicate
        combined_features = list(set(tree_features + linear_features))
        
        # If we have too many features, select the best ones
        if len(combined_features) > self.n_features:
            # Use cross-validation to select the best subset
            best_features = self._cv_feature_selection(X[combined_features], y, combined_features)
            selected_features = best_features[:self.n_features]
        else:
            selected_features = combined_features
        
        self.feature_scores['ensemble'] = {f: 1.0 for f in selected_features}
        self.selector_methods['ensemble'] = 'tree + linear + cv_selection'
        
        return selected_features
    
    def _select_generic_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Generic feature selection
        """
        logger.info("Using generic feature selection")
        
        # Use mutual information as default
        selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_scores['generic'] = dict(zip(X.columns, selector.scores_))
        self.selector_methods['generic'] = 'mutual_info'
        
        return selected_features
    
    def _select_correlation_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features based on correlation with target
        """
        correlations = []
        for feature in X.columns:
            try:
                corr = abs(X[feature].corr(pd.Series(y)))
                if not np.isnan(corr):
                    correlations.append((feature, corr))
            except:
                continue
        
        # Sort by correlation and select top features
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, corr in correlations[:self.n_features//3]]
        
        return top_features
    
    def _cv_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                            candidate_features: List[str]) -> List[str]:
        """
        Use cross-validation to select best features
        """
        logger.info("Performing cross-validation feature selection")
        
        # Test different feature subsets
        best_score = 0
        best_features = candidate_features[:self.n_features]
        
        # Test subsets of different sizes
        for n in range(10, min(len(candidate_features), self.n_features) + 1, 5):
            subset_features = candidate_features[:n]
            X_subset = X[subset_features]
            
            try:
                # Use Random Forest for evaluation
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                scores = cross_val_score(rf, X_subset, y, cv=3, scoring='f1_weighted', n_jobs=-1)
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_features = subset_features
                    
            except Exception as e:
                logger.warning(f"CV failed for {n} features: {e}")
                continue
        
        logger.info(f"Best CV score: {best_score:.4f} with {len(best_features)} features")
        return best_features
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_scores:
            return pd.DataFrame()
        
        scores = self.feature_scores[model_name]
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': score}
            for feature, score in scores.items()
        ])
        
        return importance_df.sort_values('importance', ascending=False)
    
    def get_selection_summary(self) -> Dict:
        """
        Get summary of feature selection results
        
        Returns:
            Dictionary with selection summary
        """
        summary = {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'selector_methods': self.selector_methods,
            'total_features_selected': sum(len(features) for features in self.selected_features.values())
        }
        
        return summary
