"""
Advanced Ensemble Model for Cyber Threat Detection
Combines multiple models with sophisticated voting and stacking techniques
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os

logger = logging.getLogger(__name__)

class AdvancedEnsembleModel:
    """
    Advanced ensemble model combining Fuzzy_RF, IntruDTree, and ACA_SVM
    """
    
    def __init__(self, models: Dict[str, Any], ensemble_method: str = 'voting'):
        """
        Initialize advanced ensemble model
        
        Args:
            models: Dictionary of trained models
            ensemble_method: 'voting', 'stacking', or 'weighted'
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.ensemble_model = None
        self.model_weights = {}
        self.is_trained = False
        self.feature_names = []
        
        logger.info(f"Advanced ensemble model initialized with {ensemble_method} method")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train the ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training results
        """
        logger.info("Training advanced ensemble model...")
        
        self.feature_names = X_train.columns.tolist()
        
        if self.ensemble_method == 'voting':
            return self._train_voting_ensemble(X_train, y_train)
        elif self.ensemble_method == 'stacking':
            return self._train_stacking_ensemble(X_train, y_train)
        elif self.ensemble_method == 'weighted':
            return self._train_weighted_ensemble(X_train, y_train)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _train_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train voting ensemble
        """
        logger.info("Training voting ensemble...")
        
        # Create voting classifier
        estimators = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                estimators.append((name, model))
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_pred_train = self.ensemble_model.predict(X_train)
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        
        logger.info(f"Voting ensemble training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'Advanced_Ensemble_Voting',
            'training_metrics': training_metrics,
            'ensemble_method': self.ensemble_method,
            'models_count': len(estimators)
        }
    
    def _train_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train stacking ensemble
        """
        logger.info("Training stacking ensemble...")
        
        # Create stacking classifier
        estimators = []
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                estimators.append((name, model))
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        self.ensemble_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=-1
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_pred_train = self.ensemble_model.predict(X_train)
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        
        logger.info(f"Stacking ensemble training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'Advanced_Ensemble_Stacking',
            'training_metrics': training_metrics,
            'ensemble_method': self.ensemble_method,
            'models_count': len(estimators)
        }
    
    def _train_weighted_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train weighted ensemble based on individual model performance
        """
        logger.info("Training weighted ensemble...")
        
        # Evaluate individual models
        model_scores = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                try:
                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                    model_scores[name] = np.mean(cv_scores)
                    logger.info(f"{name} CV F1 score: {model_scores[name]:.4f}")
                except Exception as e:
                    logger.warning(f"Could not evaluate {name}: {e}")
                    model_scores[name] = 0.0
        
        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            # Equal weights if no scores available
            self.model_weights = {name: 1.0/len(model_scores) for name in model_scores.keys()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        # Calculate training metrics using weighted predictions
        y_pred_train = self._weighted_predict(X_train)
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
            'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        }
        
        self.is_trained = True
        
        logger.info(f"Weighted ensemble training completed. Accuracy: {training_metrics['accuracy']:.4f}")
        
        return {
            'model_type': 'Advanced_Ensemble_Weighted',
            'training_metrics': training_metrics,
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'models_count': len(model_scores)
        }
    
    def _weighted_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted predictions
        """
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict') and name in self.model_weights:
                try:
                    pred = model.predict(X)
                    predictions[name] = pred
                except Exception as e:
                    logger.warning(f"Error getting predictions from {name}: {e}")
                    continue
        
        if not predictions:
            # Fallback to majority class
            return np.zeros(len(X), dtype=int)
        
        # Weighted voting
        final_predictions = np.zeros(len(X), dtype=int)
        unique_classes = set()
        
        for pred in predictions.values():
            unique_classes.update(pred)
        
        unique_classes = sorted(list(unique_classes))
        
        for i in range(len(X)):
            class_votes = {cls: 0.0 for cls in unique_classes}
            
            for name, pred in predictions.items():
                weight = self.model_weights[name]
                class_votes[pred[i]] += weight
            
            # Select class with highest weighted vote
            final_predictions[i] = max(class_votes.items(), key=lambda x: x[1])[0]
        
        return final_predictions
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict using ensemble model
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, prediction_details)
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        try:
            if self.ensemble_method == 'weighted':
                predictions = self._weighted_predict(X)
            else:
                predictions = self.ensemble_model.predict(X)
            
            # Get individual model predictions for analysis
            individual_predictions = {}
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    try:
                        individual_predictions[name] = model.predict(X)
                    except Exception as e:
                        logger.warning(f"Error getting predictions from {name}: {e}")
            
            prediction_details = {
                'ensemble_predictions': predictions,
                'individual_predictions': individual_predictions,
                'ensemble_method': self.ensemble_method,
                'model_weights': self.model_weights if self.ensemble_method == 'weighted' else None
            }
            
            return predictions, prediction_details
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            # Fallback predictions
            fallback_predictions = np.zeros(len(X), dtype=int)
            fallback_details = {
                'ensemble_predictions': fallback_predictions,
                'individual_predictions': {},
                'ensemble_method': self.ensemble_method,
                'error': str(e)
            }
            return fallback_predictions, fallback_details
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate ensemble model
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before evaluation")
        
        predictions, details = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        # Add individual model performance
        individual_metrics = {}
        for name, pred in details['individual_predictions'].items():
            individual_metrics[name] = {
                'accuracy': accuracy_score(y_test, pred),
                'precision': precision_score(y_test, pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, pred, average='weighted', zero_division=0)
            }
        
        metrics['individual_model_metrics'] = individual_metrics
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before saving")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Advanced ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
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
        
        self.ensemble_model = model_data['ensemble_model']
        self.ensemble_method = model_data['ensemble_method']
        self.model_weights = model_data['model_weights']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Advanced ensemble model loaded from {filepath}")
