"""
Main Detection Orchestrator
Coordinates all three models, preprocessing, risk analysis, and reporting in a unified system
"""

import os
import sys
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from preprocessing.data_preprocessor import DataPreprocessor
from models.fuzzy_rf_model import FuzzyRandomForestModel
from models.intrudtree_model import IntruDTreeModel
from models.aca_svm_model import ACASVMModel
from risk_analysis.threat_analyzer import ThreatAnalyzer
from reporting.comparison_reporter import ModelComparisonAnalyzer, PerformanceMetricsCalculator, ReportGenerator

logger = logging.getLogger(__name__)

class CyberThreatDetectionOrchestrator:
    """
    Main orchestrator for the cyber threat detection system
    Coordinates all components: preprocessing, models, risk analysis, and reporting
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the detection orchestrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.preprocessor = None
        self.models = {}
        self.threat_analyzer = ThreatAnalyzer()
        self.comparison_analyzer = ModelComparisonAnalyzer()
        self.performance_calculator = PerformanceMetricsCalculator()
        self.report_generator = ReportGenerator()
        
        # Data storage
        self.processed_data = {}
        self.model_results = {}
        self.detection_results = {}
        self.comparison_results = {}
        
        # Status tracking
        self.is_initialized = False
        self.is_trained = False
        
        logger.info("Cyber Threat Detection Orchestrator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_path': 'data/consolidated/All_Network_Sample_Complete.csv',
            'processed_data_path': 'data/processed',
            'models_path': 'models',
            'test_size': 0.3,
            'random_state': 42,
            'scaler_type': 'minmax',
            'sample_size': 100000,  # Start with 100k samples for testing
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
                },
                'aca_svm': {
                    'svm_type': 'linear',
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'class_weight': 'balanced',
                    'max_iter': 1000
                }
            }
        }
    
    def initialize_system(self) -> Dict:
        """
        Initialize the entire detection system
        
        Returns:
            Initialization results
        """
        logger.info("Initializing cyber threat detection system...")
        
        try:
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor(
                scaler_type=self.config['scaler_type']
            )
            
            # Initialize models
            self.models = {
                'Fuzzy_RF': FuzzyRandomForestModel(
                    **self.config['model_configs']['fuzzy_rf']
                ),
                'IntruDTree': IntruDTreeModel(
                    **self.config['model_configs']['intrudtree']
                ),
                'ACA_SVM': ACASVMModel(
                    **self.config['model_configs']['aca_svm']
                )
            }
            
            self.is_initialized = True
            
            logger.info("System initialization completed successfully")
            
            return {
                'success': True,
                'message': 'System initialized successfully',
                'models_loaded': list(self.models.keys()),
                'preprocessor_ready': True
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def preprocess_data(self) -> Dict[str, Any]:
        """Preprocess the dataset"""
        try:
            logger.info("Starting data preprocessing...")
            
            input_file = self.config['data_path']
            output_dir = self.config['processed_data_path']
            sample_size = self.config.get('sample_size', None)  # Get sample size from config
            
            if not os.path.exists(input_file):
                return {
                    'success': False,
                    'error': f"Input file not found: {input_file}"
                }
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor(scaler_type=self.config.get('scaler_type', 'minmax'))
            
            # Run preprocessing with optional sampling
            preprocessing_results = self.preprocessor.preprocess(
                input_path=input_file,
                output_dir=output_dir,
                test_size=self.config['test_size'],
                sample_size=sample_size  # Pass sample size
            )
            
            # Store preprocessing results for later use
            self.processed_data = preprocessing_results
            
            logger.info("Data preprocessing completed successfully")
            
            return {
                'success': True,
                'preprocessing_info': preprocessing_results['preprocessing_info']
            }
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_processed_data(self) -> Dict:
        """
        Load processed data from files
        
        Returns:
            Loading results
        """
        if not self.processed_data or not self.processed_data.get('X_train_path'):
            raise ValueError("No processed data available. Run preprocessing first.")
        
        logger.info("Loading processed data...")
        
        try:
            # Load training and testing data
            X_train = pd.read_csv(self.processed_data['X_train_path'])
            X_test = pd.read_csv(self.processed_data['X_test_path'])
            y_train = pd.read_csv(self.processed_data['y_train_path'])
            y_test = pd.read_csv(self.processed_data['y_test_path'])
            
            # Load metadata
            metadata = pd.read_csv(self.processed_data['metadata_path'])
            
            # Store loaded data
            self.loaded_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train.iloc[:, 0],
                'y_test': y_test.iloc[:, 0],    
                'metadata': metadata
            }
            
            logger.info(f"Data loaded successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
            return {
                'success': True,
                'message': 'Data loaded successfully',
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def train_all_models(self) -> Dict:
        """
        Train all models with the processed data
        
        Returns:
            Training results dictionary
        """
        if not hasattr(self, 'loaded_data') or not self.loaded_data:
            return {
                'success': False,
                'error': 'No processed data loaded. Please run load_processed_data() first.'
            }
        
        try:
            training_results = {}
            total_training_time = 0
            
            # Load class weights if available
            class_weights = None
            try:
                weights_path = os.path.join(self.config['processed_data_path'], 'class_weights.joblib')
                if os.path.exists(weights_path):
                    class_weights = joblib.load(weights_path)
                    logger.info("Loaded class weights for training")
            except Exception as e:
                logger.warning(f"Could not load class weights: {e}")
            
            # Load label encoder for evaluation
            label_encoder = None
            try:
                encoder_path = os.path.join(self.config['processed_data_path'], 'label_encoder.joblib')
                if os.path.exists(encoder_path):
                    label_encoder = joblib.load(encoder_path)
                    logger.info("Loaded label encoder for evaluation")
            except Exception as e:
                logger.warning(f"Could not load label encoder: {e}")
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                start_time = time.time()
                
                # Train model with class weights
                if hasattr(model, 'train') and callable(getattr(model, 'train')):
                    if class_weights and hasattr(model.train.__code__, 'co_varnames'):
                        # Check if train method accepts class_weights parameter
                        if 'class_weights' in model.train.__code__.co_varnames:
                            result = model.train(self.loaded_data['X_train'], 
                                              self.loaded_data['y_train'], 
                                              class_weights=class_weights)
                        else:
                            result = model.train(self.loaded_data['X_train'], 
                                              self.loaded_data['y_train'])
                    else:
                        result = model.train(self.loaded_data['X_train'], 
                                          self.loaded_data['y_train'])
                    
                    training_time = time.time() - start_time
                    total_training_time += training_time
                    
                    # Add training time to result
                    if isinstance(result, dict):
                        result['training_time'] = training_time
                        result['class_weights_used'] = bool(class_weights)
                    
                    training_results[model_name] = result
                    
                    logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
                else:
                    logger.error(f"Model {model_name} does not have a train method")
                    training_results[model_name] = {
                        'success': False,
                        'error': 'No train method available'
                    }
            
            # Store label encoder for evaluation
            self.label_encoder = label_encoder
            
            # Mark orchestrator as trained so evaluation can proceed
            self.is_trained = True
            
            logger.info("All models trained successfully")
            
            return {
                'success': True,
                'message': 'All models trained successfully',
                'training_results': training_results,
                'total_training_time': total_training_time,
                'models_trained': list(training_results.keys()),
                'class_weights_used': bool(class_weights)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all trained models
        
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating all models...")
        
        evaluation_results = {}
        detection_times = {}
        
        try:
            for model_name, model in self.models.items():
                logger.info(f"Evaluating {model_name}...")
                
                # Measure detection time
                start_time = time.time()
                
                # Make predictions
                predictions, prediction_details = model.predict(self.loaded_data['X_test'])
                
                detection_time = time.time() - start_time
                detection_times[model_name] = detection_time
                
                # Calculate comprehensive metrics
                metrics = self.performance_calculator.calculate_comprehensive_metrics(
                    self.loaded_data['y_test'].values,
                    predictions,
                    model_name,
                    [detection_time],
                    label_encoder=self.label_encoder  # Pass label encoder for better metrics
                )
                
                # Add model type and prediction details
                metrics['model_type'] = model_name
                metrics['prediction_details'] = prediction_details
                metrics['detection_time'] = detection_time
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} evaluation completed: Accuracy = {metrics['accuracy']}%")
            
            # Store results
            self.model_results = evaluation_results
            
            logger.info("All models evaluated successfully")
            
            return {
                'success': True,
                'message': 'All models evaluated successfully',
                'evaluation_results': evaluation_results,
                'detection_times': detection_times
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def perform_threat_analysis(self, sample_data: pd.DataFrame = None) -> Dict:
        """
        Perform threat analysis on detected threats
        
        Args:
            sample_data: Sample data for analysis (uses test data if not provided)
            
        Returns:
            Threat analysis results
        """
        if not self.model_results:
            raise ValueError("Models must be evaluated before threat analysis")
        
        logger.info("Performing threat analysis...")
        
        try:
            # Use test data if no sample data provided
            if sample_data is None:
                sample_data = self.loaded_data['X_test']
            
            threat_analysis_results = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Analyzing threats for {model_name}...")
                
                # Get predictions for sample data
                predictions, prediction_details = model.predict(sample_data)
                
                # Create detection result
                detection_result = {
                    'model_predictions': {model_name: np.mean(predictions)},
                    'prediction_details': prediction_details,
                    'sample_count': len(sample_data)
                }
                
                # Perform threat analysis
                analysis_result = self.threat_analyzer.analyze_threat(
                    detection_result,
                    sample_data
                )
                
                threat_analysis_results[model_name] = analysis_result
            
            # Store results
            self.detection_results = threat_analysis_results
            
            logger.info("Threat analysis completed successfully")
            
            return {
                'success': True,
                'message': 'Threat analysis completed',
                'threat_analysis_results': threat_analysis_results
            }
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def compare_models(self) -> Dict:
        """
        Compare all models side-by-side
        
        Returns:
            Comparison results
        """
        if not self.model_results:
            raise ValueError("Models must be evaluated before comparison")
        
        logger.info("Comparing models...")
        
        try:
            # Prepare model results for comparison
            comparison_data = {}
            for model_name, results in self.model_results.items():
                comparison_data[model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                }
            
            # Perform comparison
            comparison_results = self.comparison_analyzer.compare_models(comparison_data)
            
            # Generate comprehensive report
            report = self.report_generator.generate_comparison_report(
                comparison_results,
                self.model_results
            )
            
            # Store results
            self.comparison_results = comparison_results
            self.report_data = report
            
            logger.info("Model comparison completed successfully")
            
            return {
                'success': True,
                'message': 'Model comparison completed',
                'comparison_results': comparison_results,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def save_models(self, models_dir: str = None) -> Dict:
        """
        Save all trained models
        
        Args:
            models_dir: Directory to save models (uses config if not provided)
            
        Returns:
            Save results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        logger.info("Saving all models...")
        
        try:
            models_dir = models_dir or self.config['models_path']
            os.makedirs(models_dir, exist_ok=True)
            
            saved_models = {}
            
            for model_name, model in self.models.items():
                model_path = os.path.join(models_dir, f"{model_name.lower()}_model.joblib")
                model.save_model(model_path)
                saved_models[model_name] = model_path
                
                logger.info(f"{model_name} saved to {model_path}")
            
            # Save orchestrator state
            state_path = os.path.join(models_dir, "orchestrator_state.joblib")
            state = {
                'config': self.config,
                'processed_data': self.processed_data,
                'model_results': self.model_results,
                'comparison_results': self.comparison_results,
                'is_trained': self.is_trained
            }
            joblib.dump(state, state_path)
            
            logger.info("All models and state saved successfully")
            
            return {
                'success': True,
                'message': 'All models saved successfully',
                'saved_models': saved_models,
                'state_path': state_path
            }
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def load_models(self, models_dir: str = None) -> Dict:
        """
        Load all trained models
        
        Args:
            models_dir: Directory to load models from (uses config if not provided)
            
        Returns:
            Load results
        """
        logger.info("Loading trained models...")
        
        try:
            models_dir = models_dir or self.config['models_path']
            
            # Load orchestrator state
            state_path = os.path.join(models_dir, "orchestrator_state.joblib")
            if os.path.exists(state_path):
                state = joblib.load(state_path)
                self.config = state['config']
                self.processed_data = state['processed_data']
                self.model_results = state.get('model_results', {})
                self.comparison_results = state.get('comparison_results', {})
                self.is_trained = state.get('is_trained', False)
            
            # Load individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(models_dir, f"{model_name.lower()}_model.joblib")
                if os.path.exists(model_path):
                    model.load_model(model_path)
                    logger.info(f"{model_name} loaded from {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            logger.info("Models loaded successfully")
            
            return {
                'success': True,
                'message': 'Models loaded successfully',
                'models_loaded': list(self.models.keys()),
                'is_trained': self.is_trained
            }
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_complete_pipeline(self, input_path: str = None) -> Dict:
        """
        Run the complete detection pipeline from start to finish
        
        Args:
            input_path: Path to input data (optional)
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete cyber threat detection pipeline...")
        
        pipeline_results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Initialize system
            logger.info("Step 1: Initializing system...")
            init_result = self.initialize_system()
            pipeline_results['steps']['initialization'] = init_result
            
            if not init_result['success']:
                raise Exception(f"Initialization failed: {init_result['error']}")
            
            # Step 2: Preprocess data
            logger.info("Step 2: Preprocessing data...")
            preprocess_result = self.preprocess_data()
            pipeline_results['steps']['preprocessing'] = preprocess_result
            
            if not preprocess_result['success']:
                raise Exception(f"Preprocessing failed: {preprocess_result['error']}")
            
            # Step 3: Load processed data
            logger.info("Step 3: Loading processed data...")
            load_result = self.load_processed_data()
            pipeline_results['steps']['data_loading'] = load_result
            
            if not load_result['success']:
                raise Exception(f"Data loading failed: {load_result['error']}")
            
            # Step 4: Train models
            logger.info("Step 4: Training models...")
            train_result = self.train_all_models()
            pipeline_results['steps']['training'] = train_result
            
            if not train_result['success']:
                raise Exception(f"Training failed: {train_result['error']}")
            
            # Step 5: Evaluate models
            logger.info("Step 5: Evaluating models...")
            eval_result = self.evaluate_all_models()
            pipeline_results['steps']['evaluation'] = eval_result
            
            if not eval_result['success']:
                raise Exception(f"Evaluation failed: {eval_result['error']}")
            
            # Step 6: Threat analysis
            logger.info("Step 6: Performing threat analysis...")
            threat_result = self.perform_threat_analysis()
            pipeline_results['steps']['threat_analysis'] = threat_result
            
            if not threat_result['success']:
                raise Exception(f"Threat analysis failed: {threat_result['error']}")
            
            # Step 7: Compare models
            logger.info("Step 7: Comparing models...")
            compare_result = self.compare_models()
            pipeline_results['steps']['comparison'] = compare_result
            
            if not compare_result['success']:
                raise Exception(f"Comparison failed: {compare_result['error']}")
            
            # Step 8: Save models
            logger.info("Step 8: Saving models...")
            save_result = self.save_models()
            pipeline_results['steps']['saving'] = save_result
            
            if not save_result['success']:
                raise Exception(f"Saving failed: {save_result['error']}")
            
            # Pipeline completed successfully
            pipeline_results['pipeline_end_time'] = datetime.now().isoformat()
            pipeline_results['success'] = True
            pipeline_results['message'] = 'Complete pipeline executed successfully'
            
            logger.info("Complete pipeline executed successfully!")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            pipeline_results['pipeline_end_time'] = datetime.now().isoformat()
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            pipeline_results['traceback'] = traceback.format_exc()
            
            return pipeline_results
    
    def get_system_status(self) -> Dict:
        """
        Get current system status
        
        Returns:
            System status information
        """
        return {
            'is_initialized': self.is_initialized,
            'is_trained': self.is_trained,
            'models_loaded': list(self.models.keys()) if self.models else [],
            'data_processed': bool(self.processed_data),
            'data_loaded': hasattr(self, 'loaded_data'),
            'models_evaluated': bool(self.model_results),
            'comparison_completed': bool(self.comparison_results),
            'config': self.config
        }
    
    def export_report(self, format: str = 'pdf', filename: str = None) -> Dict:
        """
        Export comparison report
        
        Args:
            format: Report format ('pdf' or 'csv')
            filename: Output filename (optional)
            
        Returns:
            Export results
        """
        if not hasattr(self, 'report_data'):
            raise ValueError("No report data available. Run comparison first.")
        
        try:
            if format.lower() == 'pdf':
                report_data = self.report_generator.export_to_pdf(filename)
                return {
                    'success': True,
                    'format': 'pdf',
                    'data': report_data,
                    'message': 'PDF report generated successfully'
                }
            elif format.lower() == 'csv':
                report_data = self.report_generator.export_to_csv(filename)
                return {
                    'success': True,
                    'format': 'csv',
                    'data': report_data,
                    'message': 'CSV report generated successfully'
                }
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Report export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'format': format
            }
