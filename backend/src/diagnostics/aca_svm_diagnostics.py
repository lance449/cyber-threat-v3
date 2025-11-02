"""
Diagnostic tools for ACA + SVM model
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.aca_svm_model import ACAPatternMatcher, ACAFeatureExtractor, ACASVMModel

logger = logging.getLogger(__name__)

class ACASVMDiagnostics:
    """
    Diagnostic tools for ACA + SVM model
    """
    
    def __init__(self, model: ACASVMModel = None):
        """
        Initialize diagnostics
        
        Args:
            model: ACA + SVM model instance
        """
        self.model = model
        self.diagnostic_results = {}
        
    def verify_aca_automaton(self) -> Dict[str, Any]:
        """
        Verify ACA automaton construction and functionality
        
        Returns:
            Diagnostic results
        """
        logger.info("Verifying ACA automaton...")
        
        results = {
            'automaton_built': False,
            'patterns_loaded': False,
            'pattern_count': 0,
            'test_matches': [],
            'automaton_size': 0,
            'errors': []
        }
        
        try:
            # Check if automaton is built
            if self.model and self.model.pattern_matcher.automaton:
                results['automaton_built'] = True
                results['pattern_count'] = len(self.model.pattern_matcher.patterns)
                results['patterns_loaded'] = True
                
                # Test pattern matching
                test_texts = [
                    "This contains malware_signature_1",
                    "Normal network traffic",
                    "trojan_horse_pattern detected",
                    "sql_injection attempt"
                ]
                
                for text in test_texts:
                    matches = self.model.pattern_matcher.match_patterns(text)
                    results['test_matches'].append({
                        'text': text,
                        'matches': len(matches),
                        'patterns_found': [match[2] for match in matches]
                    })
                
                # Get automaton size (approximate)
                results['automaton_size'] = len(self.model.pattern_matcher.patterns)
                
            else:
                results['errors'].append("Automaton not built or model not available")
                
        except Exception as e:
            results['errors'].append(f"Automaton verification failed: {str(e)}")
            logger.error(f"ACA automaton verification failed: {e}")
        
        self.diagnostic_results['aca_automaton'] = results
        return results
    
    def verify_feature_extraction(self, sample_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Verify ACA feature extraction
        
        Args:
            sample_data: Sample data for testing
            
        Returns:
            Diagnostic results
        """
        logger.info("Verifying ACA feature extraction...")
        
        results = {
            'feature_extraction_works': False,
            'feature_count': 0,
            'feature_types': {},
            'sample_features': [],
            'errors': []
        }
        
        try:
            if self.model and self.model.feature_extractor:
                # Create sample data if not provided
                if sample_data is None:
                    sample_data = pd.DataFrame({
                        'Flow Duration': [100, 200, 150],
                        'Packet Length Mean': [1000, 1200, 800],
                        'Flow Bytes/s': [5000, 6000, 4000],
                        'Protocol': ['TCP', 'UDP', 'TCP'],
                        'Src Port': [80, 443, 22]
                    })
                
                # Test feature extraction
                sample_text = ' '.join([str(sample_data.iloc[0][col]) for col in sample_data.columns])
                
                # Extract different feature types
                binary_features = self.model.feature_extractor.extract_binary_features(sample_text)
                count_features = self.model.feature_extractor.extract_count_features(sample_text)
                frequency_features = self.model.feature_extractor.extract_frequency_features(sample_text)
                aggregated_features = self.model.feature_extractor.extract_aggregated_features(sample_text)
                all_features = self.model.feature_extractor.extract_all_features(sample_text)
                
                results['feature_extraction_works'] = True
                results['feature_count'] = len(all_features)
                results['feature_types'] = {
                    'binary': len(binary_features),
                    'count': len(count_features),
                    'frequency': len(frequency_features),
                    'aggregated': len(aggregated_features),
                    'total': len(all_features)
                }
                
                # Store sample features
                results['sample_features'] = {
                    'binary': binary_features.tolist(),
                    'count': count_features.tolist(),
                    'frequency': frequency_features.tolist(),
                    'aggregated': aggregated_features.tolist()
                }
                
                # Verify feature properties
                if len(binary_features) > 0:
                    assert np.all((binary_features == 0) | (binary_features == 1)), "Binary features should be 0 or 1"
                
                if len(count_features) > 0:
                    assert np.all(count_features >= 0), "Count features should be non-negative"
                
                if len(frequency_features) > 0:
                    assert np.all(frequency_features >= 0), "Frequency features should be non-negative"
                
            else:
                results['errors'].append("Feature extractor not available")
                
        except Exception as e:
            results['errors'].append(f"Feature extraction verification failed: {str(e)}")
            logger.error(f"ACA feature extraction verification failed: {e}")
        
        self.diagnostic_results['feature_extraction'] = results
        return results
    
    def verify_svm_model(self) -> Dict[str, Any]:
        """
        Verify SVM model functionality
        
        Returns:
            Diagnostic results
        """
        logger.info("Verifying SVM model...")
        
        results = {
            'model_trained': False,
            'model_type': None,
            'feature_count': 0,
            'class_count': 0,
            'training_metrics': {},
            'pipeline_components': [],
            'errors': []
        }
        
        try:
            if self.model:
                results['model_trained'] = self.model.is_trained
                results['model_type'] = self.model.svm_type
                
                if self.model.is_trained:
                    results['feature_count'] = len(self.model.feature_names)
                    results['class_count'] = len(self.model.class_names)
                    results['training_metrics'] = self.model.training_metrics
                    
                    # Check pipeline components
                    if self.model.pipeline:
                        results['pipeline_components'] = list(self.model.pipeline.named_steps.keys())
                    
                    # Test prediction capability
                    if hasattr(self.model, 'predict'):
                        results['prediction_available'] = True
                    else:
                        results['errors'].append("Predict method not available")
                else:
                    results['errors'].append("Model not trained")
            else:
                results['errors'].append("Model not available")
                
        except Exception as e:
            results['errors'].append(f"SVM model verification failed: {str(e)}")
            logger.error(f"SVM model verification failed: {e}")
        
        self.diagnostic_results['svm_model'] = results
        return results
    
    def test_end_to_end_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Test end-to-end pipeline
        
        Args:
            X_test: Test features
            y_test: Test labels (optional)
            
        Returns:
            Diagnostic results
        """
        logger.info("Testing end-to-end pipeline...")
        
        results = {
            'pipeline_works': False,
            'prediction_time': 0,
            'prediction_shape': None,
            'feature_shapes': {},
            'errors': []
        }
        
        try:
            if self.model and self.model.is_trained:
                import time
                
                # Test prediction
                start_time = time.time()
                predictions, details = self.model.predict(X_test)
                prediction_time = time.time() - start_time
                
                results['pipeline_works'] = True
                results['prediction_time'] = prediction_time
                results['prediction_shape'] = predictions.shape
                
                # Check feature shapes
                if 'aca_features_shape' in details:
                    results['feature_shapes']['aca_features'] = details['aca_features_shape']
                if 'combined_features_shape' in details:
                    results['feature_shapes']['combined_features'] = details['combined_features_shape']
                
                # Test with logging if available
                if hasattr(self.model, 'predict_with_logging'):
                    flow_ids = [f"test_flow_{i}" for i in range(len(X_test))]
                    predictions_logged, details_logged = self.model.predict_with_logging(X_test, flow_ids)
                    
                    results['logging_works'] = True
                    results['logged_detections'] = len(details_logged.get('logged_detections', []))
                
                # Evaluate if labels provided
                if y_test is not None:
                    evaluation = self.model.evaluate(X_test, y_test)
                    results['evaluation_metrics'] = evaluation
                
            else:
                results['errors'].append("Model not trained or not available")
                
        except Exception as e:
            results['errors'].append(f"End-to-end pipeline test failed: {str(e)}")
            logger.error(f"End-to-end pipeline test failed: {e}")
        
        self.diagnostic_results['end_to_end'] = results
        return results
    
    def check_data_quality(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Check data quality for ACA + SVM model
        
        Args:
            X: Feature data
            y: Label data (optional)
            
        Returns:
            Data quality results
        """
        logger.info("Checking data quality...")
        
        results = {
            'data_shape': X.shape,
            'missing_values': {},
            'data_types': {},
            'numeric_features': 0,
            'categorical_features': 0,
            'infinite_values': 0,
            'class_distribution': {},
            'errors': []
        }
        
        try:
            # Check missing values
            missing_counts = X.isnull().sum()
            results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
            
            # Check data types
            results['data_types'] = X.dtypes.value_counts().to_dict()
            
            # Count numeric vs categorical features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            results['numeric_features'] = len(numeric_cols)
            results['categorical_features'] = len(categorical_cols)
            
            # Check for infinite values
            numeric_data = X.select_dtypes(include=[np.number])
            infinite_count = np.isinf(numeric_data).sum().sum()
            results['infinite_values'] = infinite_count
            
            # Check class distribution if labels provided
            if y is not None:
                class_counts = y.value_counts()
                results['class_distribution'] = class_counts.to_dict()
                
                # Check for class imbalance
                min_class_count = class_counts.min()
                max_class_count = class_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                results['class_imbalance_ratio'] = imbalance_ratio
                
                if imbalance_ratio > 10:
                    results['warnings'] = results.get('warnings', [])
                    results['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
        except Exception as e:
            results['errors'].append(f"Data quality check failed: {str(e)}")
            logger.error(f"Data quality check failed: {e}")
        
        self.diagnostic_results['data_quality'] = results
        return results
    
    def generate_diagnostic_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive diagnostic report
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Report content
        """
        logger.info("Generating diagnostic report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ACA + SVM MODEL DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model information
        if self.model:
            model_info = self.model.get_model_info()
            report_lines.append("MODEL INFORMATION:")
            report_lines.append("-" * 40)
            for key, value in model_info.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Diagnostic results
        for test_name, results in self.diagnostic_results.items():
            report_lines.append(f"{test_name.upper().replace('_', ' ')}:")
            report_lines.append("-" * 40)
            
            for key, value in results.items():
                if key == 'errors' and value:
                    report_lines.append(f"  {key}:")
                    for error in value:
                        report_lines.append(f"    - {error}")
                elif key == 'warnings' and value:
                    report_lines.append(f"  {key}:")
                    for warning in value:
                        report_lines.append(f"    - {warning}")
                else:
                    report_lines.append(f"  {key}: {value}")
            
            report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        
        total_tests = len(self.diagnostic_results)
        failed_tests = sum(1 for results in self.diagnostic_results.values() if results.get('errors'))
        
        report_lines.append(f"Total diagnostic tests: {total_tests}")
        report_lines.append(f"Failed tests: {failed_tests}")
        report_lines.append(f"Success rate: {((total_tests - failed_tests) / total_tests * 100):.1f}%")
        
        if failed_tests > 0:
            report_lines.append("\nFAILED TESTS:")
            for test_name, results in self.diagnostic_results.items():
                if results.get('errors'):
                    report_lines.append(f"  - {test_name}: {', '.join(results['errors'])}")
        
        report_content = '\n'.join(report_lines)
        
        # Save to file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Diagnostic report saved to {output_file}")
        
        return report_content
    
    def run_all_diagnostics(self, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Run all diagnostic tests
        
        Args:
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            All diagnostic results
        """
        logger.info("Running all ACA + SVM diagnostics...")
        
        # Run individual diagnostics
        self.verify_aca_automaton()
        self.verify_feature_extraction()
        self.verify_svm_model()
        
        if X_test is not None:
            self.test_end_to_end_pipeline(X_test, y_test)
            self.check_data_quality(X_test, y_test)
        
        # Generate report
        report = self.generate_diagnostic_report()
        
        return {
            'diagnostic_results': self.diagnostic_results,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main function for running diagnostics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACA + SVM Model Diagnostics')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--test_data', type=str, help='Path to test data CSV')
    parser.add_argument('--output', type=str, default='aca_svm_diagnostics_report.txt', help='Output report file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Load model if path provided
    model = None
    if args.model_path and os.path.exists(args.model_path):
        try:
            model = ACASVMModel()
            model.load_model(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    
    # Load test data if path provided
    X_test = None
    y_test = None
    if args.test_data and os.path.exists(args.test_data):
        try:
            test_df = pd.read_csv(args.test_data)
            # Assume last column is label
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            logger.info(f"Loaded test data: {X_test.shape}")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return
    
    # Run diagnostics
    diagnostics = ACASVMDiagnostics(model)
    results = diagnostics.run_all_diagnostics(X_test, y_test)
    
    # Print report
    print(results['report'])
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(results['report'])
    
    logger.info(f"Diagnostics completed. Report saved to {args.output}")

if __name__ == '__main__':
    main()
