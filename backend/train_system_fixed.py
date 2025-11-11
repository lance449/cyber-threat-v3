#!/usr/bin/env python3
"""
Training Script for Cyber Threat Detection System
Demonstrates the complete pipeline: preprocessing, training, evaluation, and reporting
"""

import os
import sys
import logging
import time
from datetime import datetime
import traceback
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the orchestrator
from detection_orchestrator import CyberThreatDetectionOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_optimized_config_for_balanced_dataset():
    """Get optimized configuration specifically for balanced dataset (11k per class)
    This configuration targets 80%+ accuracy while maintaining algorithm integrity"""
    return {
        'fuzzy_rf': {
            'n_estimators': 500,        # More trees for better ensemble (target 80%+)
            'max_depth': 35,            # Deeper trees to capture complex patterns
            'min_samples_split': 2,     # Allow fine-grained splits
            'min_samples_leaf': 1,       # Minimal leaf size for detailed patterns
            'max_features': 'sqrt',     # Feature diversity per tree
            'bootstrap': True,
            'class_weight': None,       # No class weights needed (already balanced)
            'max_samples': 1.0,          # Use all samples (balanced dataset)
            'min_impurity_decrease': 0.0,  # No early stopping
            'criterion': 'gini',        # Gini for multi-class
            'ccp_alpha': 0.0,           # No cost-complexity pruning
            'n_jobs': -1               # Use all cores
            # Note: oob_score is hardcoded to True in RandomForestClassifier initialization
        },
        'intrudtree': {
            'max_depth': 50,            # Very deep for complex patterns (target 80%+)
            'min_samples_split': 2,      # Fine-grained splits
            'min_samples_leaf': 1,       # Minimal leaf size
            'prune_threshold': 0.0005,  # Very light pruning
            'n_important_features': 60  # More features for better discrimination
        },
        'aca_svm': {
            'svm_type': 'kernel',       # Use kernel SVM (non-linear)
            'kernel': 'rbf',            # RBF kernel for non-linear patterns
            'C': 50.0,                  # Higher C for better fit (target 80%+)
            'gamma': 'scale',           # Scaled gamma for RBF
            'class_weight': None,       # No class weights (balanced)
            'max_iter': 10000,          # More iterations for convergence
            'tol': 1e-5,                # Tighter tolerance for better precision
            # Note: probability is handled internally in _create_svm_model for kernel SVMs
            'patterns_file': 'data/patterns/cyber_threat_patterns.txt'
        },
        'ensemble': {
            'ensemble_method': 'voting',  # Hard voting for balanced dataset
            'meta_learner': 'logistic_regression',
            'cv_folds': 5,              # More folds for better validation
            'weights': [1.0, 1.3, 1.0]   # Weight IntruDTree more (best performer)
        }
    }

def get_fast_mode_config(fast_mode=False):
    """Get model configuration based on fast mode setting"""
    if fast_mode:
        # Ultra-fast configuration for quick testing
        return {
            'fuzzy_rf': {
                'n_estimators': 50,       # Minimal trees for speed
                'max_depth': 10,          # Shallow trees
                'min_samples_split': 5,   # More restrictive
                'min_samples_leaf': 2,    # More restrictive
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'max_samples': 0.5,      # Use only 50% of samples
                'n_jobs': -1
            },
            'intrudtree': {
                'max_depth': 10,         # Shallow tree
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'prune_threshold': 0.01,
                'n_important_features': 15  # Fewer features
            },
            'aca_svm': {
                'svm_type': 'linear',
                'kernel': 'linear',
                'C': 1.0,
                'gamma': 'scale',
                'class_weight': 'balanced',
                'max_iter': 1000,        # Fewer iterations
                'tol': 1e-2,            # Relaxed tolerance
                'patterns_file': 'data/patterns/cyber_threat_patterns.txt'
            },
            'ensemble': {
                'ensemble_method': 'voting',
                'meta_learner': 'logistic_regression',
                'cv_folds': 2  # Minimal folds
            }
        }
    else:
        # Use optimized configuration for balanced dataset
        return get_optimized_config_for_balanced_dataset()

def main(args=None):
    """Main training function"""
    logger.info("=" * 80)
    logger.info("CYBER THREAT DETECTION SYSTEM - COMPLETE TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse args
    if args is None:
        parser = argparse.ArgumentParser(description='Train Cyber Threat Detection System')
        # Allow override via env var DATASET_PATH; fallback to balanced dataset
        default_data_path = os.environ.get('DATASET_PATH', 'data/consolidated/Network_Dataset_Balanced.csv')
        parser.add_argument('--data-path', default=default_data_path, help='Path to consolidated dataset CSV')
        parser.add_argument('--processed-dir', default='data/processed', help='Directory to store processed data')
        parser.add_argument('--models-dir', default='models', help='Directory to store models')
        parser.add_argument('--test-size', type=float, default=0.3, help='Test split size (0-1)')
        parser.add_argument('--scaler', choices=['minmax', 'standard'], default='minmax', help='Scaler type')
        parser.add_argument('--sample-size', type=int, default=0, help='Number of rows to sample (0 for full dataset - recommended for balanced dataset)')
        parser.add_argument('--random-state', type=int, default=42, help='Random seed')
        parser.add_argument('--fast-mode', action='store_true', help='Enable fast training mode with reduced parameters')
        args = parser.parse_args()

    # Resolve paths relative to this script directory
    script_dir = os.path.dirname(__file__)
    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(script_dir, p))

    # Re-resolve with env var precedence if set (even when args provided)
    env_data = os.environ.get('DATASET_PATH')
    requested_data_path = resolve_path(env_data if env_data else args.data_path)
    balanced_candidate = resolve_path('data/consolidated/Network_Dataset_Balanced.csv')
    sample_candidate = resolve_path('data/consolidated/All_Network_Sample_Complete.csv')

    chosen_data_path = requested_data_path
    if not os.path.exists(chosen_data_path):
        # First try balanced dataset, then fallback to sample
        if os.path.exists(balanced_candidate):
            logger.warning(f"Requested dataset not found: {requested_data_path}. Falling back to balanced dataset: {balanced_candidate}")
            chosen_data_path = balanced_candidate
        elif os.path.exists(sample_candidate):
            logger.warning(f"Requested dataset not found: {requested_data_path}. Falling back to sample: {sample_candidate}")
            chosen_data_path = sample_candidate
        else:
            logger.error(f"Dataset not found: {requested_data_path}. Also checked: {balanced_candidate}, {sample_candidate}")

    # Configuration
    config = {
        'data_path': chosen_data_path,
        'processed_data_path': resolve_path(args.processed_dir),
        'models_path': resolve_path(args.models_dir),
        'test_size': args.test_size,
        'random_state': args.random_state,
        'scaler_type': args.scaler,
        'sample_size': None if args.sample_size == 0 else args.sample_size,
        'fast_mode': getattr(args, 'fast_mode', False),
        'model_configs': get_fast_mode_config(getattr(args, 'fast_mode', False)),
        'use_feature_selection': True,  # Enable feature selection for better performance
        'n_features_to_select': 50      # Select top 50 most discriminative features
    }
    
    # Log configuration mode
    fast_mode = config.get('fast_mode', False)
    sample_size = config.get('sample_size')
    sample_size_str = 'FULL DATASET' if sample_size is None else str(sample_size)
    logger.info(f"Configuration: {'FAST MODE' if fast_mode else 'PRODUCTION MODE'}")
    logger.info(f"Sample size: {sample_size_str}")
    logger.info(f"Model configs: {list(config['model_configs'].keys())}")
    
    try:
        # Step 1: Initialize the orchestrator
        logger.info("\n" + "="*50)
        logger.info("STEP 1: INITIALIZING SYSTEM")
        logger.info("="*50)
        
        logger.info("Creating CyberThreatDetectionOrchestrator...")
        
        # Validate dataset path
        data_path = config['data_path']
        logger.info(f"Dataset path: {data_path}")
        if not os.path.exists(data_path):
            logger.error(f"Dataset file not found: {data_path}")
            return False
        
        logger.info(f"Dataset file exists: {os.path.getsize(data_path)} bytes")
        
        orchestrator = CyberThreatDetectionOrchestrator(config)
        logger.info("Orchestrator created successfully")
        init_result = orchestrator.initialize_system()
        
        if not init_result['success']:
            logger.error(f"System initialization failed: {init_result['error']}")
            return False
        
        logger.info("✓ System initialized successfully")
        logger.info(f"  - Models loaded: {init_result['models_loaded']}")
        
        # Step 2: Preprocess data
        logger.info("\n" + "="*50)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("="*50)
        
        preprocess_result = orchestrator.preprocess_data()
        
        if not preprocess_result['success']:
            logger.error(f"Data preprocessing failed: {preprocess_result['error']}")
            return False
        
        logger.info("✓ Data preprocessing completed")
        logger.info(f"  - Input file: {config['data_path']}")
        logger.info(f"  - Output directory: {config['processed_data_path']}")
        logger.info(f"  - Sample size: {config['sample_size'] if config['sample_size'] else 'FULL DATASET'}")
        logger.info(f"  - Features: {preprocess_result['preprocessing_info']['feature_count']}")
        
        # Step 3: Load processed data
        logger.info("\n" + "="*50)
        logger.info("STEP 3: LOADING PROCESSED DATA")
        logger.info("="*50)
        
        load_result = orchestrator.load_processed_data()
        
        if not load_result['success']:
            logger.error(f"Data loading failed: {load_result['error']}")
            return False
        
        logger.info("✓ Data loaded successfully")
        logger.info(f"  - Training samples: {load_result['training_samples']}")
        logger.info(f"  - Test samples: {load_result['test_samples']}")
        logger.info(f"  - Features: {load_result['features']}")
        
        # Step 4: Train all models
        logger.info("\n" + "="*50)
        logger.info("STEP 4: TRAINING MODELS")
        logger.info("="*50)
        
        train_result = orchestrator.train_all_models()
        
        if not train_result['success']:
            logger.error(f"Model training failed: {train_result['error']}")
            return False
        
        logger.info("✓ All models trained successfully")
        logger.info(f"  - Total training time: {train_result['total_training_time']:.2f} seconds")
        logger.info(f"  - Models trained: {train_result['models_trained']}")
        
        # Print individual model training results
        for model_name, result in train_result['training_results'].items():
            metrics = result['training_metrics']
            logger.info(f"  - {model_name}:")
            logger.info(f"    Accuracy: {metrics['accuracy']*100:.2f}%")
            logger.info(f"    Precision: {metrics['precision']*100:.2f}%")
            logger.info(f"    Recall: {metrics['recall']*100:.2f}%")
            logger.info(f"    F1-Score: {metrics['f1_score']*100:.2f}%")
        
        # Step 5: Evaluate all models
        logger.info("\n" + "="*50)
        logger.info("STEP 5: EVALUATING MODELS")
        logger.info("="*50)
        
        eval_result = orchestrator.evaluate_all_models()
        
        if not eval_result['success']:
            logger.error(f"Model evaluation failed: {eval_result['error']}")
            return False
        
        logger.info("✓ All models evaluated successfully")
        
        # Print evaluation results
        for model_name, metrics in eval_result['evaluation_results'].items():
            logger.info(f"  - {model_name}:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.2f}%")
            logger.info(f"    Precision: {metrics['precision']:.2f}%")
            logger.info(f"    Recall: {metrics['recall']:.2f}%")
            logger.info(f"    F1-Score: {metrics['f1_score']:.2f}%")
            logger.info(f"    Detection Time: {metrics['detection_time']:.3f} seconds")
            logger.info(f"    Threats Detected: {metrics['threats_detected']}")
            logger.info(f"    Actual Threats: {metrics['actual_threats']}")
        
        # Step 6: Perform threat analysis
        logger.info("\n" + "="*50)
        logger.info("STEP 6: PERFORMING THREAT ANALYSIS")
        logger.info("="*50)
        
        threat_result = orchestrator.perform_threat_analysis()
        
        if not threat_result['success']:
            logger.error(f"Threat analysis failed: {threat_result['error']}")
            return False
        
        logger.info("✓ Threat analysis completed")
        
        # Print threat analysis results
        for model_name, analysis in threat_result['threat_analysis_results'].items():
            risk_analysis = analysis['risk_analysis']
            logger.info(f"  - {model_name}:")
            logger.info(f"    Risk Score: {risk_analysis['risk_score']:.3f}")
            logger.info(f"    Risk Level: {risk_analysis['risk_level']}")
            logger.info(f"    Severity: {analysis['severity']}")
        
        # Step 7: Compare models
        logger.info("\n" + "="*50)
        logger.info("STEP 7: COMPARING MODELS")
        logger.info("="*50)
        
        compare_result = orchestrator.compare_models()
        
        if not compare_result['success']:
            logger.error(f"Model comparison failed: {compare_result['error']}")
            return False
        
        logger.info("✓ Model comparison completed")
        
        # Print comparison results
        comparison = compare_result['comparison_results']
        recommendations = comparison['recommendations']
        
        logger.info(f"  - Best Overall Model: {recommendations['best_overall_model']}")
        logger.info(f"  - Ensemble Recommended: {recommendations['ensemble_recommendation']}")
        
        # Print rankings
        rankings = comparison['performance_ranking']['overall']
        logger.info("  - Overall Rankings:")
        for model, rank in sorted(rankings.items(), key=lambda x: x[1]):
            logger.info(f"    {rank}. {model}")
        
        # Step 8: Save models
        logger.info("\n" + "="*50)
        logger.info("STEP 8: SAVING MODELS")
        logger.info("="*50)
        
        save_result = orchestrator.save_models()
        
        if not save_result['success']:
            logger.error(f"Model saving failed: {save_result['error']}")
            return False
        
        logger.info("✓ All models saved successfully")
        for model_name, path in save_result['saved_models'].items():
            logger.info(f"  - {model_name}: {path}")
        
        # Step 9: Generate and export report
        logger.info("\n" + "="*50)
        logger.info("STEP 9: GENERATING REPORTS")
        logger.info("="*50)
        
        # Export PDF report
        pdf_result = orchestrator.export_report('pdf')
        
        if pdf_result['success']:
            # Save PDF to file
            pdf_filename = f"reports/cyber_threat_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            os.makedirs('reports', exist_ok=True)
            
            with open(pdf_filename, 'wb') as f:
                f.write(pdf_result['data'])
            
            logger.info(f"✓ PDF report generated: {pdf_filename}")
        else:
            logger.error(f"PDF report generation failed: {pdf_result['error']}")
        
        # Export CSV report
        csv_result = orchestrator.export_report('csv')
        
        if csv_result['success']:
            # Save CSV to file
            csv_filename = f"reports/cyber_threat_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(csv_filename, 'w') as f:
                f.write(csv_result['data'])
            
            logger.info(f"✓ CSV report generated: {csv_filename}")
        else:
            logger.error(f"CSV report generation failed: {csv_result['error']}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Print final statistics
        total_threats = sum(metrics['actual_threats'] for metrics in eval_result['evaluation_results'].values())
        avg_accuracy = sum(metrics['accuracy'] for metrics in eval_result['evaluation_results'].values()) / len(eval_result['evaluation_results'])
        
        logger.info(f"Final Statistics:")
        logger.info(f"  - Total threats in dataset: {total_threats}")
        logger.info(f"  - Average accuracy across models: {avg_accuracy:.2f}%")
        logger.info(f"  - Best performing model: {recommendations['best_overall_model']}")
        logger.info(f"  - Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"\n❌ Training failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Check logs/training.log for full details")
        return False

if __name__ == "__main__":
    # Create required directories
    required_dirs = [
        'data/processed',
        'data/raw',
        'models',
        'logs',
        'reports'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Run training pipeline
    try:
        success = main()
        
        if success:
            logger.info("\n🎉 Training completed successfully! The system is ready for threat detection.")
            sys.exit(0)
        else:
            logger.error("\n❌ Training failed. Please check the logs for details.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)
