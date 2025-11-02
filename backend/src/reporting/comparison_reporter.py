"""
Comparison & Reporting Module
Provides side-by-side model comparison, performance metrics, and exportable reports
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64

logger = logging.getLogger(__name__)

class ModelComparisonAnalyzer:
    """
    Analyzes and compares the performance of different models
    """
    
    def __init__(self):
        """Initialize the model comparison analyzer"""
        self.metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        self.comparison_results = {}
        
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple models side-by-side
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Comparison results
        """
        comparison = {
            'models': list(model_results.keys()),
            'metrics': {},
            'performance_ranking': {},
            'statistical_analysis': {},
            'recommendations': {}
        }
        
        # Compare each metric
        for metric in self.metrics:
            metric_values = {}
            for model_name, results in model_results.items():
                if metric in results:
                    metric_values[model_name] = results[metric]
            
            comparison['metrics'][metric] = metric_values
        
        # Calculate performance ranking
        comparison['performance_ranking'] = self._calculate_performance_ranking(model_results)
        
        # Statistical analysis
        comparison['statistical_analysis'] = self._perform_statistical_analysis(model_results)
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations(model_results)
        
        # Add timestamp
        comparison['comparison_timestamp'] = datetime.now().isoformat()
        
        self.comparison_results = comparison
        return comparison
    
    def _calculate_performance_ranking(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Calculate performance ranking for each model
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Performance ranking
        """
        rankings = {}
        
        for metric in self.metrics:
            metric_values = {}
            for model_name, results in model_results.items():
                if metric in results:
                    metric_values[model_name] = results[metric]
            
            # Sort by metric value (higher is better)
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            rankings[metric] = {
                model: rank + 1 for rank, (model, _) in enumerate(sorted_models)
            }
        
        # Overall ranking (average rank across all metrics)
        overall_ranks = {}
        for model_name in model_results.keys():
            model_ranks = []
            for metric in self.metrics:
                if metric in rankings and model_name in rankings[metric]:
                    model_ranks.append(rankings[metric][model_name])
            
            if model_ranks:
                overall_ranks[model_name] = sum(model_ranks) / len(model_ranks)
        
        # Sort by overall rank
        sorted_overall = sorted(overall_ranks.items(), key=lambda x: x[1])
        rankings['overall'] = {
            model: rank + 1 for rank, (model, _) in enumerate(sorted_overall)
        }
        
        return rankings
    
    def _perform_statistical_analysis(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Perform statistical analysis on model performance
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Statistical analysis results
        """
        analysis = {}
        
        for metric in self.metrics:
            values = []
            for model_name, results in model_results.items():
                if metric in results:
                    values.append(results[metric])
            
            if values:
                analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        # Model consistency analysis
        consistency_scores = {}
        for model_name in model_results.keys():
            model_metrics = []
            for metric in self.metrics:
                if metric in model_results[model_name]:
                    model_metrics.append(model_results[model_name][metric])
            
            if model_metrics:
                # Calculate coefficient of variation (lower is more consistent)
                mean_val = np.mean(model_metrics)
                std_val = np.std(model_metrics)
                if mean_val > 0:
                    consistency_scores[model_name] = std_val / mean_val
                else:
                    consistency_scores[model_name] = 0
        
        analysis['consistency_scores'] = consistency_scores
        
        return analysis
    
    def _generate_recommendations(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Generate recommendations based on model performance
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Recommendations
        """
        recommendations = {
            'best_overall_model': None,
            'best_by_metric': {},
            'model_strengths': {},
            'model_weaknesses': {},
            'ensemble_recommendation': False
        }
        
        # Find best overall model
        overall_ranks = self._calculate_performance_ranking(model_results)['overall']
        if overall_ranks:
            best_model = min(overall_ranks.items(), key=lambda x: x[1])[0]
            recommendations['best_overall_model'] = best_model
        
        # Find best model for each metric
        for metric in self.metrics:
            metric_values = {}
            for model_name, results in model_results.items():
                if metric in results:
                    metric_values[model_name] = results[metric]
            
            if metric_values:
                best_model = max(metric_values.items(), key=lambda x: x[1])[0]
                recommendations['best_by_metric'][metric] = best_model
        
        # Analyze model strengths and weaknesses
        for model_name, results in model_results.items():
            strengths = []
            weaknesses = []
            
            for metric in self.metrics:
                if metric in results:
                    # Compare with average performance
                    all_values = []
                    for other_model, other_results in model_results.items():
                        if metric in other_results:
                            all_values.append(other_results[metric])
                    
                    if all_values:
                        avg_performance = np.mean(all_values)
                        if results[metric] > avg_performance * 1.1:  # 10% better than average
                            strengths.append(f"High {metric}")
                        elif results[metric] < avg_performance * 0.9:  # 10% worse than average
                            weaknesses.append(f"Low {metric}")
            
            recommendations['model_strengths'][model_name] = strengths
            recommendations['model_weaknesses'][model_name] = weaknesses
        
        # Check if ensemble would be beneficial
        performance_variance = {}
        for metric in self.metrics:
            values = []
            for model_name, results in model_results.items():
                if metric in results:
                    values.append(results[metric])
            
            if values:
                performance_variance[metric] = np.var(values)
        
        # If high variance, ensemble might be beneficial
        avg_variance = np.mean(list(performance_variance.values()))
        recommendations['ensemble_recommendation'] = avg_variance > 0.01  # Threshold
        
        return recommendations

class PerformanceMetricsCalculator:
    """
    Calculates comprehensive performance metrics
    """
    
    def __init__(self):
        """Initialize the performance metrics calculator"""
        self.metrics = {}
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      model_name: str, detection_times: List[float] = None,
                                      label_encoder=None) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            detection_times: List of detection times
            label_encoder: Label encoder for class names
            
        Returns:
            Comprehensive metrics dictionary
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, confusion_matrix, classification_report,
                                   roc_auc_score, average_precision_score)
        
        # Basic metrics (multiclass-safe, used for per-class and overall)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # For threat counting metrics, binarize vs Benign
        def get_benign_label_index():
            if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                classes = list(label_encoder.classes_)
                if 'Benign' in classes:
                    return classes.index('Benign')
            # Fallback heuristic: assume the minority named 'Benign' not available; default to 0
            # This can be adjusted if metadata is available
            return 0
        benign_idx = get_benign_label_index()
        y_true_bin = (np.array(y_true) != benign_idx).astype(int)
        y_pred_bin = (np.array(y_pred) != benign_idx).astype(int)
        
        # Binary confusion matrix for threat vs benign
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        if cm_bin.size == 4:
            tn, fp, fn, tp = cm_bin.ravel()
        else:
            tn, fp, fn, tp = (0, 0, 0, 0)
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Detection rate
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Performance metrics
        metrics = {
            'model_name': model_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'specificity': round(specificity * 100, 2),
            'false_positive_rate': round(fpr * 100, 2),
            'false_negative_rate': round(fnr * 100, 2),
            'detection_rate': round(detection_rate * 100, 2),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true),
            'threats_detected': int(tp + fp),
            'actual_threats': int(tp + fn)
        }
        
        # Timing metrics
        if detection_times:
            metrics.update({
                'avg_detection_time': round(np.mean(detection_times), 3),
                'min_detection_time': round(np.min(detection_times), 3),
                'max_detection_time': round(np.max(detection_times), 3),
                'detection_time_std': round(np.std(detection_times), 3)
            })
        
        # Classification report with per-class metrics
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = class_report
            
            # Extract per-class metrics
            per_class_metrics = {}
            for class_name, class_metrics in class_report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    per_class_metrics[class_name] = {
                        'precision': round(class_metrics['precision'] * 100, 2),
                        'recall': round(class_metrics['recall'] * 100, 2),
                        'f1_score': round(class_metrics['f1-score'] * 100, 2),
                        'support': int(class_metrics['support'])
                    }
            
            metrics['per_class_metrics'] = per_class_metrics
            
            # Log per-class performance
            logger.info(f"Per-class metrics for {model_name}:")
            for class_name, class_metrics in per_class_metrics.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    logger.info(f"  {class_name}: Precision={class_metrics['precision']}%, "
                              f"Recall={class_metrics['recall']}%, F1={class_metrics['f1_score']}%")
            
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            metrics['classification_report'] = {}
            metrics['per_class_metrics'] = {}
        
        # Confusion matrix (multiclass) for reference
        cm_multi = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm_multi.tolist()
        
        # Calculate efficiency metrics
        metrics['efficiency'] = {
            'detection_efficiency': round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0,
            'precision_efficiency': round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0,
            'overall_efficiency': round((tp + tn) / len(y_true) * 100, 2)
        }
        
        # Calculate risk score based on actual threats detected
        if metrics['actual_threats'] > 0:
            risk_score = (metrics['threats_detected'] / metrics['actual_threats']) * 0.5 + \
                        (metrics['false_positives'] / len(y_true)) * 0.3 + \
                        (metrics['false_negatives'] / len(y_true)) * 0.2
        else:
            risk_score = 0.0
        
        metrics['risk_score'] = round(risk_score, 3)
        
        # Determine risk level based on risk score
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        metrics['risk_level'] = risk_level
        
        # Determine severity based on threat types detected
        if label_encoder:
            try:
                # Count threats by class
                threat_counts = {}
                for i, label in enumerate(y_pred):
                    if label != 0:  # Assuming 0 is benign
                        class_name = label_encoder.classes_[label] if label < len(label_encoder.classes_) else f"Class_{label}"
                        threat_counts[class_name] = threat_counts.get(class_name, 0) + 1
                
                # Determine severity based on threat types
                high_severity = ['Backdoor', 'Rootkit', 'Trojan', 'Virus', 'Worm']
                medium_severity = ['Exploit', 'HackTool']
                low_severity = ['Hoax', 'Benign']
                
                severity_score = 0
                for threat, count in threat_counts.items():
                    if threat in high_severity:
                        severity_score += count * 3
                    elif threat in medium_severity:
                        severity_score += count * 2
                    elif threat in low_severity:
                        severity_score += count * 1
                
                if severity_score > 100:
                    severity = "High"
                elif severity_score > 50:
                    severity = "Medium"
                else:
                    severity = "Low"
                
                metrics['severity'] = severity
                metrics['threat_distribution'] = threat_counts
                metrics['severity_score'] = severity_score
                
            except Exception as e:
                logger.warning(f"Could not determine severity: {e}")
                metrics['severity'] = "Unknown"
                metrics['threat_distribution'] = {}
                metrics['severity_score'] = 0
        else:
            metrics['severity'] = "Unknown"
            metrics['threat_distribution'] = {}
            metrics['severity_score'] = 0
        
        return metrics

class ReportGenerator:
    """
    Generates comprehensive reports and visualizations
    """
    
    def __init__(self):
        """Initialize the report generator"""
        self.report_data = {}
        
    def generate_comparison_report(self, comparison_results: Dict, 
                                 model_results: Dict[str, Dict]) -> Dict:
        """
        Generate comprehensive comparison report
        
        Args:
            comparison_results: Results from model comparison
            model_results: Individual model results
            
        Returns:
            Report data
        """
        # Persist model_results for downstream sections that read per-class/CM
        self.model_results = model_results

        report = {
            'executive_summary': self._generate_executive_summary(comparison_results),
            'detailed_analysis': self._generate_detailed_analysis(comparison_results),
            'visualizations': self._generate_visualizations(comparison_results, model_results),
            'recommendations': comparison_results.get('recommendations', {}),
            'technical_details': self._generate_technical_details(model_results),
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_models': len(model_results),
                'metrics_analyzed': len(comparison_results.get('metrics', {}))
            }
        }
        
        self.report_data = report
        return report
    
    def _generate_executive_summary(self, comparison_results: Dict) -> Dict:
        """Generate executive summary"""
        best_model = comparison_results.get('performance_ranking', {}).get('overall', {})
        best_model_name = min(best_model.items(), key=lambda x: x[1])[0] if best_model else "N/A"
        
        return {
            'best_performing_model': best_model_name,
            'key_findings': [
                f"Model {best_model_name} achieved the best overall performance",
                f"Average accuracy across all models: {self._calculate_avg_metric(comparison_results, 'accuracy')}%",
                f"Performance variance: {self._calculate_performance_variance(comparison_results)}%"
            ],
            'recommendations': [
                f"Deploy {best_model_name} for primary threat detection",
                "Consider ensemble approach for improved reliability",
                "Monitor model performance continuously"
            ]
        }
    
    def _generate_detailed_analysis(self, comparison_results: Dict) -> Dict:
        """
        Generate detailed analysis section
        
        Args:
            comparison_results: Results from model comparison
            
        Returns:
            Detailed analysis dictionary
        """
        analysis = {
            'performance_summary': {},
            'per_class_analysis': {},
            'confusion_matrices': {},
            'risk_assessment': {},
            'recommendations': comparison_results.get('recommendations', {}),
            # Add sections expected by exporters
            'performance_comparison': comparison_results.get('metrics', {}),
            'ranking_analysis': comparison_results.get('performance_ranking', {})
        }
        
        # Performance summary
        if 'metrics' in comparison_results:
            for metric, values in comparison_results['metrics'].items():
                if isinstance(values, dict):
                    best_model = max(values.items(), key=lambda x: x[1])[0]
                    worst_model = min(values.items(), key=lambda x: x[1])[0]
                    avg_value = sum(values.values()) / len(values)
                    
                    analysis['performance_summary'][metric] = {
                        'best_model': best_model,
                        'worst_model': worst_model,
                        'average': round(avg_value, 2),
                        'range': round(max(values.values()) - min(values.values()), 2)
                    }
        
        # Per-class analysis (if available)
        if hasattr(self, 'model_results'):
            for model_name, results in self.model_results.items():
                if 'per_class_metrics' in results:
                    analysis['per_class_analysis'][model_name] = results['per_class_metrics']
                
                if 'confusion_matrix' in results:
                    analysis['confusion_matrices'][model_name] = results['confusion_matrix']
                
                if 'risk_score' in results and 'severity' in results:
                    analysis['risk_assessment'][model_name] = {
                        'risk_score': results['risk_score'],
                        'risk_level': results['risk_level'],
                        'severity': results['severity'],
                        'threat_distribution': results.get('threat_distribution', {})
                    }
        
        return analysis
    
    def _generate_visualizations(self, comparison_results: Dict, 
                               model_results: Dict[str, Dict]) -> Dict:
        """Generate visualization data"""
        # Create performance comparison chart
        metrics_data = comparison_results.get('metrics', {})
        
        # Prepare data for visualization
        viz_data = {
            'performance_chart': self._create_performance_chart(metrics_data),
            'ranking_chart': self._create_ranking_chart(comparison_results.get('performance_ranking', {})),
            'statistical_chart': self._create_statistical_chart(comparison_results.get('statistical_analysis', {}))
        }
        
        return viz_data
    
    def _create_performance_chart(self, metrics_data: Dict) -> Dict:
        """Create performance comparison chart data"""
        chart_data = {
            'labels': list(metrics_data.get('accuracy', {}).keys()),
            'datasets': []
        }
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in metrics_data:
                dataset = {
                    'label': metric.replace('_', ' ').title(),
                    'data': list(metrics_data[metric].values()),
                    'backgroundColor': self._get_color_for_metric(metric)
                }
                chart_data['datasets'].append(dataset)
        
        return chart_data
    
    def _create_ranking_chart(self, ranking_data: Dict) -> Dict:
        """Create ranking chart data"""
        if 'overall' in ranking_data:
            chart_data = {
                'labels': list(ranking_data['overall'].keys()),
                'datasets': [{
                    'label': 'Overall Ranking',
                    'data': list(ranking_data['overall'].values()),
                    'backgroundColor': 'rgba(54, 162, 235, 0.8)'
                }]
            }
            return chart_data
        return {}
    
    def _create_statistical_chart(self, statistical_data: Dict) -> Dict:
        """Create statistical analysis chart data"""
        if 'consistency_scores' in statistical_data:
            chart_data = {
                'labels': list(statistical_data['consistency_scores'].keys()),
                'datasets': [{
                    'label': 'Consistency Score (Lower is Better)',
                    'data': list(statistical_data['consistency_scores'].values()),
                    'backgroundColor': 'rgba(255, 99, 132, 0.8)'
                }]
            }
            return chart_data
        return {}
    
    def _get_color_for_metric(self, metric: str) -> str:
        """Get color for metric"""
        colors = {
            'accuracy': 'rgba(75, 192, 192, 0.8)',
            'precision': 'rgba(255, 159, 64, 0.8)',
            'recall': 'rgba(153, 102, 255, 0.8)',
            'f1_score': 'rgba(255, 205, 86, 0.8)'
        }
        return colors.get(metric, 'rgba(201, 203, 207, 0.8)')
    
    def _calculate_avg_metric(self, comparison_results: Dict, metric: str) -> float:
        """Calculate average for a specific metric"""
        metric_data = comparison_results.get('metrics', {}).get(metric, {})
        if metric_data:
            return round(np.mean(list(metric_data.values())), 2)
        return 0.0
    
    def _calculate_performance_variance(self, comparison_results: Dict) -> float:
        """Calculate performance variance"""
        all_values = []
        for metric_data in comparison_results.get('metrics', {}).values():
            all_values.extend(metric_data.values())
        
        if all_values:
            return round(np.var(all_values), 2)
        return 0.0
    
    def _generate_technical_details(self, model_results: Dict[str, Dict]) -> Dict:
        """Generate technical details"""
        technical_details = {}
        
        for model_name, results in model_results.items():
            technical_details[model_name] = {
                'model_type': results.get('model_type', 'Unknown'),
                'feature_count': results.get('feature_count', 0),
                'training_samples': results.get('total_samples', 0),
                'detection_statistics': {
                    'threats_detected': results.get('threats_detected', 0),
                    'actual_threats': results.get('actual_threats', 0),
                    'false_positives': results.get('false_positives', 0),
                    'false_negatives': results.get('false_negatives', 0)
                }
            }
        
        return technical_details
    
    def export_to_pdf(self, filename: str = None) -> bytes:
        """
        Export report to PDF format
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            PDF data as bytes
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        # Create PDF document
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Build PDF content
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Cyber Threat Detection Model Comparison Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        exec_summary = self.report_data['executive_summary']
        story.append(Paragraph(f"<b>Best Performing Model:</b> {exec_summary['best_performing_model']}", styles['Normal']))
        story.append(Spacer(1, 6))
        
        for finding in exec_summary['key_findings']:
            story.append(Paragraph(f"• {finding}", styles['Normal']))
            story.append(Spacer(1, 3))
        
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for rec in exec_summary['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 3))
        
        story.append(Spacer(1, 20))
        
        # Performance Comparison Table
        story.append(Paragraph("Performance Comparison", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        metrics_data = self.report_data['detailed_analysis']['performance_comparison']
        if metrics_data:
            # Create table data
            table_data = [['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']]
            
            models = list(metrics_data.get('accuracy', {}).keys())
            for model in models:
                row = [model]
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value = metrics_data.get(metric, {}).get(model, 0)
                    row.append(f"{value:.2f}")
                table_data.append(row)
            
            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export report data to CSV format
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            CSV data as string
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        # Create CSV data
        csv_data = []
        
        # Performance comparison
        metrics_data = self.report_data['detailed_analysis']['performance_comparison']
        if metrics_data:
            csv_data.append("Performance Comparison")
            csv_data.append("Model,Accuracy (%),Precision (%),Recall (%),F1-Score (%)")
            
            models = list(metrics_data.get('accuracy', {}).keys())
            for model in models:
                row = [model]
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value = metrics_data.get(metric, {}).get(model, 0)
                    row.append(f"{value:.2f}")
                csv_data.append(",".join(row))
            
            csv_data.append("")  # Empty line
        
        # Ranking analysis
        ranking_data = self.report_data['detailed_analysis']['ranking_analysis']
        if ranking_data and 'overall' in ranking_data:
            csv_data.append("Overall Ranking")
            csv_data.append("Model,Rank")
            for model, rank in ranking_data['overall'].items():
                csv_data.append(f"{model},{rank}")
            csv_data.append("")
        
        # Technical details
        technical_data = self.report_data['technical_details']
        if technical_data:
            csv_data.append("Technical Details")
            csv_data.append("Model,Model Type,Feature Count,Training Samples,Threats Detected,Actual Threats")
            for model, details in technical_data.items():
                row = [
                    model,
                    details.get('model_type', 'Unknown'),
                    str(details.get('feature_count', 0)),
                    str(details.get('training_samples', 0)),
                    str(details['detection_statistics'].get('threats_detected', 0)),
                    str(details['detection_statistics'].get('actual_threats', 0))
                ]
                csv_data.append(",".join(row))
        
        return "\n".join(csv_data)
