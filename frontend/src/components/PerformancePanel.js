import React from 'react';
import { Card, Row, Col, Badge, Button, ButtonGroup } from 'react-bootstrap';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';
// Import jspdf-autotable - it extends jsPDF prototype when imported as side-effect
import 'jspdf-autotable';
// Also import as default for function call pattern
import autoTable from 'jspdf-autotable';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ComposedChart, Bar, Line
} from 'recharts';
import '../styles/performance-panel.css';

// Helper function to calculate confidence interval
const calculateConfidenceInterval = (value, sampleSize) => {
  if (!sampleSize) return 0;
  const z = 1.96; // 95% confidence level
  const standardError = Math.sqrt((value * (1 - value)) / sampleSize);
  return (z * standardError * 100).toFixed(1);
};

const prepareExportData = (models, metrics) => {
  return models.map(model => ({
    'Model Name': model.name,
    'Detection Accuracy (%)': metrics[model.name].accuracy,
    'Total Detections': metrics[model.name].totalDetections,
    'True Positives': metrics[model.name].truePositiveCount,
    'False Positives': metrics[model.name].falsePositiveCount,
    'Coverage (%)': metrics[model.name].detectionCoverage,
    'Reliability Score': metrics[model.name].automationReliability,
    'Efficiency Ratio': metrics[model.name].efficiencyRatio
  }));
};

const downloadCSV = (data) => {
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(header => row[header]).join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
  saveAs(blob, `model-performance-${new Date().toISOString().slice(0,10)}.csv`);
};

const generatePDFReport = (data) => {
  const doc = new jsPDF();
  
  doc.setFontSize(16);
  doc.text('Model Performance Report', 14, 15);
  doc.setFontSize(10);
  doc.text(`Generated on ${new Date().toLocaleString()}`, 14, 25);
  
  // Use autoTable - try multiple patterns for compatibility
  try {
    // Pattern 1: If autoTable extends jsPDF prototype (v3.x + v5.x)
    if (typeof doc.autoTable === 'function') {
      doc.autoTable({
        head: [Object.keys(data[0])],
        body: data.map(row => Object.values(row)),
        startY: 35,
        theme: 'grid'
      });
    }
    // Pattern 2: If autoTable is a function (v5.x default export)
    else if (typeof autoTable === 'function') {
      autoTable(doc, {
        head: [Object.keys(data[0])],
        body: data.map(row => Object.values(row)),
        startY: 35,
        theme: 'grid'
      });
    }
    // Pattern 3: Fallback - use default export with function call
    else if (autoTable && typeof autoTable.default === 'function') {
      autoTable.default(doc, {
        head: [Object.keys(data[0])],
        body: data.map(row => Object.values(row)),
        startY: 35,
        theme: 'grid'
      });
    }
    else {
      throw new Error('autoTable function not available');
    }
  } catch (error) {
    console.error('Error generating PDF table:', error);
    doc.text('Error: PDF table generation failed', 14, 40);
    doc.text('Please check console for details', 14, 50);
  }
  
  doc.save(`model-performance-${new Date().toISOString().slice(0,10)}.pdf`);
};

export default function PerformancePanel({ rows = [], verifications = {}, performanceStats = {} }) {
  // Filter to only include threats detected by at least one model
  const detectedByModels = rows.filter(r => {
    // Exclude "Unknown" threats
    if (r.attack_type === 'Unknown' || r.attack_type === 'unknown') {
      return false;
    }
    
    // Only include threats detected by at least one model
    const modelUsed = String(r.model_used || '').trim().toLowerCase();
    if (!modelUsed || modelUsed === 'unknown' || modelUsed === 'simulated') {
      return false; // Skip this threat as no model detected it
    }
    
    return true;
  });
  
  // Calculate accuracy percentage for display - NOW INCLUDES OVERALL ACCURACY
  const calculateModelMetrics = (modelName) => {
    // Map display name to backend model name
    let backendModelName = '';
    switch(modelName) {
      case 'Fuzzy Logic + Random Forest':
        backendModelName = 'Fuzzy_RF';
        break;
      case 'IntruDTree Algorithm':
        backendModelName = 'IntruDTree';
        break;
      case 'Aho–Corasick + SVM':
        backendModelName = 'ACA_SVM';
        break;
    }
    
    // Calculate detection metrics from model_predictions
    let totalDetections = 0;  // All flows where model predicted threat
    let manualTruePositives = 0;  // Manually verified as true threats
    let manualFalsePositives = 0;  // Manually verified as false positives
    let totalFlowsAnalyzed = 0;  // All flows this model analyzed
    let truePositives = 0;
    let falsePositives = 0;
    let trueNegatives = 0;
    let falseNegatives = 0;
    let correctPredictions = 0;
    
    // Calculate from model_predictions field - only use threats detected by models
    detectedByModels.forEach(r => {
      if (r.model_predictions && r.model_predictions[backendModelName]) {
        const pred = r.model_predictions[backendModelName];
        
        // Count all flows analyzed by this model
        totalFlowsAnalyzed++;
        
        // Get the actual attack type from the row
        const actualAttackLabel = r.attack_type || r.attack_label || 'Benign';
        const predictedAttackLabel = pred.attack_label || 'Unknown';
        
        // Count detections ONLY when model's predicted attack type matches the actual attack type
        // Example: If actual is "DoS" and model predicts "DoS" → counts as detection
        // Example: If actual is "DoS" but model predicts "DDoS" → does NOT count as detection
        const isTypeMatch = predictedAttackLabel === actualAttackLabel;
        
        if (pred.predicted_threat && isTypeMatch) {
          totalDetections++;
          
          // Check manual verification status
          // If a threat was manually verified as true, count it as true positive for this model
          // If manually verified as false (false alarm), count it as false positive for this model
          if (verifications.hasOwnProperty(r.flow_id)) {
            if (verifications[r.flow_id] === true) {
              manualTruePositives++;
            } else if (verifications[r.flow_id] === false) {
              manualFalsePositives++;
            }
          }
        }
        
        // Also track correct predictions for overall accuracy
        if (pred.is_correct) {
          correctPredictions++;
        }
        
        // Calculate confusion matrix metrics based on threat detection
        if (pred.predicted_threat && r.true_is_threat) {
          truePositives++;
        } else if (pred.predicted_threat && !r.true_is_threat) {
          falsePositives++;
        } else if (!pred.predicted_threat && !r.true_is_threat) {
          trueNegatives++;
        } else if (!pred.predicted_threat && r.true_is_threat) {
          falseNegatives++;
        }
      }
    });
    
    // Detection Accuracy = (Manually Verified True Positives / Total Detections) × 100
    // ONLY calculated AFTER manual verification
    // Only counts detections where predicted attack type matches actual attack type
    // Example: Model detected 10 threats (correct types), manually verified 3 as true threats → 30% accuracy
    const detectionAccuracy = totalDetections > 0 ? (manualTruePositives / totalDetections) : 0;
    
    // Overall Accuracy = (TP + TN) / Total (shows overall correctness)
    const overallAccuracy = totalDetections > 0 ? (correctPredictions / totalDetections) : 0;
    
    // Fallback to manual verification if no model_predictions data
    let fallbackAccuracy = 0;
    let verifiedDetections = [];
    if (totalDetections === 0) {
      const modelRows = detectedByModels.filter(r => {
        const models = r.model_used?.split(',').map(m => m.trim()) || [];
        return models.some(m => {
          switch(m.toLowerCase()) {
            case 'fuzzy_rf': return modelName === 'Fuzzy Logic + Random Forest';
            case 'intrudtree': return modelName === 'IntruDTree Algorithm';
            case 'aca_svm': return modelName === 'Aho–Corasick + SVM';
            default: return false;
          }
        });
      });

      verifiedDetections = modelRows.filter(r => verifications.hasOwnProperty(r.flow_id));
      const verifiedTruePositives = verifiedDetections.filter(r => verifications[r.flow_id] === true);
      fallbackAccuracy = verifiedDetections.length > 0 
        ? (verifiedTruePositives.length / verifiedDetections.length) 
        : 0;
    }
    
    const marginOfError = totalDetections > 0
      ? 1.96 * Math.sqrt((overallAccuracy * (1-overallAccuracy)) / totalDetections)
      : 0;

    return {
      totalDetections: totalDetections || verifiedDetections.length,
      totalAnalyzed: totalFlowsAnalyzed || totalDetections,
      overallAccuracy: (overallAccuracy * 100).toFixed(1),
      detectionAccuracy: (detectionAccuracy * 100).toFixed(1),
      verifiedCount: verifiedDetections.length,
      truePositiveCount: manualTruePositives,
      falsePositiveCount: manualFalsePositives,
      trueNegativeCount: trueNegatives,
      falseNegativeCount: falseNegatives,
      accuracy: (detectionAccuracy * 100).toFixed(1),  // Use detection accuracy for display
      confidenceInterval: `±${(marginOfError * 100).toFixed(1)}%`,
      falsePositiveRate: totalDetections > 0
        ? ((manualFalsePositives / totalDetections) * 100).toFixed(1)
        : 0,
      truePositiveRate: (truePositives + falseNegatives) > 0
        ? ((truePositives / (truePositives + falseNegatives)) * 100).toFixed(1)
        : 0,
      precisionCI: calculateConfidenceInterval(overallAccuracy, totalDetections),
      coverageCI: calculateConfidenceInterval(overallAccuracy, totalDetections)
    };
  };

  const models = [
    {
      name: 'Fuzzy Logic + Random Forest',
      color: '#3b82f6',
      icon: 'fa-tree'
    },
    {
      name: 'IntruDTree Algorithm',
      color: '#10b981',
      icon: 'fa-code-branch'
    },
    {
      name: 'Aho–Corasick + SVM',
      color: '#8b5cf6',
      icon: 'fa-network-wired'
    }
  ];

  const exportMetrics = (format) => {
    const modelMetrics = {};
    models.forEach(model => {
      modelMetrics[model.name] = calculateModelMetrics(model.name);
    });
    
    const data = prepareExportData(models, modelMetrics);
    if (format === 'csv') {
      downloadCSV(data);
    } else if (format === 'pdf') {
      generatePDFReport(data);
    }
  };

  // Prepare data for charts
  const modelPerformanceData = models.map(model => {
    const metrics = calculateModelMetrics(model.name);
    return {
      name: model.name.replace(' + ', '\n'),
      accuracy: parseFloat(metrics.accuracy),
      detections: metrics.totalDetections,
      truePositives: metrics.truePositiveCount,
      falsePositives: metrics.falsePositiveCount,
      color: model.color
    };
  });

  return (
    <div className="performance-panel mb-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="panel-title mb-0">
          <i className="fas fa-chart-line me-2"></i>
          Model Performance Comparison
        </h5>
        <ButtonGroup>
          <Button 
            variant="outline-secondary" 
            size="sm"
            onClick={() => exportMetrics('csv')}
          >
            <i className="fas fa-file-csv me-2"></i>
            Export CSV
          </Button>
          <Button 
            variant="outline-secondary" 
            size="sm"
            onClick={() => exportMetrics('pdf')}
          >
            <i className="fas fa-file-pdf me-2"></i>
            Export PDF
          </Button>
        </ButtonGroup>
      </div>

      {/* Dual-axis Combo Chart */}
      <Row className="mb-4">
        <Col md={12}>
          <Card>
            <Card.Body>
              <Card.Title className="h6">Model Performance Overview</Card.Title>
              <div style={{ height: '400px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={modelPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis yAxisId="left" label={{ value: 'Counts', angle: -90, position: 'insideLeft' }} />
                    <YAxis yAxisId="right" orientation="right" domain={[0, 100]} label={{ value: 'Accuracy %', angle: 90, position: 'insideRight' }} />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="detections" name="Total Detections" fill="#3b82f6" />
                    <Bar yAxisId="left" dataKey="truePositives" name="True Positives" fill="#10b981" />
                    <Bar yAxisId="left" dataKey="falsePositives" name="False Positives" fill="#ef4444" />
                    <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy %" stroke="#8884d8" strokeWidth={3} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Model Performance Cards */}
      <Row className="mb-4">
        {models.map(model => {
          const metrics = calculateModelMetrics(model.name);
          return (
            <Col md={4} key={model.name}>
              <Card className="model-card h-100">
                <Card.Body>
                  <Card.Title className="model-name" style={{ color: model.color }}>
                    <i className={`fas ${model.icon} me-2`}></i>
                    {model.name}
                  </Card.Title>
                  <div className="metrics-container">
                    <div className="metric">
                      <span className="metric-label">Detection Accuracy</span>
                      <Badge bg={metrics.accuracy > 70 ? 'success' : 'warning'}>
                        {metrics.accuracy}%
                      </Badge>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Total Detections</span>
                      <Badge bg="info">{metrics.totalDetections}</Badge>
                    </div>
                    <div className="metric">
                      <span className="metric-label">True Positives</span>
                      <Badge bg="success">{metrics.truePositiveCount}</Badge>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Status</span>
                      <Badge bg={metrics.totalDetections > 0 ? 'success' : 'secondary'}>
                        {metrics.totalDetections > 0 ? 'Active' : 'Waiting'}
                      </Badge>
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          );
        })}
      </Row>
    </div>
  );
}
