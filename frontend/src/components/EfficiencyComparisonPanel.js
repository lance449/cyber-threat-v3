import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Badge, Alert, Table, Button } from 'react-bootstrap';

export default function EfficiencyComparisonPanel() {
  const [efficiencyData, setEfficiencyData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchEfficiencyData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchEfficiencyData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchEfficiencyData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/efficiency/comparison');
      const data = await response.json();
      
      if (data.success) {
        setEfficiencyData(data);
        setError(null);
      } else {
        setError(data.message || 'Failed to load efficiency data');
      }
    } catch (err) {
      setError('Error connecting to server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getEfficiencyColor = (ratio) => {
    if (ratio >= 100) return '#10b981'; // Green - Extremely High
    if (ratio >= 50) return '#3b82f6'; // Blue - Very High
    if (ratio >= 10) return '#f59e0b'; // Orange - High
    return '#ef4444'; // Red - Moderate
  };

  const getEfficiencyBadgeVariant = (level) => {
    switch(level?.toLowerCase()) {
      case 'extremely high': return 'success';
      case 'very high': return 'info';
      case 'high': return 'warning';
      case 'moderate': return 'secondary';
      default: return 'secondary';
    }
  };

  if (loading) {
    return (
      <div className="text-center p-5">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="mt-3">Loading efficiency analysis...</p>
      </div>
    );
  }

  if (error || !efficiencyData || !efficiencyData.efficiency_metrics || Object.keys(efficiencyData.efficiency_metrics).length === 0) {
    return (
      <Alert variant="info" className="m-4">
        <Alert.Heading>
          <i className="fas fa-info-circle me-2"></i>
          No Efficiency Data Available
        </Alert.Heading>
        <p>
          {error || efficiencyData?.message || 'Please verify some threats in the Real-time Detection tab first.'}
        </p>
        <p className="mb-0">
          <strong>How to generate efficiency data:</strong>
          <ol>
            <li>Go to the "Real-time Detection" tab</li>
            <li>Click "► Show Features" to expand a threat row (this starts the timer)</li>
            <li>Review the threat details</li>
            <li>Click "✓ Confirm Threat" or "✗ False Positive" to verify (this stops the timer)</li>
            <li>Repeat for multiple threats</li>
            <li>Return here to see the efficiency analysis</li>
          </ol>
        </p>
      </Alert>
    );
  }

  const { efficiency_metrics, summary } = efficiencyData;
  const { automated_detection, manual_verification, efficiency_comparison } = efficiency_metrics;

  return (
    <div className="efficiency-comparison-panel">
      {/* Header */}
      <div className="mb-4">
        <h2 className="mb-2">
          <i className="fas fa-tachometer-alt me-2 text-primary"></i>
          Efficiency Comparison: Automated vs Manual Detection
        </h2>
        <p className="text-muted">
          Statement of Problem 3: How does automated, model-based detection compare to manual threat identification in terms of efficiency?
        </p>
      </div>

      {/* Summary Conclusion */}
      <Alert variant="success" className="mb-4">
        <Alert.Heading>
          <i className="fas fa-lightbulb me-2"></i>
          Efficiency Summary
        </Alert.Heading>
        <p className="mb-0">{summary.conclusion}</p>
      </Alert>

      {/* Detailed Metrics Table */}
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <i className="fas fa-table me-2"></i>
            Detailed Efficiency Metrics
          </h5>
        </Card.Header>
        <Card.Body>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Automated</th>
                <th>Manual</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Time per Threat</strong></td>
                <td>{automated_detection.time_per_threat_seconds}s</td>
                <td>{manual_verification.median_time_seconds}s (median)</td>
                <td>Average time to process one threat</td>
              </tr>
              <tr>
                <td><strong>Throughput</strong></td>
                <td>{automated_detection.throughput_per_minute} threats/min</td>
                <td>{manual_verification.throughput_per_minute} threats/min</td>
                <td>Number of threats processed per minute</td>
              </tr>
              <tr>
                <td><strong>Time Saved per Threat</strong></td>
                <td colSpan="2" className="text-success">
                  <strong>{efficiency_comparison.time_saved_per_threat_seconds}s</strong> ({efficiency_comparison.time_saved_percentage}% faster)
                </td>
                <td>Time saved using automated detection per threat</td>
              </tr>
              <tr className="table-success">
                <td><strong>Efficiency Ratio</strong></td>
                <td colSpan="2" className="text-center">
                  <strong style={{ fontSize: '1.2em', color: getEfficiencyColor(efficiency_comparison.efficiency_ratio) }}>
                    {efficiency_comparison.efficiency_ratio}x faster
                  </strong>
                </td>
                <td>How many times faster automated detection is</td>
              </tr>
              <tr>
                <td><strong>Manual Analysis Stats</strong></td>
                <td colSpan="2">
                  Min: {manual_verification.min_time_seconds}s | 
                  Max: {manual_verification.max_time_seconds}s | 
                  Mean: {manual_verification.mean_time_seconds}s | 
                  Std: {manual_verification.std_time_seconds}s
                </td>
                <td>Statistical distribution of manual analysis times</td>
              </tr>
            </tbody>
          </Table>
        </Card.Body>
      </Card>

      {/* Sample Data Table - Show Recent Detection Times */}
      {efficiency_metrics.calculation_breakdown?.sample_data && (
        <Card className="mb-4">
          <Card.Header>
            <h5 className="mb-0">
              <i className="fas fa-list me-2"></i>
              Sample Detection Times (Recent Data)
            </h5>
          </Card.Header>
          <Card.Body>
            <Row>
              <Col md={6}>
                <h6 className="text-primary mb-3">
                  <i className="fas fa-robot me-2"></i>
                  Automated Detection Times (Last 20 threats)
                </h6>
                <div className="p-3 bg-light rounded" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {efficiency_metrics.calculation_breakdown.sample_data.automated_times.length > 0 ? (
                    <div>
                      <div className="d-flex flex-wrap gap-2 mb-3">
                        {efficiency_metrics.calculation_breakdown.sample_data.automated_times.map((time, idx) => (
                          <Badge key={idx} bg="primary" className="p-2">
                            {time}s
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-3 p-2 bg-white rounded border">
                        <strong>Average:</strong> {automated_detection.time_per_threat_seconds}s
                        <br />
                        <small className="text-muted">
                          Calculated from {automated_detection.total_threats_processed || 0} detected non-benign threats (excludes benign traffic)
                        </small>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted">No detection time data available yet</p>
                  )}
                </div>
              </Col>
              <Col md={6}>
                <h6 className="text-warning mb-3">
                  <i className="fas fa-user-check me-2"></i>
                  Manual Analysis Times (Last 20 verifications)
                </h6>
                <div className="p-3 bg-light rounded" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {efficiency_metrics.calculation_breakdown.sample_data.manual_times.length > 0 ? (
                    <div>
                      <div className="d-flex flex-wrap gap-2 mb-3">
                        {efficiency_metrics.calculation_breakdown.sample_data.manual_times.map((time, idx) => (
                          <Badge key={idx} bg="warning" className="p-2">
                            {time}s
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-3 p-2 bg-white rounded border">
                        <strong>Median:</strong> {manual_verification.median_time_seconds}s
                        <br />
                        <strong>Mean:</strong> {manual_verification.mean_time_seconds}s
                        <br />
                        <small className="text-muted">
                          Based on {efficiency_metrics.samples_with_analysis_time || 0} verified threats
                        </small>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted">No analysis time data available yet</p>
                  )}
                </div>
              </Col>
            </Row>
          </Card.Body>
        </Card>
      )}

      {/* Refresh Button */}
      <div className="text-center mb-4">
        <Button variant="primary" onClick={fetchEfficiencyData}>
          <i className="fas fa-sync-alt me-2"></i>
          Refresh Efficiency Data
        </Button>
      </div>
    </div>
  );
}

