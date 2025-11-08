import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Badge, Alert, Table, Button } from 'react-bootstrap';

export default function ReliabilityComparisonPanel() {
  const [reliabilityData, setReliabilityData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchReliabilityData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchReliabilityData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchReliabilityData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/reliability/comparison');
      const data = await response.json();
      
      if (data.success) {
        setReliabilityData(data);
        setError(null);
      } else {
        setError(data.message || 'Failed to load reliability data');
      }
    } catch (err) {
      setError('Error connecting to server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getKappaColor = (kappa) => {
    if (kappa >= 0.81) return '#10b981'; // Green - Excellent
    if (kappa >= 0.61) return '#3b82f6'; // Blue - Good
    if (kappa >= 0.41) return '#f59e0b'; // Orange - Moderate
    if (kappa >= 0.21) return '#ef4444'; // Red - Fair
    return '#6b7280'; // Gray - Poor
  };

  const getKappaBadgeVariant = (interpretation) => {
    switch(interpretation?.toLowerCase()) {
      case 'excellent': return 'success';
      case 'good': return 'info';
      case 'moderate': return 'warning';
      case 'fair': return 'warning';
      case 'poor': return 'danger';
      default: return 'secondary';
    }
  };

  if (loading) {
    return (
      <div className="text-center p-5">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="mt-3">Loading reliability analysis...</p>
      </div>
    );
  }

  if (error || !reliabilityData || !reliabilityData.reliability_metrics || Object.keys(reliabilityData.reliability_metrics).length === 0) {
    return (
      <Alert variant="info" className="m-4">
        <Alert.Heading>
          <i className="fas fa-info-circle me-2"></i>
          No Reliability Data Available
        </Alert.Heading>
        <p>
          {error || reliabilityData?.message || 'Please verify some threats in the Real-time Detection tab first.'}
        </p>
        <p className="mb-0">
          <strong>How to generate reliability data:</strong>
          <ol>
            <li>Go to the "Real-time Detection" tab</li>
            <li>Review detected threats</li>
            <li>Click "✓ Confirm Threat" or "✗ False Positive" for each threat</li>
            <li>Return here to see the reliability analysis</li>
          </ol>
        </p>
      </Alert>
    );
  }

  const { reliability_metrics, cohens_kappa, summary } = reliabilityData;

  return (
    <div className="reliability-comparison-panel">
      {/* Header */}
      <div className="mb-4">
        <h2 className="mb-2">
          <i className="fas fa-check-circle me-2 text-success"></i>
          Reliability Comparison: Automated vs Manual Detection
        </h2>
        <p className="text-muted">
          Statement of Problem 3: How does automated, model-based detection compare to manual threat identification in terms of reliability?
        </p>
      </div>

      {/* Summary Conclusion */}
      <Alert variant="success" className="mb-4">
        <Alert.Heading>
          <i className="fas fa-lightbulb me-2"></i>
          Reliability Summary
        </Alert.Heading>
        <p className="mb-0">{summary.conclusion}</p>
      </Alert>

      {/* Key Metrics Cards */}
      <Row className="mb-4">
        <Col md={4}>
          <Card className="text-center h-100">
            <Card.Body>
              <Card.Title className="text-primary">
                <i className="fas fa-handshake me-2"></i>
                Agreement Rate
              </Card.Title>
              <h2 className="text-primary mb-0">
                {reliability_metrics.agreement_rate}%
              </h2>
              <small className="text-muted">
                {reliability_metrics.manual_true_positives} / {reliability_metrics.total_verified} confirmed
              </small>
              <p className="small text-muted mt-2 mb-0">
                Percentage of automated detections confirmed by manual verification
              </p>
            </Card.Body>
          </Card>
        </Col>

        <Col md={4}>
          <Card className="text-center h-100 border-success">
            <Card.Body>
              <Card.Title className="text-success">
                <i className="fas fa-chart-line me-2"></i>
                Cohen's Kappa (κ)
              </Card.Title>
              <h2 className="text-success mb-1" style={{ color: getKappaColor(cohens_kappa.kappa_display) }}>
                {cohens_kappa.kappa_display}
              </h2>
              <Badge bg={getKappaBadgeVariant(cohens_kappa.interpretation)}>
                {cohens_kappa.interpretation}
              </Badge>
              <p className="small text-muted mt-2 mb-0">
                {cohens_kappa.description}
              </p>
            </Card.Body>
          </Card>
        </Col>

        <Col md={4}>
          <Card className="text-center h-100">
            <Card.Body>
              <Card.Title className="text-warning">
                <i className="fas fa-exclamation-triangle me-2"></i>
                False Positive Rate
              </Card.Title>
              <h2 className="text-warning mb-0">
                {reliability_metrics.false_positive_rate}%
              </h2>
              <small className="text-muted">
                {reliability_metrics.manual_false_positives} false alarms
              </small>
              <p className="small text-muted mt-2 mb-0">
                Percentage of automated detections marked as false alarms
              </p>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Detailed Metrics Table */}
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <i className="fas fa-table me-2"></i>
            Detailed Reliability Metrics
          </h5>
        </Card.Header>
        <Card.Body>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Total Verified</strong></td>
                <td>{reliability_metrics.total_verified}</td>
                <td>Number of threats manually verified</td>
              </tr>
              <tr>
                <td><strong>Manual True Positives</strong></td>
                <td>{reliability_metrics.manual_true_positives}</td>
                <td>Threats confirmed as legitimate by manual verification</td>
              </tr>
              <tr>
                <td><strong>Manual False Positives</strong></td>
                <td>{reliability_metrics.manual_false_positives}</td>
                <td>Threats marked as false alarms by manual verification</td>
              </tr>
              <tr>
                <td><strong>Agreement Rate</strong></td>
                <td>{reliability_metrics.agreement_rate}%</td>
                <td>Percentage of automated detections confirmed by manual verification</td>
              </tr>
              <tr>
                <td><strong>False Positive Rate</strong></td>
                <td>{reliability_metrics.false_positive_rate}%</td>
                <td>Percentage of automated detections that are false alarms</td>
              </tr>
              <tr className="table-success">
                <td><strong>Cohen's Kappa (κ)</strong></td>
                <td><strong>{cohens_kappa.kappa_display}</strong></td>
                <td>Agreement measure accounting for chance (κ &gt; 0.8 = excellent)</td>
              </tr>
              <tr>
                <td><strong>Reliability Score</strong></td>
                <td>{reliability_metrics.reliability_score}%</td>
                <td>Composite metric: (Agreement Rate × 60%) + (Low False Positive Rate × 40%)</td>
              </tr>
            </tbody>
          </Table>
        </Card.Body>
      </Card>

      {/* Refresh Button */}
      <div className="text-center mb-4">
        <Button variant="primary" onClick={fetchReliabilityData}>
          <i className="fas fa-sync-alt me-2"></i>
          Refresh Reliability Data
        </Button>
      </div>
    </div>
  );
}

