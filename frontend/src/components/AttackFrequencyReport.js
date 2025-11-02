import React, { useState, useEffect } from 'react';
import { Table, Card, Badge, Button, Row, Col, Alert, ProgressBar } from 'react-bootstrap';
import { getAttackFrequencyReport, clearReportData, getReportStatus } from '../api';
import 'bootstrap/dist/css/bootstrap.min.css';

const styles = `
  .report-card {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-radius: 12px;
    border: none;
    margin-bottom: 20px;
  }
  
  .report-header {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border-radius: 12px 12px 0 0;
    padding: 20px;
  }
  
  .attack-type-row {
    transition: background-color 0.2s ease;
  }
  
  .attack-type-row:hover {
    background-color: #f8f9fa;
  }
  
  .frequency-bar {
    height: 20px;
    border-radius: 10px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    position: relative;
    overflow: hidden;
  }
  
  .frequency-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    font-size: 0.8rem;
  }
  
  .stats-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .stats-number {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 5px;
  }
  
  .stats-label {
    color: #6c757d;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .chart-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  
  .export-buttons {
    margin-top: 20px;
  }
  
  .attack-badge {
    font-size: 0.8rem;
    padding: 0.4rem 0.8rem;
  }
  
  .bar-chart-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #dee2e6;
  }
  
  .bar-chart-item {
    margin-bottom: 15px;
  }
  
  .bar-chart-bar-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    height: 30px;
    position: relative;
    overflow: hidden;
  }
  
  .bar-chart-bar {
    height: 100%;
    border-radius: 10px;
    position: relative;
    transition: width 0.3s ease;
    min-width: 20px;
  }
  
  .bar-chart-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    font-size: 0.8rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
  }
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default function AttackFrequencyReport() {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch data from backend API
  const fetchReportData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getAttackFrequencyReport();
      
      if (response.success) {
        console.log('Attack Frequency Report Data:', response.data);  // Debug log
        if (!response.data.attackFrequencyData) {
          console.warn('No attack frequency data in response');  // Debug warning
        }
        setReportData(response.data);
      } else {
        console.error('Attack Frequency Report Error:', response.error);
        setError(response.error || 'Failed to fetch report data');
      }
    } catch (err) {
      console.error('Attack Frequency Report Exception:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Clear all cached data
  const clearData = async () => {
    setLoading(true);
    try {
      const response = await clearReportData();
      if (response.success) {
        console.log('Report data cleared successfully');
        setReportData(null);
        setError(null);
      } else {
        console.error('Failed to clear data:', response.error);
        setError(response.error || 'Failed to clear report data');
      }
    } catch (err) {
      console.error('Clear data exception:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load data on component mount and set up auto-refresh
  useEffect(() => {
    fetchReportData();
    
    // Auto-refresh every 30 seconds to get latest data
    const interval = setInterval(fetchReportData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const exportToPDF = () => {
    const printWindow = window.open('', '_blank');
    const reportContent = document.getElementById('attack-frequency-report');
    
    printWindow.document.write(`
      <html>
        <head>
          <title>Attack Type Frequency Report</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .report-header { background: #f093fb; color: white; padding: 20px; text-align: center; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .frequency-bar { height: 20px; background: #667eea; border-radius: 10px; }
          </style>
        </head>
        <body>
          ${reportContent.innerHTML}
        </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
  };

  const exportToExcel = () => {
    if (!reportData) return;

    let csvContent = "Attack Type,Count,Percentage,High Severity,Medium Severity,Low Severity\n";
    
    reportData.attackFrequencyData.forEach(attack => {
      csvContent += `${attack.attackType},${attack.count},${attack.percentage}%,${attack.severityBreakdown.High},${attack.severityBreakdown.Medium},${attack.severityBreakdown.Low}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attack_frequency_report_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getAttackTypeColor = (attackType) => {
    const colors = {
      'DoS': '#dc3545',
      'DDoS': '#dc3545',
      'Trojan': '#fd7e14',
      'Virus': '#fd7e14',
      'Worm': '#fd7e14',
      'Backdoor': '#fd7e14',
      'Rootkit': '#6f42c1',
      'Exploit': '#6f42c1',
      'Port Scan': '#ffc107',
      'Brute Force': '#ffc107',
      'Web Attack': '#20c997',
      'Botnet': '#20c997',
      'HackTool': '#6c757d',
      'Hoax': '#6c757d',
      'Infiltration': '#e83e8c'
    };
    return colors[attackType] || '#6c757d';
  };

  if (loading) {
    return (
      <Card className="report-card">
        <Card.Body className="text-center">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-2">Generating Attack Frequency Report...</p>
        </Card.Body>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Error Generating Report</Alert.Heading>
        <p>{error}</p>
      </Alert>
    );
  }

  if (!reportData) {
    return (
      <Alert variant="info">
        <Alert.Heading>No Data Available</Alert.Heading>
        <p>Please run threat detection to generate attack frequency data.</p>
        <Button variant="primary" onClick={fetchReportData} disabled={loading}>
          {loading ? '🔄 Loading...' : '🔄 Refresh Data'}
        </Button>
      </Alert>
    );
  }

  // Check if we have actual attack data and proper structure
  const hasAttackData = reportData.summary && reportData.summary.total_attacks > 0;
  const hasAttackFrequencyData = reportData.attackFrequencyData && Array.isArray(reportData.attackFrequencyData);
  const hasSeverityDistribution = reportData.summary && reportData.summary.severity_distribution && typeof reportData.summary.severity_distribution === 'object';

  return (
    <div id="attack-frequency-report">
      <Card className="report-card">
        <div className="report-header">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h2 className="mb-0">📊 Attack Type Frequency Report</h2>
              <p className="mb-0 mt-2">Analysis of attack patterns and frequency distribution in detected threats</p>
            </div>
            <div className="d-flex gap-2">
              <Button 
                variant="light" 
                onClick={fetchReportData}
                disabled={loading}
              >
                {loading ? '🔄 Loading...' : '🔄 Refresh'}
              </Button>
              <Button 
                variant="outline-danger" 
                onClick={clearData}
                disabled={loading}
                size="sm"
              >
                🗑️ Clear Data
              </Button>
            </div>
          </div>
        </div>
        
        <Card.Body>
          {/* Show message if no attack data */}
          {!hasAttackData ? (
            <Alert variant="info" className="text-center">
              <Alert.Heading>📊 No Attack Data Detected</Alert.Heading>
              <p className="mb-3">
                The system has not detected any attacks yet. Run threat detection to generate data for analysis.
              </p>
              <div className="mb-3">
                <strong>Summary:</strong><br />
                Total Attacks: {reportData.summary?.total_attacks || 0}<br />
                Unique Attack Types: {reportData.summary?.unique_attack_types || 0}<br />
                High Severity: {hasSeverityDistribution ? reportData.summary.severity_distribution.High : 0}<br />
                Medium Severity: {hasSeverityDistribution ? reportData.summary.severity_distribution.Medium : 0}
              </div>
              <Button variant="primary" onClick={fetchReportData} disabled={loading}>
                {loading ? '🔄 Loading...' : '🔄 Refresh Data'}
              </Button>
            </Alert>
          ) : (
            <>
              {/* Summary Statistics */}
              <Row className="mb-4">
                <Col md={3}>
                  <div className="stats-card">
                    <div className="stats-number text-primary">{reportData.summary?.total_attacks || 0}</div>
                    <div className="stats-label">Total Attacks</div>
                  </div>
                </Col>
                <Col md={3}>
                  <div className="stats-card">
                    <div className="stats-number text-warning">{reportData.summary?.unique_attack_types || 0}</div>
                    <div className="stats-label">Attack Types</div>
                  </div>
                </Col>
                <Col md={3}>
                  <div className="stats-card">
                    <div className="stats-number text-danger">{hasSeverityDistribution ? reportData.summary.severity_distribution.High : 0}</div>
                    <div className="stats-label">High Severity</div>
                  </div>
                </Col>
                <Col md={3}>
                  <div className="stats-card">
                    <div className="stats-number text-info">{hasSeverityDistribution ? reportData.summary.severity_distribution.Medium : 0}</div>
                    <div className="stats-label">Medium Severity</div>
                  </div>
                </Col>
              </Row>

          {/* Most/Least Common Attacks */}
          {reportData.summary.mostCommonAttack && (
            <Row className="mb-4">
              <Col md={6}>
                <Card className="border-success">
                  <Card.Header className="bg-success text-white">
                    <strong>Most Common Attack</strong>
                  </Card.Header>
                  <Card.Body>
                    <Badge className="attack-badge" style={{backgroundColor: getAttackTypeColor(reportData.summary.mostCommonAttack.attackType)}}>
                      {reportData.summary.mostCommonAttack.attackType}
                    </Badge>
                    <div className="mt-2">
                      <strong>{reportData.summary.mostCommonAttack.count}</strong> occurrences ({reportData.summary.mostCommonAttack.percentage}%)
                    </div>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={6}>
                <Card className="border-info">
                  <Card.Header className="bg-info text-white">
                    <strong>Least Common Attack</strong>
                  </Card.Header>
                  <Card.Body>
                    <Badge className="attack-badge" style={{backgroundColor: getAttackTypeColor(reportData.summary.leastCommonAttack.attackType)}}>
                      {reportData.summary.leastCommonAttack.attackType}
                    </Badge>
                    <div className="mt-2">
                      <strong>{reportData.summary.leastCommonAttack.count}</strong> occurrences ({reportData.summary.leastCommonAttack.percentage}%)
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}

              {/* Attack Frequency Table - Exact format from requirements */}
              <h4 className="mb-3">Attack Type Frequency Analysis</h4>
              <Table hover responsive className="table-striped">
                <thead className="table-dark">
                  <tr>
                    <th>Attack Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {hasAttackFrequencyData && reportData.attackFrequencyData.map((attack, index) => (
                    <tr key={index} className="attack-type-row">
                      <td>
                        <Badge 
                          className="attack-badge fs-6" 
                          style={{backgroundColor: getAttackTypeColor(attack.attackType)}}
                        >
                          {attack.attackType || 'Unknown'} {/* Add fallback */}
                        </Badge>
                      </td>
                      <td>
                        <strong className="fs-5">{attack.count}</strong>
                      </td>
                      <td>
                        <div className="d-flex align-items-center">
                          <div 
                            className="frequency-bar me-2" 
                            style={{width: `${Math.max(attack.percentage, 5)}%`}}
                          >
                            <span className="frequency-text">{attack.percentage}%</span>
                          </div>
                          <span className="fw-bold">{attack.percentage}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              
              {!hasAttackFrequencyData && (
                <Alert variant="warning" className="mt-3">
                  <Alert.Heading>⚠️ Data Structure Issue</Alert.Heading>
                  <p>The attack frequency data structure is incomplete. Please refresh or contact support.</p>
                  <Button variant="outline-warning" onClick={fetchReportData}>
                    🔄 Refresh Data
                  </Button>
                </Alert>
              )}
              
              {!hasSeverityDistribution && (
                <Alert variant="info" className="mt-3">
                  <Alert.Heading>ℹ️ Severity Data Missing</Alert.Heading>
                  <p>Severity distribution data is not available. This may be normal if no threats have been detected.</p>
                </Alert>
              )}

          {/* Bar Chart Visualization - As per requirements */}
          <div className="chart-container">
            <h5 className="mb-3">📊 Attack Type Frequency Bar Chart</h5>
            <div className="bar-chart-container">
              {hasAttackFrequencyData && reportData.attackFrequencyData.map((attack, index) => (
                <div key={attack.attackType} className="bar-chart-item mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className="fw-bold">{attack.attackType}</span>
                    <span className="text-muted">{attack.count} ({attack.percentage}%)</span>
                  </div>
                  <div className="bar-chart-bar-container">
                    <div 
                      className="bar-chart-bar"
                      style={{
                        width: `${Math.max(attack.percentage, 2)}%`,
                        backgroundColor: getAttackTypeColor(attack.attackType)
                      }}
                    >
                      <span className="bar-chart-label">{attack.percentage}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Export Buttons */}
          <div className="export-buttons text-center">
            <Button variant="outline-primary" onClick={exportToPDF} className="me-2">
              📄 Export to PDF
            </Button>
            <Button variant="outline-success" onClick={exportToExcel}>
              📊 Export to Excel
            </Button>
          </div>

          <div className="mt-3 text-center">
            <small className="text-muted">
              Report generated on {new Date().toLocaleString()}
            </small>
          </div>
            </>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}
