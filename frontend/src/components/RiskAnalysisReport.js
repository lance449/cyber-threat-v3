import React, { useState, useEffect } from 'react';
import { Table, Card, Badge, Button, Row, Col, Alert } from 'react-bootstrap';
import { getRiskAnalysisReport, clearReportData, getReportStatus } from '../api';
import 'bootstrap/dist/css/bootstrap.min.css';

const styles = `
  .report-card {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-radius: 12px;
    border: none;
    margin-bottom: 20px;
  }
  
  .report-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px 12px 0 0;
    padding: 20px;
  }
  
  .severity-high {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
  }
  
  .severity-medium {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
  }
  
  .severity-low {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
  }
  
  .recommendation-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
  }
  
  .export-buttons {
    margin-top: 20px;
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
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default function RiskAnalysisReport() {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch data from backend API
  const fetchReportData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getRiskAnalysisReport();
      
      if (response.success) {
        console.log('Risk Analysis Report Data:', response.data);
        setReportData(response.data);
      } else {
        console.error('Risk Analysis Report Error:', response.error);
        setError(response.error || 'Failed to fetch report data');
      }
    } catch (err) {
      console.error('Risk Analysis Report Exception:', err);
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
    // Create a simple PDF export using browser print functionality
    const printWindow = window.open('', '_blank');
    const reportContent = document.getElementById('risk-analysis-report');
    
    printWindow.document.write(`
      <html>
        <head>
          <title>Risk Analysis Summary Report</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .report-header { background: #667eea; color: white; padding: 20px; text-align: center; }
            .severity-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
            .severity-high { border-left: 4px solid #f44336; }
            .severity-medium { border-left: 4px solid #ff9800; }
            .severity-low { border-left: 4px solid #4caf50; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
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

  const exportToCSV = () => {
    if (!reportData) return;

    let csvContent = "Severity Level,Count,Percentage,Top Attack Types,Recommended Action\n";
    
    Object.entries(reportData.severityAnalysis).forEach(([severity, data]) => {
      const topAttacks = data.topAttacks.map(a => `${a.attack} (${a.count})`).join('; ');
      const action = reportData.recommendedActions[severity].action;
      csvContent += `${severity},${data.count},${data.percentage}%,${topAttacks},${action}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `risk_analysis_report_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <Card className="report-card">
        <Card.Body className="text-center">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-2">Generating Risk Analysis Report...</p>
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
        <p>Please run threat detection to generate risk analysis data.</p>
        <Button variant="primary" onClick={fetchReportData} disabled={loading}>
          {loading ? '🔄 Loading...' : '🔄 Refresh Data'}
        </Button>
      </Alert>
    );
  }

  // Check if we have actual threat data and proper structure
  const hasThreatData = reportData?.summary?.total_threats > 0;
  const hasSeverityAnalysis = reportData?.severityAnalysis && 
                             typeof reportData.severityAnalysis === 'object' &&
                             Object.keys(reportData.severityAnalysis).length > 0;

  return (
    <div id="risk-analysis-report">
      <Card className="report-card">
        <div className="report-header">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h2 className="mb-0">🧩 Risk Analysis Summary Report</h2>
              <p className="mb-0 mt-2">Comprehensive threat analysis with severity breakdown and response recommendations</p>
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
          {/* Show message if no threat data */}
          {!hasThreatData ? (
            <Alert variant="info" className="text-center">
              <Alert.Heading>📊 No Threat Data Detected</Alert.Heading>
              <p className="mb-3">
                The system has not detected any threats yet. Run threat detection to generate data for analysis.
              </p>
              <div className="mb-3">
                <strong>Summary:</strong><br />
                Total Threats: {reportData.summary?.total_threats || 0}<br />
                Total Flows: {reportData.summary?.total_flows || 0}<br />
                Threat Rate: {reportData.summary?.threat_rate || 0}%
              </div>
              <Button variant="primary" onClick={fetchReportData} disabled={loading}>
                {loading ? '🔄 Loading...' : '🔄 Refresh Data'}
              </Button>
            </Alert>
          ) : (
            <>
              {/* Summary Statistics */}
              <Row className="mb-4">
                <Col md={4}>
                  <div className="stats-card">
                    <div className="stats-number text-primary">{reportData.summary?.total_threats || 0}</div>
                    <div className="stats-label">Total Threats</div>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="stats-card">
                    <div className="stats-number text-warning">{reportData.summary?.threat_rate || 0}%</div>
                    <div className="stats-label">Threat Rate</div>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="stats-card">
                    <div className="stats-number text-info">{reportData.summary?.total_flows || 0}</div>
                    <div className="stats-label">Total Flows</div>
                  </div>
                </Col>
              </Row>

              {/* Severity Analysis Table - Exact format from requirements */}
              <h4 className="mb-3">Threat Breakdown by Severity</h4>
              <Table hover responsive className="table-striped">
                <thead className="table-dark">
                  <tr>
                    <th>Severity</th>
                    <th>Count</th>
                    <th>Common Attack Types</th>
                    <th>Recommended Action</th>
                  </tr>
                </thead>
                <tbody>
                  {hasSeverityAnalysis ? (
                    Object.entries(reportData.severityAnalysis || {}).map(([severity, data]) => (
                      <tr key={severity} className={`severity-${severity.toLowerCase()}`}>
                        <td>
                          <Badge bg={
                            severity === 'High' ? 'danger' :
                            severity === 'Medium' ? 'warning' : 'success'
                          } className="fs-6">
                            {severity === 'High' ? '🔴 High' : 
                             severity === 'Medium' ? '🟡 Medium' : '🟢 Low'}
                          </Badge>
                        </td>
                        <td>
                          <strong className="fs-5">{data?.count || 0}</strong>
                        </td>
                        <td>
                          {Array.isArray(data?.top_attacks) ? (
                            <div>
                              {data.top_attacks.map((attack, idx) => (
                                <span key={idx}>
                                  <Badge bg="secondary" className="me-1 mb-1">
                                    {attack[0]} ({attack[1]})
                                  </Badge>
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-muted">No attack data available</span>
                          )}
                        </td>
                        <td>
                          <div className="recommendation-box">
                            <strong className="text-primary">
                              {reportData?.recommendedActions?.[severity]?.action || 'No action specified'}
                            </strong>
                            <br />
                            <small className="text-muted">
                              {severity === 'High' && 'Immediate isolation and system scan'}
                              {severity === 'Medium' && 'Monitor and restrict access'}
                              {severity === 'Low' && 'Log and schedule review'}
                            </small>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="text-center">
                        <Alert variant="info" className="my-3">
                          No severity analysis data available
                        </Alert>
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
              
              {!hasSeverityAnalysis && (
                <Alert variant="warning" className="mt-3">
                  <Alert.Heading>⚠️ Data Structure Issue</Alert.Heading>
                  <p>The report data structure is incomplete. Please refresh or contact support.</p>
                  <Button variant="outline-warning" onClick={fetchReportData}>
                    🔄 Refresh Data
                  </Button>
                </Alert>
              )}

          {/* Detailed Recommendations */}
          <h4 className="mb-3 mt-4">Detailed Response Recommendations</h4>
          <Row>
            {reportData.recommendedActions && Object.entries(reportData.recommendedActions).map(([severity, action]) => (
              <Col md={4} key={severity}>
                <Card className={`severity-${severity.toLowerCase()}`}>
                  <Card.Header>
                    <Badge bg={
                      severity === 'High' ? 'danger' :
                      severity === 'Medium' ? 'warning' : 'success'
                    }>
                      {severity} Priority
                    </Badge>
                  </Card.Header>
                  <Card.Body>
                    <h6>{action?.action || 'No action specified'}</h6>
                    <p className="small">{action?.description || 'No description available'}</p>
                    <div className="mt-2">
                      <small className="text-muted">
                        <strong>Timeframe:</strong> {action?.timeframe || 'Not specified'}
                      </small>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>

          {/* Export Buttons */}
          <div className="export-buttons text-center">
            <Button variant="outline-primary" onClick={exportToPDF} className="me-2">
              📄 Export to PDF
            </Button>
            <Button variant="outline-success" onClick={exportToCSV}>
              📊 Export to CSV
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
