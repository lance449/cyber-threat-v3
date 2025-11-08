import React, { useState, useEffect } from 'react';
import { Card, Badge, Button, Row, Col, Alert } from 'react-bootstrap';
import { getAttackFrequencyReport, clearReportData } from '../api';
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
  .attack-badge {
    font-size: 0.8rem;
    padding: 0.4rem 0.8rem;
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
        setReportData(response.data);
      } else {
        setError(response.error || 'Failed to fetch report data');
      }
    } catch (err) {
      setError(err.message || 'Unknown error');
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
        setReportData(null);
        setError(null);
      } else {
        setError(response.error || 'Failed to clear report data');
      }
    } catch (err) {
      setError(err.message || 'Unknown error');
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
    const reportContent = document.getElementById('attack-frequency-chart');
    
    printWindow.document.write(`
      <html>
        <head><title>Attack Type Frequency Report</title></head>
        <body>${reportContent ? reportContent.innerHTML : ''}</body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
  };

  const exportToExcel = () => {
    if (!reportData || !Array.isArray(reportData.attackFrequencyData)) return;
    let csvContent = "Attack Type,Count,Percentage\n";
    reportData.attackFrequencyData.forEach(a => {
      csvContent += `${a.attackType || ''},${a.count || 0},${a.percentage || 0}\n`;
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
      'DoS': '#dc3545', 'DDoS': '#dc3545', 'Trojan': '#fd7e14', 'Virus': '#fd7e14',
      'Worm': '#fd7e14', 'Backdoor': '#fd7e14', 'Rootkit': '#6f42c1', 'Exploit': '#6f42c1',
      'Port Scan': '#ffc107', 'Brute Force': '#ffc107', 'Web Attack': '#20c997',
      'Botnet': '#20c997', 'HackTool': '#6c757d', 'Hoax': '#6c757d', 'Infiltration': '#e83e8c'
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
          <p className="mt-2">Generating Attack Frequency Chart...</p>
        </Card.Body>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Error</Alert.Heading>
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

  const freq = Array.isArray(reportData.attackFrequencyData) ? reportData.attackFrequencyData : [];

  return (
    <div id="attack-frequency-chart">
      <Card className="report-card">
        <div className="report-header">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h2 className="mb-0">📊 Attack Type Frequency</h2>
              <p className="mb-0 mt-2">Bar chart visualization of detected attack types</p>
            </div>
            <div className="d-flex gap-2">
              <Button variant="light" onClick={fetchReportData} disabled={loading}>
                {loading ? '🔄 Loading...' : '🔄 Refresh'}
              </Button>
              <Button variant="outline-danger" onClick={clearData} disabled={loading} size="sm">
                🗑️ Clear Data
              </Button>
            </div>
          </div>
        </div>

        <Card.Body>
          <div className="chart-container">
            <h5 className="mb-3">📊 Attack Type Frequency Bar Chart</h5>
            <div className="bar-chart-container">
              {freq.length === 0 && (
                <Alert variant="info">No attack frequency data to display.</Alert>
              )}
              {freq.map((attack, index) => (
                <div key={attack.attackType || index} className="bar-chart-item mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className="fw-bold">{attack.attackType || 'Unknown'}</span>
                    <span className="text-muted">{attack.count || 0} ({attack.percentage || 0}%)</span>
                  </div>
                  <div className="bar-chart-bar-container">
                    <div
                      className="bar-chart-bar"
                      style={{
                        width: `${Math.max(attack.percentage || 0, 2)}%`,
                        backgroundColor: getAttackTypeColor(attack.attackType)
                      }}
                    >
                      <span className="bar-chart-label">{(attack.percentage || 0) + '%'}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="export-buttons text-center mt-3">
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
        </Card.Body>
      </Card>
    </div>
  );
}
