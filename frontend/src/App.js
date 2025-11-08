import React, { useState } from 'react';
import './styles/App.css';
import DetectionTable from './components/DetectionTable';
import RiskAnalysisReport from './components/RiskAnalysisReport';
import AttackFrequencyReport from './components/AttackFrequencyReport';
import ReliabilityComparisonPanel from './components/ReliabilityComparisonPanel';
import EfficiencyComparisonPanel from './components/EfficiencyComparisonPanel';
import { Nav, Tab, Container, Row, Col } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const styles = `
  .main-container {
    background-color: #f8f9fa;
    min-height: 100vh;
    padding: 20px 0;
  }
  
  .nav-tabs {
    border-bottom: 2px solid #dee2e6;
    margin-bottom: 20px;
  }
  
  .nav-tabs .nav-link {
    border: none;
    border-radius: 8px 8px 0 0;
    margin-right: 5px;
    padding: 12px 24px;
    font-weight: 500;
    color: #6c757d;
    background-color: transparent;
    transition: all 0.3s ease;
  }
  
  .nav-tabs .nav-link:hover {
    background-color: #e9ecef;
    color: #495057;
  }
  
  .nav-tabs .nav-link.active {
    background-color: #007bff;
    color: white;
    border-color: #007bff;
  }
  
  .tab-content {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 20px;
  }
  
  .dashboard-header {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
  }
  
  .dashboard-title {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 10px;
  }
  
  .dashboard-subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
  }
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

function App() {
  const [activeTab, setActiveTab] = useState('detection');

  return (
    <div className="main-container">
      <Container fluid>
        <div className="dashboard-header">
          <h1 className="dashboard-title">🛡️ Cyber Threat Detection System</h1>
          <p className="dashboard-subtitle">Machine Learning-Powered Security Analysis Dashboard</p>
        </div>
        
        <Tab.Container activeKey={activeTab} onSelect={setActiveTab}>
          <Row>
            <Col>
              <Nav variant="tabs" className="nav-tabs">
                <Nav.Item>
                  <Nav.Link eventKey="detection">
                    🔍 Real-time Detection
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="risk-analysis">
                    🧩 Risk Analysis Report
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="attack-frequency">
                    📊 Attack Frequency Report
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="reliability-comparison">
                    ✅ Reliability Comparison
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="efficiency-comparison">
                    ⚡ Efficiency Comparison
                  </Nav.Link>
                </Nav.Item>
              </Nav>
              
              <Tab.Content>
                <Tab.Pane eventKey="detection">
                  <div className="tab-content">
                    <DetectionTable />
                  </div>
                </Tab.Pane>
                
                <Tab.Pane eventKey="risk-analysis">
                  <div className="tab-content">
                    <RiskAnalysisReport />
                  </div>
                </Tab.Pane>
                
                <Tab.Pane eventKey="attack-frequency">
                  <div className="tab-content">
                    <AttackFrequencyReport />
                  </div>
                </Tab.Pane>
                <Tab.Pane eventKey="reliability-comparison">
                  <div className="tab-content">
                    <ReliabilityComparisonPanel />
                  </div>
                </Tab.Pane>
                <Tab.Pane eventKey="efficiency-comparison">
                  <div className="tab-content">
                    <EfficiencyComparisonPanel />
                  </div>
                </Tab.Pane>
              </Tab.Content>
            </Col>
          </Row>
        </Tab.Container>
      </Container>
    </div>
  );
}

export default App;