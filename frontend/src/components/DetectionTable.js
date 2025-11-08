import React, { useEffect, useState, useRef } from 'react';
import { resetStream, getNextRow } from '../api';
import PerformancePanel from './PerformancePanel';
import { Table, Badge, Button, ButtonGroup, Form, Row, Col } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../styles/detection-table.css'; 

function severityStyle(sev) {
  if (!sev) return {};
  if (sev.toLowerCase() === 'high') return { backgroundColor: '#ffeded' };
  if (sev.toLowerCase() === 'medium') return { backgroundColor: '#fff9e6' };
  return { backgroundColor: '#e9ffec' };
}

const AttackTypeFilter = ({ selectedTypes, onChange, rows }) => {
  const attackTypes = [...new Set(rows.map(r => r.attack_type))].filter(Boolean).sort();
  
  const handleChange = (type) => {
    const updated = selectedTypes.includes(type)
      ? selectedTypes.filter(t => t !== type)
      : [...selectedTypes, type];
    onChange(updated);
  };

  return (
    <div className="filter-section">
      <Form.Label>Attack Types</Form.Label>
      <div className="attack-type-select">
        {attackTypes.map(type => (
          <div 
            key={type}
            className={`attack-type-option ${
              selectedTypes.includes(type) ? 'selected' : ''
            }`}
            onClick={() => handleChange(type)}
          >
            <input
              type="checkbox"
              checked={selectedTypes.includes(type)}
              onChange={() => {}}
            />
            {type}
          </div>
        ))}
      </div>
    </div>
  );
};

// Feature explanations mapping for tooltips
const featureExplanations = {
  'Flow Duration': {
    description: 'Duration of the network flow in seconds',
    whyItMatters: 'Short flows may indicate port scanning or DDoS attacks, while abnormally long flows could suggest data exfiltration or backdoors maintaining persistent connections.'
  },
  'Total Fwd Packet': {
    description: 'Total number of packets sent from source to destination',
    whyItMatters: 'Unusually high packet counts often correlate with brute force attacks, port scanning, or DDoS attempts trying to overwhelm the target.'
  },
  'Total Bwd packets': {
    description: 'Total number of packets sent from destination back to source',
    whyItMatters: 'Low backward packets relative to forward packets may indicate one-way attacks like SYN floods or port scanning without establishing full connections.'
  },
  'Fwd Packet Length Mean': {
    description: 'Average length of packets traveling forward (source to destination)',
    whyItMatters: 'Very small packets might indicate scanning or probing attacks, while abnormally large packets could signal data exfiltration attempts.'
  },
  'Bwd Packet Length Mean': {
    description: 'Average length of packets traveling backward (destination to source)',
    whyItMatters: 'Irregular backward packet sizes can indicate malicious responses, command-and-control traffic, or successful exploitation payloads.'
  },
  'Flow Bytes/s': {
    description: 'Rate of data transfer in bytes per second',
    whyItMatters: 'Extremely high transfer rates may indicate data exfiltration or botnet activity, while suspiciously low rates could indicate slow infiltration or stealth attacks.'
  },
  'Flow Packets/s': {
    description: 'Rate of packet transmission in packets per second',
    whyItMatters: 'High packet rates often signal flooding attacks or rapid scanning, while abrupt rate changes can indicate attack progression or tool switching.'
  },
  'FIN Flag Count': {
    description: 'Number of packets with FIN flag set, indicating connection termination',
    whyItMatters: 'Abnormal FIN flag patterns may indicate port scanning techniques or attempts to probe firewall behavior and identify open ports.'
  },
  'SYN Flag Count': {
    description: 'Number of packets with SYN flag set, indicating connection initiation',
    whyItMatters: 'High SYN counts with no corresponding ACK confirmations are classic signs of SYN flood attacks or port scanning attempts.'
  },
  'ACK Flag Count': {
    description: 'Number of packets with ACK flag set, indicating acknowledgment of received data',
    whyItMatters: 'Low ACK counts paired with high SYN counts suggest incomplete connections typical of scanning or flooding attacks.'
  },
  'Total Length of Fwd Packets': {
    description: 'Total bytes in all forward packets',
    whyItMatters: 'Abnormally high total forward bytes may indicate data exfiltration or brute force attempts, while very low values could signal stealthy reconnaissance.'
  },
  'Total Length of Bwd Packets': {
    description: 'Total bytes in all backward packets',
    whyItMatters: 'Low backward bytes suggest one-way attacks like DDoS flooding, while high values might indicate command-and-control traffic or data exfiltration.'
  },
  'Fwd PSH Flags': {
    description: 'Number of packets with PSH flag (push) from source',
    whyItMatters: 'Abnormal PSH patterns can indicate rapid data transmission typical of scanning, brute force, or data exfiltration attacks.'
  },
  'Bwd PSH Flags': {
    description: 'Number of packets with PSH flag (push) from destination',
    whyItMatters: 'High backward PSH flags may indicate data exfiltration or command-and-control responses from compromised systems.'
  },
  'Average Packet Size': {
    description: 'Mean size of all packets in the flow',
    whyItMatters: 'Very small average packet sizes often indicate port scanning or probing, while large packets may signal data transfer or payloads.'
  },
  'Init_Win_bytes_forward': {
    description: 'Initial window size for forward direction in bytes',
    whyItMatters: 'Unusual initial window sizes can indicate connection hijacking attempts or stealth techniques used by advanced persistent threats.'
  },
  'Init_Win_bytes_backward': {
    description: 'Initial window size for backward direction in bytes',
    whyItMatters: 'Non-standard backward window sizes may suggest reverse connections established by backdoors or command-and-control channels.'
  },
  'Active Mean': {
    description: 'Mean time between active forward packets',
    whyItMatters: 'Very short intervals suggest burst attacks or scanning, while long intervals may indicate slow-and-low infiltration techniques.'
  },
  'Idle Mean': {
    description: 'Mean time between idle periods',
    whyItMatters: 'Irregular idle patterns can reveal human-driven attacks, botnet coordination pauses, or evasion techniques designed to avoid detection.'
  }
};

function getFeatureExplanation(featureName) {
  return featureExplanations[featureName] || {
    description: 'Network flow metric',
    whyItMatters: 'This feature contributes to the overall threat assessment by analyzing network traffic patterns.'
  };
}

// Generate contextual recommendations based on antivirusype, features, and model consensus
function generateContextualAnalysis(row) {
  const attackType = row.attack_type || 'Unknown';
  const severity = row.severity?.toLowerCase() || 'low';
  const keyFeatures = row.key_features || {};
  const modelCount = (row.model_used || '').split(',').filter(m => m.trim()).length;
  const modelConsensus = row.model_used || 'unknown';
  
  // Count how many models detected this
  const modelsArray = modelConsensus.split(',').map(m => m.trim()).filter(m => m);
  const detectedModels = modelsArray.length;
  
  // Analyze key features to determine reasons
  const reasons = [];
  
  // Format numbers for display
  const fmt = (n) => {
    if (!n) return '';
    const v = parseFloat(n);
    if (v >= 1000000) return (v/1000000).toFixed(1) + 'M';
    if (v >= 1000) return (v/1000).toFixed(0) + 'K';
    return v.toFixed(1);
  };
  
  // Analyze actual values to find specific anomalies
  const fwdb = keyFeatures['Flow Bytes/s'];
  const pktsps = keyFeatures['Flow Packets/s'];
  const dur = keyFeatures['Flow Duration'];
  const fwdpkts = keyFeatures['Total Fwd Packet'];
  const bwdpkts = keyFeatures['Total Bwd packets'];
  const syn = keyFeatures['SYN Flag Count'];
  const ack = keyFeatures['ACK Flag Count'];
  const fwdbw = keyFeatures['Total Length of Fwd Packets'];
  const bwpush = keyFeatures['Bwd PSH Flags'];
  
  // Analyze THIS traffic's specific behavioral anomalies - why this instance is suspicious
  
  // One-way traffic anomaly (asymmetric pattern)
  if (fwdpkts && bwdpkts !== undefined && bwdpkts < fwdpkts * 0.15 && fwdpkts > 50) {
    const ratio = (fwdpkts / (bwdpkts || 1)).toFixed(0);
    reasons.push(`${ratio}x more outbound than return traffic (${fmt(fwdpkts)} vs ${bwdpkts} packets)`);
  }
  
  // Sustained connection (backdoor/long-lived threat)
  if (dur && dur > 180) {
    const minutes = Math.round(dur / 60);
    if (minutes >= 60) {
      reasons.push(`Continuous ${minutes}min connection (unusually long)`);
    } else if (minutes > 10) {
      reasons.push(`Persistent ${minutes}min connection`);
    }
  }
  
  // Incomplete handshakes (scanning/flooding pattern)
  if (syn && ack !== undefined && syn > 10) {
    const completionRate = ((ack / syn) * 100).toFixed(0);
    if (completionRate < 30) {
      reasons.push(`Only ${completionRate}% connections complete (${syn} attempts, ${ack} succeeded)`);
    }
  }
  
  // Data exfiltration patterns
  if (fwdbw && dur && dur > 5) {
    const rate = fwdbw / dur;
    if (rate > 50000) {
      reasons.push(`${fmt(fwdbw)}B transferred at ${fmt(rate)}B/s (high volume)`);
    }
  }
  
  // Packet burst anomalies
  if (pktsps && pktsps > 150) {
    reasons.push(`Burst traffic at ${fmt(pktsps)} packets/sec (potential flood)`);
  }
  
  // Command & control indicators (imbalanced push responses)
  if (bwpush && fwdpkts && bwpush > fwdpkts * 0.15 && fwdpkts > 30) {
    reasons.push(`${bwpush} forced push responses from destination (possible C2)`);
  }
  
  // Fallback: Extract and show the most important contributing features
  if (reasons.length === 0) {
    const attackLower = attackType.toLowerCase();
    
    if (attackLower === 'benign') {
      reasons.push('Normal balanced traffic');
    } else if (attackLower.includes('hoax')) {
      reasons.push('False positive - benign pattern');
    } else {
      // Find the top contributing features from key_features
      const features = row.key_features || {};
      
      // Sort features by absolute value to find the most significant ones
      const sortedFeatures = Object.entries(features)
        .filter(([key, value]) => value !== undefined && value !== null && !isNaN(value) && Math.abs(parseFloat(value)) > 0.001)
        .sort((a, b) => Math.abs(parseFloat(b[1])) - Math.abs(parseFloat(a[1])))
        .slice(0, 3); // Get top 3 features
      
      if (sortedFeatures.length > 0) {
        // Build a meaningful reason from the top features
        const featureDescriptions = sortedFeatures.map(([key, value]) => {
          const numValue = parseFloat(value);
          // Format based on feature type
          if (key.includes('Packet') || key.includes('packet')) {
            return `${Math.round(numValue)} pkts`;
          } else if (key.includes('Duration')) {
            return `${Math.round(numValue)}s`;
          } else if (key.includes('Bytes') || key.includes('bytes')) {
            return `${fmt(value)}B`;
          } else if (key.includes('Flags') || key.includes('flags')) {
            return `${Math.round(numValue)} ${key.includes('SYN') ? 'SYNs' : key.includes('ACK') ? 'ACKs' : 'flags'}`;
          } else if (key.includes('Mean') || key.includes('mean')) {
            return `${numValue.toFixed(1)} avg`;
          } else {
            return numValue.toFixed(2);
          }
        });
        
        // Show feature names with values and explain severity
        const topFeatureName = sortedFeatures[0][0];
        let reasonText = '';
        
        if (attackLower.includes('backdoor') || attackLower.includes('exploit') || attackLower.includes('virus') || attackLower.includes('trojan') || attackLower.includes('worm')) {
          reasonText = `HIGH RISK threat type - ${topFeatureName} pattern detected`;
        } else if (attackLower.includes('hacktool')) {
          reasonText = `Tool usage detected via ${topFeatureName}`;
        } else {
          reasonText = `${topFeatureName} pattern matching threat signatures`;
        }
        
        reasons.push(reasonText);
      } else {
        reasons.push('ML threat pattern detected');
      }
    }
  }
  
  
  // Generate attack-specific recommendations
  const dstIp = row.dst_ip;
  const srcIp = row.src_ip;
  let recommendation = '';
  
  // Generate intelligent, actionable recommendations based on threat type and features
  const attackLower = attackType.toLowerCase();
  
  if (attackLower === 'benign') {
    recommendation = 'Normal traffic - no action required';
  } 
  // Virus: Focus on containment and cleanup
  else if (attackLower.includes('virus')) {
    if (fwdb > 500000 || fwdbw > 1000000) {
      recommendation = `Quarantine ${dstIp} | Full AV scan | Disable autorun | Check critical processes`;
    } else if (fwdpkts > 5000) {
      recommendation = `Isolate ${dstIp} from network | Run anti-virus scan | Check for mass connections`;
    } else {
      recommendation = `Disconnect ${dstIp} | Update AV definitions | Scan for malware | Review startup programs`;
    }
  }
  // Trojan: Focus on backdoor removal
  else if (attackLower.includes('trojan')) {
    if (bwpush > 30 && dur > 3000) {
      recommendation = `${dstIp} likely compromised | Block C2 outbound traffic | Terminate suspicious processes | Check scheduled tasks`;
    } else if (dur > 1800) {
      recommendation = `Backdoor active on ${dstIp} | Disconnect network | Audit running services | Reinstall if compromised`;
    } else {
      recommendation = `Audit ${dstIp} installed software | Check registry | Review network connections | Verify system integrity`;
    }
  }
  // Worm: Focus on network containment
  else if (attackLower.includes('worm')) {
    if (fwdpkts > 1000 || pktsps > 200) {
      recommendation = `Quarantine entire network segment (192.168.x.x) | Scan all adjacent hosts | Block lateral movement`;
    } else {
      recommendation = `Isolate ${dstIp} network segment | Scan 192.168.x.x range | Check for SMB/CIFS shares | Block propagation`;
    }
  }
  // Exploit: Focus on patching and blocking
  else if (attackLower.includes('exploit')) {
    if (syn > 100) {
      recommendation = `Block ${srcIp} at perimeter | Patch ${dstIp} immediately | Review auth logs for breach | Verify no compromise`;
    } else {
      recommendation = `Assess ${dstIp} patch level | Review security logs | Verify attempted exploitation failed`;
    }
  }
  // Backdoor: Focus on immediate isolation
  else if (attackLower.includes('backdoor')) {
    if (dur > 1800) {
      recommendation = `${dstIp} has active backdoor | Disconnect immediately | Review listening ports | Check for persistent access`;
    } else {
      recommendation = `Isolate ${dstIp} | List listening ports | Audit scheduled tasks | Check startup entries | Block persistence`;
    }
  }
  // Rootkit: Focus on deep remediation
  else if (attackLower.includes('rootkit')) {
    recommendation = `${dstIp} kernel compromised | Boot from clean media | Full system reinstall recommended | Verify BIOS integrity`;
  }
  // DDoS: Focus on mitigation
  else if (attackLower.includes('ddos')) {
    recommendation = `Block ${srcIp} at firewall | Enable DDoS protection | Scale bandwidth | Rate limit connections`;
  }
  // Scan: Focus on monitoring
  else if (attackLower.includes('scan')) {
    recommendation = `Watch ${srcIp} activity | Add firewall rules | Log probes | Document reconnaissance pattern`;
  }
  // HackTool: Focus on investigation
  else if (attackLower.includes('hacktool')) {
    if (fwdbw > 2000000) {
      recommendation = `${dstIp} may be exfiltrating data | Check network shares | Review recent file transfers | Audit user activity`;
    } else {
      recommendation = `Review ${dstIp} processes | Check installed tools | Verify legitimate use | Monitor for activity`;
    }
  }

  else if (attackLower.includes('hoax')) {
    recommendation = 'Low priority - verify with additional monitoring before action';
  }
  // Generic: Standard IR
  else {
    recommendation = `🔍 INVESTIGATE: Review ${dstIp} network behavior | Apply standard incident response protocol | Document findings`;
  }
  
  // Add confidence indicator to recommendation
  if (detectedModels === 3 && attackType.toLowerCase() !== 'benign') {
    recommendation += ' [3/3 high confidence]';
  } else if (detectedModels === 2 && attackType.toLowerCase() !== 'benign') {
    recommendation += ' [2/3 models]';
  } else if (detectedModels === 1 && attackType.toLowerCase() !== 'benign') {
    recommendation += ' [Verify manually]';
  }
  
  // Generate the display text - show detection info and actionable recommendation
  let displayText = `${attackType} — detected by ${detectedModels}/3 models`;
  displayText += `\n💡 Recommendation: ${recommendation}`;
  
  return displayText;
}

export default function DetectionTable() {
  const [rows, setRows] = useState([]);
  const [running, setRunning] = useState(false);
  const [remaining, setRemaining] = useState(null);
  const timerRef = useRef(null);
  const [verifications, setVerifications] = useState({});
  const [filters, setFilters] = useState({
    attackTypes: [],
    severity: '',
    verification: '',
    models: []
  });
  const [performanceStats, setPerformanceStats] = useState({
    'Fuzzy Logic + Random Forest': { accuracy: 0, verified: 0 },
    'IntruDTree Algorithm': { accuracy: 0, verified: 0 },
    'Aho–Corasick + SVM': { accuracy: 0, verified: 0 }
  }); // Add state for performance stats
  const [expandedRows, setExpandedRows] = useState({});
  const [analysisTimers, setAnalysisTimers] = useState({}); // Track analysis time per threat

  // computed stats for PerformancePanel
  const totalDetected = rows.length;
  // prefer an explicit predicted flag if available, otherwise treat model_used presence as positive
  const algoPositives = rows.filter(r => {
    if (!r) return false;
    if (typeof r.predicted_is_threat === 'boolean') return r.predicted_is_threat === true;
    if (!r.model_used) return false;
    const mu = String(r.model_used).trim().toLowerCase();
    if (!mu || mu === 'unknown' || mu === 'simulated') return false;
    return true;
  }).length;
  const totalVerified = Object.keys(verifications || {}).length;
  const verifiedTrue = Object.values(verifications || {}).filter(v => v === true).length;
  const detectionAccuracy = algoPositives > 0 ? Math.round((verifiedTrue / algoPositives) * 100) : 0;
  const manualPrecision = totalVerified > 0 ? Math.round((verifiedTrue / totalVerified) * 100) : 0;

  // Start analysis timer when user expands row (starts reviewing threat)
  const handleRowExpand = (flowId) => {
    setExpandedRows(prev => {
      const isExpanding = !prev[flowId];
      
      // Start timer when user expands row (starts reviewing)
      if (isExpanding && !analysisTimers[flowId]) {
        setAnalysisTimers(prevTimers => ({
          ...prevTimers,
          [flowId]: {
            startTime: Date.now(),
            isActive: true
          }
        }));
      }
      
      return {
        ...prev,
        [flowId]: !prev[flowId]
      };
    });
  };

  const handleManualVerification = (flowId, isVerified) => {
    setVerifications(prev => ({
      ...prev,
      [flowId]: isVerified
    }));

    // Calculate analysis time (Solution 1: Auto-start timer on user interaction)
    let analysisTime = 0;
    if (analysisTimers[flowId]) {
      const timer = analysisTimers[flowId];
      const endTime = Date.now();
      const elapsed = (endTime - timer.startTime) / 1000; // Convert to seconds
      
      // Minimum 2 seconds (Solution 6: Minimum threshold to prevent accidental clicks)
      analysisTime = Math.max(elapsed, 2);
      
      // Remove timer after use
      setAnalysisTimers(prev => {
        const newTimers = { ...prev };
        delete newTimers[flowId];
        return newTimers;
      });
    } else {
      // If user verifies without expanding row, use default minimum time
      analysisTime = 2;
    }

    // Make API call to backend with analysis_time
    fetch('http://localhost:5000/api/manual/verify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        verifications: [{
          flow_id: flowId,
          is_threat: isVerified,
          analysis_time: analysisTime // Send analysis time to backend
        }]
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // After successful verification, update performance metrics
        updateModelPerformance();
      }
    })
    .catch(error => console.error('Error:', error));
  };

  // Add this new function to update performance metrics
  const updateModelPerformance = () => {
    fetch('http://localhost:5000/api/manual/metrics')
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          // Update the performance stats in state
          setPerformanceStats(data.metrics);
        }
      })
      .catch(error => console.error('Error:', error));
  };

  useEffect(() => {
    let mounted = true;

    async function startStream() {
      const r = await resetStream();
      if (!mounted) return;
      if (r && r.success) {
        setRunning(true);
        await fetchOne();
        timerRef.current = setInterval(fetchOne, 5000);
      } else {
        console.error('resetStream failed', r);
      }
    }

    startStream();

    async function fetchOne() {
      try {
        const res = await getNextRow();
        if (!res) return;
        if (res.finished) {
          setRunning(false);
          setRemaining(0);
          clearInterval(timerRef.current);
          return;
        }
        if (res.success && res.row) {
          setRows(prev => [res.row, ...prev]);
          if (typeof res.remaining !== 'undefined') setRemaining(res.remaining);
        } else {
          setRunning(false);
          clearInterval(timerRef.current);
        }
      } catch (e) {
        console.error('fetchOne error', e);
        setRunning(false);
        clearInterval(timerRef.current);
      }
    }

    return () => { mounted = false; clearInterval(timerRef.current); };
  }, []);

  // Helper function to get valid rows (detected by models, not Unknown)
  const getValidRows = () => {
    return rows.filter(row => {
      // Exclude "Unknown" threats
      if (row.attack_type === 'Unknown' || row.attack_type === 'unknown') {
        return false;
      }
      
      // Only include threats detected by at least one model
      const modelUsed = String(row.model_used || '').trim().toLowerCase();
      if (!modelUsed || modelUsed === 'unknown' || modelUsed === 'simulated') {
        return false;
      }
      
      return true;
    });
  };

  // Add this helper function to get unique models from rows
  const getUniqueModels = (rows) => {
    const models = new Set();
    rows.forEach(row => {
      if (row.model_used) {
        row.model_used.split(',').forEach(model => {
          const normalizedModel = model.trim().toLowerCase();
          switch(normalizedModel) {
            case 'fuzzy_rf':
              models.add('Fuzzy Logic + Random Forest');
              break;
            case 'intrudtree':
              models.add('IntruDTree Algorithm');
              break;
            case 'aca_svm':
              models.add('Aho–Corasick + SVM');
              break;
            default:
              if (normalizedModel !== 'unknown' && normalizedModel !== 'simulated') {
                models.add(model.trim());
              }
          }
        });
      }
    });
    return Array.from(models).sort();
  };

  const getFilteredRows = () => {
    return rows.filter(row => {
      // Exclude "Unknown" threats as they are unnecessary
      if (row.attack_type === 'Unknown' || row.attack_type === 'unknown') {
        return false;
      }
      
      // Only display threats detected by at least one model
      // Exclude rows where no model detected the threat (ground truth only)
      const modelUsed = String(row.model_used || '').trim().toLowerCase();
      if (!modelUsed || modelUsed === 'unknown' || modelUsed === 'simulated') {
        return false; // Skip this threat as no model detected it
      }
      
      // Attack types filter (multiple)
      if (filters.attackTypes.length > 0 && 
          !filters.attackTypes.includes(row.attack_type)) {
        return false;
      }
      
      // Severity filter
      if (filters.severity && 
          row.severity?.toLowerCase() !== filters.severity.toLowerCase()) {
        return false;
      }
      
      // Verification filter
      if (filters.verification) {
        const isVerified = verifications.hasOwnProperty(row.flow_id);
        if (filters.verification === 'verified' && !isVerified) return false;
        if (filters.verification === 'unverified' && isVerified) return false;
      }
      
      // Model filter
      if (filters.models.length > 0) {
        const rowModels = String(row.model_used || '')
          .split(',')
          .map(model => {
            const normalizedModel = model.trim().toLowerCase();
            switch(normalizedModel) {
              case 'fuzzy_rf': return 'Fuzzy Logic + Random Forest';
              case 'intrudtree': return 'IntruDTree Algorithm';
              case 'aca_svm': return 'Aho–Corasick + SVM';
              default: return model.trim();
            }
          });
        
        if (!filters.models.some(selectedModel => 
          rowModels.some(rowModel => rowModel === selectedModel)
        )) {
          return false;
        }
      }
      
      return true;
    });
  };

  const filteredRows = getFilteredRows();
  
  // Calculate unverified count for header display
  const unverifiedCount = filteredRows.filter(row => {
    const modelUsed = String(row.model_used || '').trim().toLowerCase();
    const hasDetection = modelUsed && modelUsed !== 'unknown' && modelUsed !== '';
    return hasDetection && !verifications.hasOwnProperty(row.flow_id);
  }).length;

  const handleFilterChange = (filterType, value) => {
    if (filterType === 'attackTypes') {
      // Handle multi-select for attack types
      setFilters(prev => ({
        ...prev,
        attackTypes: Array.from(value)
      }));
    } else if (filterType === 'models') {
      // Handle models as an array
      setFilters(prev => ({
        ...prev,
        models: Array.from(value)
      }));
    } else {
      setFilters(prev => ({
        ...prev,
        [filterType]: value
      }));
    }
  };

  return (
    <div>
      <PerformancePanel 
        rows={rows}
        verifications={verifications}
        performanceStats={performanceStats}
      />

      <div className="mb-3 p-2 bg-light rounded">
        <strong>Real-time simulation:</strong>{' '}
        <span className={`badge ${running ? 'bg-success' : 'bg-secondary'}`}>
          {running ? 'running' : 'stopped'}
        </span>
        {remaining !== null && <span className="ms-2">remaining: {remaining}</span>}
      </div>

      <div className="filters-section mb-3 p-3 bg-light rounded">
        <div className="filters-header">
          <div className="filters-title">
            <i className="fas fa-filter"></i>
            Filter Results
          </div>
          <div className="filter-buttons">
            <button 
              className="filter-btn reset"
              onClick={() => setFilters({
                attackTypes: [], 
                severity: '', 
                verification: '',
                models: [] 
              })}
            >
              Reset Filters
            </button>
          </div>
        </div>

        <Row>
          <Col md={3}>
            <Form.Group>
              <Form.Label>
                <i className="fas fa-shield-alt me-2"></i>
                Attack Types
              </Form.Label>
              <Form.Select
                className="filter-select"
                value={filters.attackTypes.length > 0 ? filters.attackTypes[0] : ''}
                onChange={(e) => {
                  const selectedOptions = Array.from(e.target.selectedOptions, option => option.value);
                  // Filter out empty strings to ensure "All Attack Types" results in empty array
                  const filteredOptions = selectedOptions.filter(opt => opt !== '');
                  handleFilterChange('attackTypes', filteredOptions);
                }}
              >
                <option value="">All Attack Types</option>
                {[...new Set(getValidRows().map(r => r.attack_type))]
                  .filter(Boolean)
                  .sort()
                  .map(type => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))
                }
              </Form.Select>
            </Form.Group>
          </Col>
          
          <Col md={3}>
            <Form.Group>
              <Form.Label>
                <i className="fas fa-exclamation-triangle me-2"></i>
                Severity
              </Form.Label>
              <Form.Select
                className="filter-select"
                value={filters.severity}
                onChange={(e) => handleFilterChange('severity', e.target.value)}
              >
                <option value="">All Severity Levels</option>
                <option value="high">High Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="low">Low Risk</option>
              </Form.Select>
            </Form.Group>
          </Col>
          
          <Col md={3}>
            <Form.Group>
              <Form.Label>
                <i className="fas fa-check-circle me-2"></i>
                Verification Status
              </Form.Label>
              <Form.Select
                className="filter-select"
                value={filters.verification}
                onChange={(e) => handleFilterChange('verification', e.target.value)}
              >
                <option value="">All Status</option>
                <option value="verified">Verified</option>
                <option value="unverified">Unverified</option>
              </Form.Select>
            </Form.Group>
          </Col>

          <Col md={3}>
            <Form.Group>
              <Form.Label>
                <i className="fas fa-robot me-2"></i>
                Detection Models
              </Form.Label>
              <Form.Select
                className="filter-select"
                value={filters.models.length > 0 ? filters.models[0] : ''}
                onChange={(e) => {
                  const value = e.target.value;
                  // If "All Models" is selected, clear the models filter
                  if (!value) {
                    handleFilterChange('models', []);
                  } else {
                    handleFilterChange('models', [value]);
                  }
                }}
              >
                <option value="">All Models</option>
                {getUniqueModels(getValidRows()).map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
        </Row>

        {/* Active filters section */}
        {(filters.attackTypes.length > 0 || filters.severity || filters.verification) && (
          <div className="active-filters mt-3">
            <span className="filter-label me-2">Active Filters:</span>
            
            {filters.attackTypes.map(type => (
              <span key={type} className="filter-tag attack-type">
                <i className="fas fa-shield-alt me-1"></i>
                {type}
                <span 
                  className="remove"
                  onClick={() => handleFilterChange('attackTypes', 
                    filters.attackTypes.filter(t => t !== type)
                  )}
                  title="Remove filter"
                >
                  ×
                </span>
              </span>
            ))}
            
            {filters.severity && (
              <span className={`filter-tag severity ${filters.severity.toLowerCase()}`}>
                <i className="fas fa-exclamation-triangle"></i>
                {filters.severity.charAt(0).toUpperCase() + filters.severity.slice(1)} Risk
                <span 
                  className="remove"
                  onClick={() => handleFilterChange('severity', '')}
                  title="Remove filter"
                >×</span>
              </span>
            )}
            
            {filters.verification && (
              <span className={`filter-tag verification ${filters.verification.toLowerCase()}`}>
                <i className="fas fa-check-circle"></i>
                {filters.verification.charAt(0).toUpperCase() + filters.verification.slice(1)}
                <span 
                  className="remove"
                  onClick={() => handleFilterChange('verification', '')}
                  title="Remove filter"
                >×</span>
              </span>
            )}

            {/* Add this in the active filters section */}
            {filters.models.map(model => (
              <span key={model} className="filter-tag model">
                <i className="fas fa-robot me-1"></i>
                {model}
                <span 
                  className="remove"
                  onClick={() => handleFilterChange('models', 
                    filters.models.filter(m => m !== model)
                  )}
                  title="Remove filter"
                >
                  ×
                </span>
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="table-container">
        <Table hover className="custom-table">
          <thead>
            <tr>
              <th style={{ width: '160px', minWidth: '160px' }}>
                <div className="d-flex flex-column align-items-center gap-1">
                  <div className="fw-bold">Verification</div>
                  {unverifiedCount > 0 && (
                    <span className="badge bg-warning" style={{ fontSize: '0.7rem' }}>
                      {unverifiedCount} pending
                    </span>
                  )}
                </div>
              </th>
              <th>Attack IP</th>
              <th>Victim IP</th>
              <th>Detection Source</th>
              <th>Attack Details</th>
              <th>Severity</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {getFilteredRows().map((r, idx) => (
              <React.Fragment key={idx}>
                <tr className={`align-middle ${
                  r.severity?.toLowerCase() === 'high' ? 'table-danger' : 
                  r.severity?.toLowerCase() === 'medium' ? 'table-warning' : 
                  'table-success'
                }`}>
                  <td>
                    {(() => {
                      const modelUsed = String(r.model_used || '').trim().toLowerCase();
                      const hasDetection = modelUsed && modelUsed !== 'unknown' && modelUsed !== '';
                      const isVerified = verifications.hasOwnProperty(r.flow_id);
                      const verificationStatus = verifications[r.flow_id];
                      
                      if (!hasDetection) {
                        return (
                          <div className="text-center">
                            <span className="badge bg-secondary" style={{ fontSize: '0.7rem' }} title="Ground truth only - no verification needed">
                              ⚪
                            </span>
                          </div>
                        );
                      }
                      
                      // Check if threat is benign - don't show verification buttons for benign
                      // For accuracy/precision calculation, we only verify threats detected as threats (not benign)
                      const attackType = String(r.attack_type || r.attack_label || '').trim().toLowerCase();
                      const isBenign = attackType === 'benign';
                      
                      if (isBenign) {
                        return (
                          <div className="text-center">
                            <span className="badge bg-info" style={{ fontSize: '0.7rem' }} title="Benign traffic - no verification needed for accuracy calculation">
                              🟢 Benign
                            </span>
                            <br />
                            <small className="text-muted" style={{ fontSize: '0.65rem' }}>
                              Not verifiable
                            </small>
                          </div>
                        );
                      }
                      
                      // Show status badge
                      let statusBadge;
                      if (isVerified && verificationStatus === true) {
                        statusBadge = <span className="badge bg-success" style={{ fontSize: '0.7rem' }} title="Confirmed">🟢</span>;
                      } else if (isVerified && verificationStatus === false) {
                        statusBadge = <span className="badge bg-danger" style={{ fontSize: '0.7rem' }} title="False">🔴</span>;
                      } else {
                        statusBadge = <span className="badge bg-warning" style={{ fontSize: '0.7rem' }} title="Pending verification">⚪</span>;
                      }
                      
                      // Show verified status prominently, or show action buttons for unverified
                      if (isVerified) {
                        return (
                          <div className="d-flex flex-column gap-2 align-items-center" style={{ padding: '0.75rem 0.5rem', width: '100%' }}>
                            {/* Verified status - show prominently with solid button styling */}
                            <div className="d-flex flex-column align-items-center gap-1" style={{ width: '100%' }}>
                              {statusBadge}
                              <div 
                                className={`fw-bold ${verificationStatus === true ? 'text-success' : 'text-danger'}`}
                                style={{ fontSize: '0.85rem', textAlign: 'center', width: '100%' }}
                              >
                                {verificationStatus === true ? 'Threat Confirmed' : 'False Positive'}
                              </div>
                              {/* Show what was verified with a visual indicator */}
                              <div 
                                className={`badge ${verificationStatus === true ? 'bg-success' : 'bg-danger'}`}
                                style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem', marginTop: '0.25rem' }}
                              >
                                {verificationStatus === true ? '✓ Verified' : '✗ False'}
                              </div>
                            </div>
                            {/* Allow changing verification */}
                            <button
                              className="btn btn-sm btn-outline-secondary"
                              style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                              onClick={() => {
                                // Clear verification to allow re-verification
                                setVerifications(prev => {
                                  const updated = { ...prev };
                                  delete updated[r.flow_id];
                                  return updated;
                                });
                                // Also notify backend that verification was cleared
                                fetch('http://localhost:5000/api/manual/verify', {
                                  method: 'POST',
                                  headers: {
                                    'Content-Type': 'application/json'
                                  },
                                  body: JSON.stringify({
                                    verifications: [{
                                      flow_id: r.flow_id,
                                      is_threat: null // Clear verification
                                    }]
                                  })
                                })
                                .catch(error => console.error('Error clearing verification:', error));
                              }}
                              title="Clear and change verification"
                            >
                              Change
                            </button>
                          </div>
                        );
                      }
                      
                      // Unverified - show action buttons clearly
                      return (
                        <div className="d-flex flex-column gap-2 align-items-center" style={{ padding: '0.75rem 0.5rem', width: '100%' }}>
                          {/* Status indicator */}
                          <div className="d-flex flex-column align-items-center gap-1" style={{ width: '100%' }}>
                            {statusBadge}
                            <small className="text-muted fw-bold" style={{ fontSize: '0.75rem', textAlign: 'center', width: '100%', display: 'block' }}>
                              Needs Review
                            </small>
                          </div>
                          {/* Action buttons - use outline style for unverified items */}
                          <div className="d-flex flex-column gap-1 w-100" style={{ maxWidth: '130px' }}>
                            <button
                              className="btn btn-outline-success"
                              style={{ padding: '0.6rem 0.75rem', fontSize: '0.9rem', fontWeight: '600', width: '100%', borderWidth: '2px' }}
                              onClick={() => handleManualVerification(r.flow_id, true)}
                              title="Click to confirm this is a real threat"
                            >
                              ✓ Confirm Threat
                            </button>
                            <button
                              className="btn btn-outline-danger"
                              style={{ padding: '0.6rem 0.75rem', fontSize: '0.9rem', fontWeight: '600', width: '100%', borderWidth: '2px' }}
                              onClick={() => handleManualVerification(r.flow_id, false)}
                              title="Click to mark as false positive"
                            >
                              ✗ False Positive
                            </button>
                          </div>
                        </div>
                      );
                    })()}
                  </td>
                  <td>
                    <div className="d-flex flex-column">
                      <div className="fw-bold text-danger">
                        🎯 {r.src_ip}
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="d-flex flex-column">
                      <div className="fw-bold text-primary">
                        🛡️ {r.dst_ip}
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="d-flex flex-wrap gap-1">
                      {
                        r.model_used?.split(',')
                          .map((model, modelIdx) => {
                            const modelLower = model.trim().toLowerCase();
                            let displayName = model.trim();
                            let badgeStyle = { backgroundColor: '#6c757d', color: 'white' };
                            
                            // Map model names and assign muted professional colors with white text
                            switch(modelLower) {
                              case 'fuzzy_rf':
                                displayName = 'Fuzzy Logic + Random Forest';
                                badgeStyle = { backgroundColor: '#0d9488', color: 'white' };
                                break;
                              case 'intrudtree':
                                displayName = 'IntruDTree Algorithm';
                                badgeStyle = { backgroundColor: '#84cc16', color: 'white' };
                                break;
                              case 'aca_svm':
                                displayName = 'Aho–Corasick + SVM';
                                badgeStyle = { backgroundColor: '#3b82f6', color: 'white' };
                                break;
                              default:
                                badgeStyle = { backgroundColor: '#6c757d', color: 'white' };
                            }
                            
                            return (
                              <Badge key={modelIdx} className="px-2 py-1" style={badgeStyle}>
                                {displayName}
                              </Badge>
                            );
                          })
                      }
                    </div>
                  </td>
                  <td>
                    <div className="d-flex flex-column align-items-start">
                      <Badge className={`attack-type ${r.attack_type?.toLowerCase()}`}>
                        {r.attack_type}
                      </Badge>
                      <small className="text-muted mt-1" style={{ whiteSpace: 'pre-line', lineHeight: '1.6' }}>
                        {(() => {
                          const severity = r.severity?.toLowerCase();
                          const contextualAnalysis = generateContextualAnalysis(r);
                          
                          if (severity === 'high') {
                            return '🔥 ' + contextualAnalysis;
                          } else if (severity === 'medium') {
                            return '⚠️ ' + contextualAnalysis;
                          } else {
                            return '✓ ' + contextualAnalysis;
                          }
                        })()}
                      </small>
                    </div>
                  </td>
                  <td>
                    <Badge className={`risk-level ${r.severity?.toLowerCase()}`}>
                      {r.severity === 'High' ? '🔴 High' : 
                       r.severity === 'Medium' ? '🟡 Medium' : '🟢 Low'}
                    </Badge>
                  </td>
                  <td>
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => handleRowExpand(r.flow_id)}
                    >
                      {expandedRows[r.flow_id] ? '▼ Hide' : '► Show'} Features
                    </Button>
                  </td>
                </tr>
                {expandedRows[r.flow_id] && r.key_features && (
                  <tr>
                    <td colSpan="7" className="bg-light">
                      <div className="p-3">
                        <h6 className="mb-2">🔍 Feature Data Used for Detection:</h6>
                        <div className="d-flex flex-wrap gap-2">
                          {Object.entries(r.key_features).map(([key, value]) => {
                            const explanation = getFeatureExplanation(key);
                            return (
                              <div 
                                key={key} 
                                className="badge bg-secondary px-2 py-1" 
                                title={`${explanation.description}\n\nWhy this matters:\n${explanation.whyItMatters}`}
                                style={{ cursor: 'help' }}
                              >
                                <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
                                <i className="fas fa-info-circle ms-1" style={{ fontSize: '0.7em', opacity: 0.8 }}></i>
                              </div>
                            );
                          })}
                        </div>
                      <small className="text-muted d-block mt-2">
                        💡 These are the top features the models analyzed to make their detection decision. Hover over features for detailed explanations.
                      </small>
                      
                      {/* Individual Model Predictions - Only show models that match the detected attack type */}
                      {r.models && Object.keys(r.models).length > 0 && (
                        <>
                          <hr className="my-3" />
                          <h6 className="mb-2">🤖 Individual Model Predictions:</h6>
                          <div className="d-flex flex-column gap-2">
                            {(() => {
                              // Show ALL model predictions, not just matching ones
                              const allModels = Object.entries(r.models || {});
                              
                              if (allModels.length === 0) {
                                return (
                                  <small className="text-muted d-block">
                                    <i className="fas fa-info-circle me-2"></i>
                                    No model prediction data available.
                                  </small>
                                );
                              }

                              // Get severity badge class and text
                              const getSeverityBadge = (severity) => {
                                const severityLower = severity?.toLowerCase() || 'low';
                                if (severityLower === 'high') {
                                  return { className: 'bg-danger', icon: '🔴', text: 'High Risk' };
                                } else if (severityLower === 'medium') {
                                  return { className: 'bg-warning', icon: '🟡', text: 'Medium Risk' };
                                } else {
                                  return { className: 'bg-success', icon: '🟢', text: 'Low Risk' };
                                }
                              };

                              return allModels.map(([modelName, modelData]) => {
                                // Map backend model names to display names
                                const displayName = modelName === 'Fuzzy_RF' ? 'Fuzzy Logic + Random Forest' :
                                                   modelName === 'IntruDTree' ? 'IntruDTree Algorithm' :
                                                   modelName === 'ACA_SVM' ? 'Aho–Corasick + SVM' : modelName;
                                
                                const isThreat = modelData.is_threat === 1;
                                const severityInfo = getSeverityBadge(modelData.severity);
                                
                                // Determine border and background color based on threat status
                                const borderClass = isThreat ? 'border-danger' : 'border-success';
                                const bgClass = isThreat ? 'bg-danger-subtle' : 'bg-success-subtle';
                                
                                return (
                                  <div 
                                    key={modelName} 
                                    className={`p-2 rounded border ${borderClass} ${bgClass}`}
                                  >
                                    <div className="d-flex justify-content-between align-items-center flex-wrap gap-2">
                                      <span className="fw-bold">{displayName}:</span>
                                      <div className="d-flex gap-2 align-items-center">
                                        {isThreat ? (
                                          <>
                                            <span className="text-danger fw-bold">⚠️ {modelData.attack_label || 'Threat'}</span>
                                            <span className={`badge ${severityInfo.className}`}>
                                              {severityInfo.icon} {severityInfo.text}
                                            </span>
                                          </>
                                        ) : (
                                          <>
                                            <span className="text-success fw-bold">✓ {modelData.attack_label || 'Benign'}</span>
                                            <span className="badge bg-success">🟢 Low Risk</span>
                                          </>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                );
                              });
                            })()}
                          </div>
                          <small className="text-muted d-block mt-2">
                            <i className="fas fa-lightbulb me-1"></i>
                            Individual predictions from all models. The final detection uses ensemble voting based on majority agreement.
                          </small>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              )}
              </React.Fragment>
            ))}
            {getFilteredRows().length === 0 && (
              <tr>
                <td colSpan="7" className="text-center py-4">
                  <em>No results match the selected filters</em>
                </td>
              </tr>
            )}
          </tbody>
        </Table>
      </div>
    </div>
  );
}