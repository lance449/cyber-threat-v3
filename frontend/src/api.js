// use env var if present, fallback to localhost:5000
const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

async function fetchJson(path, opts) {
  const res = await fetch(`${API_BASE}${path}`, opts);
  const text = await res.text();
  try {
    const data = text ? JSON.parse(text) : {};
    return res.ok ? data : { success: false, status: res.status, error: data || text };
  } catch (e) {
    // not JSON
    return { success: false, status: res.status, error: text };
  }
}

export async function resetStream() {
  return fetchJson('/api/stream/reset', { method: 'POST' });
}

export async function getNextRow() {
  return fetchJson('/api/stream/next', { method: 'GET' });
}

export async function getStreamState() {
  return fetchJson('/api/stream/state', { method: 'GET' });
}

export async function getManualMetrics() {
  try {
    const response = await fetch(`${API_BASE}/api/manual/metrics`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching manual metrics:', error);
    return { success: false, error: error.message };
  }
}

export async function getModelPerformance() {
  try {
    const response = await fetch(`${API_BASE}/api/models/performance`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching model performance:', error);
    return { success: false, error: error.message };
  }
}

export async function getRiskAnalysisReport() {
  try {
    const response = await fetch(`${API_BASE}/api/reports/risk-analysis`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching risk analysis report:', error);
    return { success: false, error: error.message };
  }
}

export async function getAttackFrequencyReport() {
  try {
    const response = await fetch(`${API_BASE}/api/reports/attack-frequency`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching attack frequency report:', error);
    return { success: false, error: error.message };
  }
}

export async function exportReport(reportType, format) {
  try {
    const response = await fetch(`${API_BASE}/api/reports/export/${reportType}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ format })
    });
    return await response.json();
  } catch (error) {
    console.error('Error exporting report:', error);
    return { success: false, error: error.message };
  }
}

export async function debugRecentThreats() {
  try {
    const response = await fetch(`${API_BASE}/api/debug/recent-threats`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching debug data:', error);
    return { success: false, error: error.message };
  }
}

export async function clearReportData() {
  try {
    const response = await fetch(`${API_BASE}/api/reports/clear-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    return await response.json();
  } catch (error) {
    console.error('Error clearing report data:', error);
    return { success: false, error: error.message };
  }
}

export async function getReportStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/reports/status`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching report status:', error);
    return { success: false, error: error.message };
  }
}