import React from 'react';

export default function SummaryPanel({ rows }) {
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
  
  const total = detectedByModels.length;
  const benign = detectedByModels.filter(r => r.attack_type && r.attack_type.toLowerCase().includes('benign')).length;
  const malicious = total - benign;

  const perModel = {};
  
  detectedByModels.forEach(r => {
    // Count models used for threat detection
    // ignore placeholder values like 'Unknown' or 'Simulated'
    const models = (r.model_used || '')
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)
      .filter(m => m.toLowerCase() !== 'unknown' && m.toLowerCase() !== 'simulated')
      .map(m => {
        switch(m.toLowerCase()) {
          case 'fuzzy_rf': return 'Fuzzy Logic + Random Forest';
          case 'intrudtree': return 'IntruDTree Algorithm';
          case 'aca_svm': return 'Aho–Corasick + SVM';
          default: return m;
        }
      });
    models.forEach(m => {
      perModel[m] = (perModel[m] || 0) + 1;
    });
  });

  return (
    <div style={{padding: '12px', border:'1px solid #ddd', borderRadius:6, marginBottom:12}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div>
          <strong>Total detected:</strong> {total}
          <div style={{fontSize:12, color:'#555'}}>Benign: {benign} · Malicious: {malicious}</div>
        </div>
        <div>
          <strong>Per-model counts</strong>
          <div style={{fontSize:11, color:'#333'}}>
            {Object.keys(perModel).length === 0 ? <span>No detections yet</span> : Object.entries(perModel).map(([k,v]) => (
               <div key={k}>{k}: {v}</div>
             ))}
          </div>
        </div>
      </div>
    </div>
  );
}