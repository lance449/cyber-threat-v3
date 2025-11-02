# backend/src/fuzzy_logic.py

import pandas as pd
import numpy as np

def apply_fuzzy_logic(df):

    
    # Initialize score array
    scores = np.zeros(len(df))
    
    # Rule 1: Flow Duration Analysis
    scores += np.where(df['Flow Duration'] > df['Flow Duration'].mean() + 2*df['Flow Duration'].std(), 2, 
                      np.where(df['Flow Duration'] > df['Flow Duration'].mean() + df['Flow Duration'].std(), 1, 0))
    
    # Rule 2: Packet Size Analysis
    scores += np.where(df['Packet Length Mean'] > df['Packet Length Mean'].mean() + 2*df['Packet Length Mean'].std(), 2,
                      np.where(df['Packet Length Mean'] > df['Packet Length Mean'].mean() + df['Packet Length Mean'].std(), 1, 0))
    
    # Rule 3: Flow Bytes Rate Analysis
    scores += np.where(df['Flow Bytes/s'] > df['Flow Bytes/s'].mean() + 2*df['Flow Bytes/s'].std(), 2,
                      np.where(df['Flow Bytes/s'] > df['Flow Bytes/s'].mean() + df['Flow Bytes/s'].std(), 1, 0))
    
    # Rule 4: Flag Analysis
    flag_columns = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
    flag_scores = df[flag_columns].sum(axis=1)
    scores += np.where(flag_scores > flag_scores.mean() + flag_scores.std(), 1, 0)
    
    # Rule 5: IAT (Inter Arrival Time) Analysis
    iat_columns = ['Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean']
    iat_scores = df[iat_columns].mean(axis=1)
    scores += np.where(iat_scores < iat_scores.mean() - iat_scores.std(), 1, 0)
    
    # Normalize scores to 0-1 range
    df['Fuzzy_Score'] = (scores - scores.min()) / (scores.max() - scores.min())
    
    return df

class FuzzyLogicDetector:
    
    def __init__(self):
        """Initialize the fuzzy logic detector"""
        self.rules = []
        
    def predict(self, features):
        """Predict threat level using fuzzy logic"""
        try:
            # Convert features to DataFrame if needed
            if isinstance(features, np.ndarray):
                # Create a simple DataFrame with available features
                feature_names = [
                    'packet_count', 'byte_count', 'duration', 'avg_ttl', 'avg_tos',
                    'avg_window_size', 'unique_flags', 'packets_per_second', 
                    'bytes_per_second', 'avg_packet_size', 'protocol_encoded',
                    'is_well_known_port', 'is_privileged_port', 'src_ip_private',
                    'dst_ip_private', 'flow_direction'
                ]
                
                # Ensure we have the right number of features
                if features.shape[1] != len(feature_names):
                    # Pad or truncate as needed
                    if features.shape[1] < len(feature_names):
                        padding = np.zeros((features.shape[0], len(feature_names) - features.shape[1]))
                        features = np.hstack([features, padding])
                    else:
                        features = features[:, :len(feature_names)]
                
                df = pd.DataFrame(features, columns=feature_names)
            else:
                df = features
                
            # Apply fuzzy logic rules
            scores = self._apply_fuzzy_rules(df)
            
            # Convert scores to binary predictions (threshold-based)
            predictions = (scores > 0.5).astype(int)
            
            return predictions
            
        except Exception as e:
            print(f"Error in fuzzy logic prediction: {e}")
            # Return zeros if prediction fails
            return np.zeros(features.shape[0] if hasattr(features, 'shape') else 1, dtype=int)
    
    def _apply_fuzzy_rules(self, df):
        """Apply optimized fuzzy logic rules to the data"""
        scores = np.zeros(len(df))
        
        # Pre-calculate quantiles for efficiency
        quantiles = {}
        for col in ['packets_per_second', 'bytes_per_second', 'packet_count']:
            if col in df.columns:
                quantiles[col] = df[col].quantile([0.8, 0.9])
        
        # Rule 1: High packet rate (potential DDoS) - Weight: 3
        if 'packets_per_second' in df.columns:
            high_rate = df['packets_per_second'] > quantiles['packets_per_second'][0.9]
            scores += high_rate.astype(int) * 3
        
        # Rule 2: High byte rate - Weight: 2
        if 'bytes_per_second' in df.columns:
            high_bytes = df['bytes_per_second'] > quantiles['bytes_per_second'][0.9]
            scores += high_bytes.astype(int) * 2
        
        # Rule 3: Unusual port activity - Weight: 2
        if 'is_well_known_port' in df.columns:
            unusual_ports = df['is_well_known_port'] == 0
            scores += unusual_ports.astype(int) * 2
        
        # Rule 4: External to internal communication - Weight: 2
        if 'flow_direction' in df.columns:
            external_internal = df['flow_direction'] == 2
            scores += external_internal.astype(int) * 2
        
        # Rule 5: High packet count - Weight: 1
        if 'packet_count' in df.columns:
            high_packets = df['packet_count'] > quantiles['packet_count'][0.8]
            scores += high_packets.astype(int) * 1
        
        # Rule 6: Protocol anomalies - Weight: 2
        if 'protocol_encoded' in df.columns:
            unusual_protocols = df['protocol_encoded'] > df['protocol_encoded'].quantile(0.95)
            scores += unusual_protocols.astype(int) * 2
        
        # Rule 7: TTL anomalies - Weight: 1
        if 'avg_ttl' in df.columns:
            ttl_anomaly = (df['avg_ttl'] < df['avg_ttl'].quantile(0.1)) | (df['avg_ttl'] > df['avg_ttl'].quantile(0.9))
            scores += ttl_anomaly.astype(int) * 1
        
        # Normalize scores to 0-1 range with improved scaling
        max_possible_score = 13  # Sum of all weights
        scores = np.clip(scores / max_possible_score, 0, 1)
        
        return scores
