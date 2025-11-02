"""
Threat Severity & Risk Analysis Module
Calculates risk scores and provides actionable insights for cybersecurity professionals
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ThreatSeverityAnalyzer:
    """
    Analyzes threat severity based on predicted labels and threat behavior
    """
    
    def __init__(self):
        """Initialize the threat severity analyzer"""
        self.threat_severity_mapping = self._define_threat_severity_mapping()
        self.behavior_indicators = self._define_behavior_indicators()
        
    def _define_threat_severity_mapping(self) -> Dict[str, str]:
        """
        Define mapping of threat types to severity levels
        
        Returns:
            Dictionary mapping threat types to severity levels
        """
        severity_mapping = {
            # High severity threats
            'ddos': 'High',
            'dos': 'High',
            'ransomware': 'High',
            'trojan': 'High',
            'backdoor': 'High',
            'rootkit': 'High',
            'data_exfiltration': 'High',
            'privilege_escalation': 'High',
            'lateral_movement': 'High',
            'command_control': 'High',
            'botnet': 'High',
            
            # Medium severity threats
            'virus': 'Medium',
            'worm': 'Medium',
            'spyware': 'Medium',
            'adware': 'Medium',
            'keylogger': 'Medium',
            'phishing': 'Medium',
            'sql_injection': 'Medium',
            'xss': 'Medium',
            'csrf': 'Medium',
            'port_scan': 'Medium',
            'vulnerability_scan': 'Medium',
            'network_discovery': 'Medium',
            
            # Low severity threats
            'scan': 'Low',
            'probe': 'Low',
            'reconnaissance': 'Low',
            'normal': 'Low',
            'benign': 'Low'
        }
        
        return severity_mapping
    
    def _define_behavior_indicators(self) -> Dict[str, Dict]:
        """
        Define behavior indicators for threat analysis
        
        Returns:
            Dictionary of behavior indicators
        """
        indicators = {
            'network_behavior': {
                'high_packet_rate': {'weight': 0.3, 'severity': 'High'},
                'high_byte_rate': {'weight': 0.25, 'severity': 'High'},
                'unusual_port_activity': {'weight': 0.2, 'severity': 'Medium'},
                'protocol_anomaly': {'weight': 0.15, 'severity': 'Medium'},
                'encrypted_traffic': {'weight': 0.1, 'severity': 'Low'}
            },
            'flow_characteristics': {
                'short_flow_duration': {'weight': 0.2, 'severity': 'Medium'},
                'long_flow_duration': {'weight': 0.15, 'severity': 'Low'},
                'large_packet_size': {'weight': 0.25, 'severity': 'Medium'},
                'small_packet_size': {'weight': 0.1, 'severity': 'Low'},
                'unusual_iat': {'weight': 0.3, 'severity': 'High'}
            },
            'flag_patterns': {
                'high_flag_count': {'weight': 0.4, 'severity': 'High'},
                'syn_flood': {'weight': 0.3, 'severity': 'High'},
                'fin_scan': {'weight': 0.2, 'severity': 'Medium'},
                'null_scan': {'weight': 0.1, 'severity': 'Low'}
            }
        }
        
        return indicators
    
    def assign_severity(self, threat_type: str, behavior_indicators: Dict = None) -> str:
        """
        Assign severity level to a threat
        
        Args:
            threat_type: Type of threat detected
            behavior_indicators: Additional behavior indicators
            
        Returns:
            Severity level (Low, Medium, High)
        """
        # Get base severity from threat type
        base_severity = self.threat_severity_mapping.get(threat_type.lower(), 'Medium')
        
        # Adjust severity based on behavior indicators
        if behavior_indicators:
            severity_score = self._calculate_behavior_severity(behavior_indicators)
            
            # Combine base severity with behavior severity
            if severity_score > 0.7:
                return 'High'
            elif severity_score > 0.4:
                return 'Medium'
            else:
                return 'Low'
        
        return base_severity
    
    def _calculate_behavior_severity(self, behavior_indicators: Dict) -> float:
        """
        Calculate severity score based on behavior indicators
        
        Args:
            behavior_indicators: Dictionary of behavior indicators
            
        Returns:
            Severity score (0-1)
        """
        total_weight = 0
        weighted_score = 0
        
        for category, indicators in self.behavior_indicators.items():
            if category in behavior_indicators:
                for indicator, value in behavior_indicators[category].items():
                    if indicator in indicators:
                        weight = indicators[indicator]['weight']
                        severity = indicators[indicator]['severity']
                        
                        # Convert severity to score
                        severity_score = {'Low': 0.3, 'Medium': 0.6, 'High': 0.9}[severity]
                        
                        weighted_score += weight * severity_score * value
                        total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        
        return 0.0

class RiskAnalyzer:
    """
    Analyzes risk factors and calculates overall risk scores
    """
    
    def __init__(self):
        """Initialize the risk analyzer"""
        self.risk_factors = self._define_risk_factors()
        self.impact_levels = self._define_impact_levels()
        
    def _define_risk_factors(self) -> Dict[str, Dict]:
        """
        Define risk factors and their weights
        
        Returns:
            Dictionary of risk factors
        """
        factors = {
            'severity': {
                'weight': 0.4,
                'levels': {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
            },
            'attack_type': {
                'weight': 0.25,
                'types': {
                    'ddos': 0.9, 'dos': 0.8, 'ransomware': 0.95,
                    'trojan': 0.7, 'virus': 0.6, 'worm': 0.65,
                    'spyware': 0.5, 'phishing': 0.4, 'scan': 0.3
                }
            },
            'source_destination': {
                'weight': 0.2,
                'external_source': 0.7,
                'internal_source': 0.3,
                'critical_destination': 0.8,
                'normal_destination': 0.2
            },
            'protocol': {
                'weight': 0.15,
                'protocols': {
                    'tcp': 0.4, 'udp': 0.6, 'icmp': 0.8,
                    'http': 0.3, 'https': 0.2, 'ftp': 0.5,
                    'ssh': 0.6, 'telnet': 0.7, 'smtp': 0.4
                }
            }
        }
        
        return factors
    
    def _define_impact_levels(self) -> Dict[str, Dict]:
        """
        Define impact levels for different types of attacks
        
        Returns:
            Dictionary of impact levels
        """
        impacts = {
            'operational': {
                'ddos': 'Critical',
                'dos': 'High',
                'ransomware': 'Critical',
                'trojan': 'Medium',
                'virus': 'Medium',
                'worm': 'High',
                'spyware': 'Low',
                'phishing': 'Medium',
                'scan': 'Low'
            },
            'data': {
                'ddos': 'Low',
                'dos': 'Low',
                'ransomware': 'Critical',
                'trojan': 'High',
                'virus': 'Medium',
                'worm': 'Medium',
                'spyware': 'High',
                'phishing': 'High',
                'scan': 'Low'
            },
            'financial': {
                'ddos': 'High',
                'dos': 'Medium',
                'ransomware': 'Critical',
                'trojan': 'Medium',
                'virus': 'Low',
                'worm': 'Medium',
                'spyware': 'Low',
                'phishing': 'High',
                'scan': 'Low'
            },
            'reputation': {
                'ddos': 'Medium',
                'dos': 'Low',
                'ransomware': 'Critical',
                'trojan': 'Medium',
                'virus': 'Low',
                'worm': 'Medium',
                'spyware': 'High',
                'phishing': 'High',
                'scan': 'Low'
            }
        }
        
        return impacts
    
    def calculate_risk_score(self, threat_info: Dict) -> Dict:
        """
        Calculate comprehensive risk score
        
        Args:
            threat_info: Dictionary containing threat information
            
        Returns:
            Dictionary with risk analysis results
        """
        risk_score = 0.0
        factor_scores = {}
        
        # Calculate severity factor
        severity = threat_info.get('severity', 'Medium')
        severity_score = self.risk_factors['severity']['levels'].get(severity, 0.5)
        factor_scores['severity'] = severity_score
        risk_score += severity_score * self.risk_factors['severity']['weight']
        
        # Calculate attack type factor
        attack_type = threat_info.get('attack_type', 'unknown')
        attack_score = self.risk_factors['attack_type']['types'].get(attack_type.lower(), 0.5)
        factor_scores['attack_type'] = attack_score
        risk_score += attack_score * self.risk_factors['attack_type']['weight']
        
        # Calculate source/destination factor
        source_dest_score = 0.0
        if threat_info.get('external_source', False):
            source_dest_score += self.risk_factors['source_destination']['external_source']
        if threat_info.get('critical_destination', False):
            source_dest_score += self.risk_factors['source_destination']['critical_destination']
        source_dest_score = min(source_dest_score, 1.0)
        factor_scores['source_destination'] = source_dest_score
        risk_score += source_dest_score * self.risk_factors['source_destination']['weight']
        
        # Calculate protocol factor
        protocol = threat_info.get('protocol', 'tcp')
        protocol_score = self.risk_factors['protocol']['protocols'].get(protocol.lower(), 0.4)
        factor_scores['protocol'] = protocol_score
        risk_score += protocol_score * self.risk_factors['protocol']['weight']
        
        # Calculate impact levels
        impact_analysis = self._calculate_impact_levels(attack_type)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'factor_scores': factor_scores,
            'impact_analysis': impact_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_impact_levels(self, attack_type: str) -> Dict:
        """
        Calculate impact levels for different categories
        
        Args:
            attack_type: Type of attack
            
        Returns:
            Dictionary of impact levels
        """
        impact_analysis = {}
        
        for category, impacts in self.impact_levels.items():
            impact_level = impacts.get(attack_type.lower(), 'Low')
            impact_analysis[category] = impact_level
        
        return impact_analysis
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on risk score
        
        Args:
            risk_score: Calculated risk score (0-1)
            
        Returns:
            Risk level (Low, Medium, High, Critical)
        """
        if risk_score >= 0.8:
            return 'Critical'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        else:
            return 'Low'

class ActionRecommender:
    """
    Recommends actions based on threat analysis
    """
    
    def __init__(self):
        """Initialize the action recommender"""
        self.action_templates = self._define_action_templates()
        
    def _define_action_templates(self) -> Dict[str, Dict]:
        """
        Define action templates for different threat types and risk levels
        
        Returns:
            Dictionary of action templates
        """
        templates = {
            'Critical': {
                'immediate': [
                    'Isolate affected systems immediately',
                    'Disconnect from network if necessary',
                    'Activate incident response team',
                    'Notify senior management',
                    'Begin forensic analysis'
                ],
                'short_term': [
                    'Implement emergency patches',
                    'Update security controls',
                    'Monitor for lateral movement',
                    'Review access logs',
                    'Assess data compromise'
                ],
                'long_term': [
                    'Conduct post-incident review',
                    'Update security policies',
                    'Enhance monitoring capabilities',
                    'Provide staff training',
                    'Implement additional controls'
                ]
            },
            'High': {
                'immediate': [
                    'Monitor affected systems closely',
                    'Implement additional logging',
                    'Review recent activities',
                    'Update threat intelligence'
                ],
                'short_term': [
                    'Apply security patches',
                    'Review access controls',
                    'Enhance monitoring',
                    'Update security policies'
                ],
                'long_term': [
                    'Conduct security assessment',
                    'Improve detection capabilities',
                    'Update incident response procedures'
                ]
            },
            'Medium': {
                'immediate': [
                    'Log the incident',
                    'Monitor for escalation',
                    'Review system logs'
                ],
                'short_term': [
                    'Apply relevant patches',
                    'Review security controls',
                    'Update monitoring rules'
                ],
                'long_term': [
                    'Conduct periodic reviews',
                    'Update security awareness training'
                ]
            },
            'Low': {
                'immediate': [
                    'Log the incident',
                    'Continue monitoring'
                ],
                'short_term': [
                    'Review if patterns emerge',
                    'Update threat intelligence'
                ],
                'long_term': [
                    'Periodic security reviews'
                ]
            }
        }
        
        return templates
    
    def recommend_actions(self, risk_analysis: Dict) -> Dict:
        """
        Recommend actions based on risk analysis
        
        Args:
            risk_analysis: Risk analysis results
            
        Returns:
            Dictionary with recommended actions
        """
        risk_level = risk_analysis['risk_level']
        attack_type = risk_analysis.get('attack_type', 'unknown')
        
        # Get base actions for risk level
        actions = self.action_templates.get(risk_level, self.action_templates['Low']).copy()
        
        # Add specific actions based on attack type
        specific_actions = self._get_attack_specific_actions(attack_type)
        
        # Combine actions
        for timeframe in actions:
            if timeframe in specific_actions:
                actions[timeframe].extend(specific_actions[timeframe])
        
        return {
            'risk_level': risk_level,
            'recommended_actions': actions,
            'priority': self._get_priority(risk_level),
            'estimated_response_time': self._get_estimated_response_time(risk_level)
        }
    
    def _get_attack_specific_actions(self, attack_type: str) -> Dict:
        """
        Get attack-specific actions
        
        Args:
            attack_type: Type of attack
            
        Returns:
            Dictionary of attack-specific actions
        """
        specific_actions = {
            'ddos': {
                'immediate': [
                    'Activate DDoS protection services',
                    'Implement rate limiting',
                    'Contact ISP for traffic filtering'
                ]
            },
            'ransomware': {
                'immediate': [
                    'Disconnect infected systems',
                    'Identify encryption scope',
                    'Check for backup availability'
                ]
            },
            'trojan': {
                'immediate': [
                    'Isolate infected systems',
                    'Identify command and control servers',
                    'Block malicious IPs'
                ]
            },
            'phishing': {
                'immediate': [
                    'Remove malicious emails',
                    'Block sender domains',
                    'Notify affected users'
                ]
            }
        }
        
        return specific_actions.get(attack_type.lower(), {})
    
    def _get_priority(self, risk_level: str) -> str:
        """Get priority level based on risk level"""
        priority_mapping = {
            'Critical': 'Immediate',
            'High': 'High',
            'Medium': 'Medium',
            'Low': 'Low'
        }
        return priority_mapping.get(risk_level, 'Low')
    
    def _get_estimated_response_time(self, risk_level: str) -> str:
        """Get estimated response time based on risk level"""
        time_mapping = {
            'Critical': 'Immediate (0-1 hour)',
            'High': 'High (1-4 hours)',
            'Medium': 'Medium (4-24 hours)',
            'Low': 'Low (24-72 hours)'
        }
        return time_mapping.get(risk_level, 'Low (24-72 hours)')

class ThreatAnalyzer:
    """
    Main threat analyzer that combines severity, risk, and action analysis
    """
    
    def __init__(self):
        """Initialize the threat analyzer"""
        self.severity_analyzer = ThreatSeverityAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.action_recommender = ActionRecommender()
        
    def analyze_threat(self, detection_result: Dict, flow_data: pd.DataFrame = None) -> Dict:
        """
        Perform comprehensive threat analysis
        
        Args:
            detection_result: Detection result from models
            flow_data: Original flow data for additional context
            
        Returns:
            Comprehensive threat analysis
        """
        # Extract threat information
        threat_info = self._extract_threat_info(detection_result, flow_data)
        
        # Assign severity
        severity = self.severity_analyzer.assign_severity(
            threat_info.get('attack_type', 'unknown'),
            threat_info.get('behavior_indicators', {})
        )
        threat_info['severity'] = severity
        
        # Calculate risk score
        risk_analysis = self.risk_analyzer.calculate_risk_score(threat_info)
        
        # Recommend actions
        action_recommendations = self.action_recommender.recommend_actions(risk_analysis)
        
        # Combine all analysis results
        analysis_result = {
            'threat_info': threat_info,
            'severity': severity,
            'risk_analysis': risk_analysis,
            'action_recommendations': action_recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis_result
    
    def _extract_threat_info(self, detection_result: Dict, flow_data: pd.DataFrame = None) -> Dict:
        """
        Extract threat information from detection result
        
        Args:
            detection_result: Detection result from models
            flow_data: Original flow data
            
        Returns:
            Dictionary of threat information
        """
        threat_info = {
            'attack_type': 'unknown',
            'external_source': False,
            'critical_destination': False,
            'protocol': 'tcp',
            'behavior_indicators': {}
        }
        
        # Extract attack type from model predictions
        if 'model_predictions' in detection_result:
            predictions = detection_result['model_predictions']
            # Determine most likely attack type based on model confidence
            # This is a simplified approach - in practice, you'd have more sophisticated logic
            if any(pred > 0.8 for pred in predictions.values()):
                threat_info['attack_type'] = 'malware'
            elif any(pred > 0.6 for pred in predictions.values()):
                threat_info['attack_type'] = 'scan'
            else:
                threat_info['attack_type'] = 'normal'
        
        # Extract additional information from flow data if available
        if flow_data is not None and len(flow_data) > 0:
            # Analyze source/destination
            if 'Src IP' in flow_data.columns:
                # Check if source is external (simplified)
                threat_info['external_source'] = True  # Simplified logic
            
            # Analyze protocol
            if 'Protocol' in flow_data.columns:
                threat_info['protocol'] = str(flow_data['Protocol'].iloc[0]).lower()
            
            # Extract behavior indicators
            threat_info['behavior_indicators'] = self._extract_behavior_indicators(flow_data)
        
        return threat_info
    
    def _extract_behavior_indicators(self, flow_data: pd.DataFrame) -> Dict:
        """
        Extract behavior indicators from flow data
        
        Args:
            flow_data: Network flow data
            
        Returns:
            Dictionary of behavior indicators
        """
        indicators = {
            'network_behavior': {},
            'flow_characteristics': {},
            'flag_patterns': {}
        }
        
        if len(flow_data) == 0:
            return indicators
        
        # Network behavior indicators
        if 'Flow Bytes/s' in flow_data.columns:
            high_byte_rate = flow_data['Flow Bytes/s'].mean() > 1000000
            indicators['network_behavior']['high_byte_rate'] = high_byte_rate
        
        if 'Flow Packets/s' in flow_data.columns:
            high_packet_rate = flow_data['Flow Packets/s'].mean() > 1000
            indicators['network_behavior']['high_packet_rate'] = high_packet_rate
        
        # Flow characteristics
        if 'Flow Duration' in flow_data.columns:
            short_duration = flow_data['Flow Duration'].mean() < 1.0
            indicators['flow_characteristics']['short_flow_duration'] = short_duration
        
        if 'Packet Length Mean' in flow_data.columns:
            large_packets = flow_data['Packet Length Mean'].mean() > 1400
            indicators['flow_characteristics']['large_packet_size'] = large_packets
        
        # Flag patterns
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
                    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
        present_flags = [col for col in flag_cols if col in flow_data.columns]
        
        if present_flags:
            total_flags = flow_data[present_flags].sum(axis=1).mean()
            high_flags = total_flags > 5
            indicators['flag_patterns']['high_flag_count'] = high_flags
        
        return indicators
