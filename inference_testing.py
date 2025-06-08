"""
Inference Testing Mode for Behavioral Response Application
Validates model decision quality across simulated scenarios
"""
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from advanced_nlp import BehaviorQueryProcessor
from database_service import protocol_service, behavior_service
from crisis_alert import CrisisAlertSystem

logger = logging.getLogger(__name__)

class InferenceTestRunner:
    """Runs inference testing on behavioral scenarios to validate model accuracy"""
    
    def __init__(self, test_scenarios_file="test_scenarios.json"):
        self.test_scenarios_file = test_scenarios_file
        self.nlp_processor = BehaviorQueryProcessor()
        self.crisis_alert = CrisisAlertSystem()
        self.test_results = []
        logger.info("Inference test runner initialized")
    
    def load_test_scenarios(self) -> List[Dict[str, Any]]:
        """Load test scenarios from JSON file"""
        try:
            with open(self.test_scenarios_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get('test_scenarios', [])
        except FileNotFoundError:
            logger.error(f"Test scenarios file not found: {self.test_scenarios_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing test scenarios JSON: {e}")
            return []
    
    def normalize_behavior_name(self, behavior: str) -> str:
        """Normalize behavior names for comparison"""
        behavior_mapping = {
            'verbal_aggression': ['verbal aggression', 'yelling', 'threatening', 'cursing'],
            'physical_aggression': ['physical aggression', 'hitting', 'fighting', 'attacking'],
            'defiance': ['defiance', 'noncompliance', 'refusing', 'oppositional'],
            'disruption': ['disruption', 'talking out', 'interrupting', 'noise'],
            'self_harm': ['self harm', 'self-harm', 'self injury', 'banging head'],
            'property_destruction': ['property destruction', 'throwing', 'breaking', 'tearing'],
            'elopement': ['elopement', 'running', 'leaving', 'escaping'],
            'emotional_outburst': ['emotional outburst', 'crying', 'meltdown', 'upset']
        }
        
        behavior_lower = behavior.lower().replace('_', ' ')
        
        for standard_name, variations in behavior_mapping.items():
            if behavior_lower in variations or any(var in behavior_lower for var in variations):
                return standard_name
        
        return behavior_lower.replace(' ', '_')
    
    def normalize_severity(self, severity: str) -> str:
        """Normalize severity levels for comparison"""
        severity_mapping = {
            'low': ['low', 'mild', 'minor'],
            'medium': ['medium', 'moderate', 'med'],
            'high': ['high', 'major', 'significant'],
            'severe': ['severe', 'serious', 'extreme'],
            'critical': ['critical', 'emergency', 'immediate']
        }
        
        severity_lower = severity.lower()
        
        for standard_level, variations in severity_mapping.items():
            if severity_lower in variations:
                return standard_level
        
        return severity_lower
    
    def extract_behaviors_from_analysis(self, nlp_analysis: Dict[str, Any]) -> List[str]:
        """Extract behavior types from NLP analysis"""
        behaviors = []
        
        # Primary behavior type
        if nlp_analysis.get('behavior_type'):
            behaviors.append(self.normalize_behavior_name(nlp_analysis['behavior_type']))
        
        # Extract from keywords
        keywords = nlp_analysis.get('keywords', [])
        for keyword in keywords:
            normalized = self.normalize_behavior_name(keyword)
            if normalized not in behaviors:
                behaviors.append(normalized)
        
        # Check emergency signals for additional behaviors
        emergency_signals = nlp_analysis.get('emergency_signals', [])
        for signal in emergency_signals:
            if 'weapon' in signal.lower():
                behaviors.append('weapon_threat')
            elif 'threat' in signal.lower():
                behaviors.append('threat_behavior')
        
        return behaviors if behaviors else ['unclear']
    
    def get_recommendation_from_analysis(self, nlp_analysis: Dict[str, Any]) -> str:
        """Get protocol recommendation from NLP analysis"""
        behavior_type = nlp_analysis.get('behavior_type')
        severity = nlp_analysis.get('severity', 'medium')
        is_emergency = nlp_analysis.get('is_emergency', False)
        
        if is_emergency:
            return "Emergency Response Protocol"
        
        if behavior_type:
            protocol = protocol_service.get_protocol_for_behavior(behavior_type, severity)
            if protocol:
                return protocol.name
        
        # Fallback based on severity
        severity_protocols = {
            'critical': 'Safety Protocol',
            'severe': 'Crisis Response Protocol',
            'high': 'De-escalation Protocol',
            'medium': 'Redirection Protocol',
            'low': 'Engagement Protocol'
        }
        
        return severity_protocols.get(severity, 'Assessment Protocol')
    
    def compare_behaviors(self, expected: List[str], actual: List[str]) -> Dict[str, Any]:
        """Compare expected vs actual behavior identification"""
        expected_normalized = [self.normalize_behavior_name(b) for b in expected]
        actual_normalized = [self.normalize_behavior_name(b) for b in actual]
        
        # Calculate overlap
        expected_set = set(expected_normalized)
        actual_set = set(actual_normalized)
        
        intersection = expected_set.intersection(actual_set)
        union = expected_set.union(actual_set)
        
        # Calculate precision, recall, and F1
        precision = len(intersection) / len(actual_set) if actual_set else 0
        recall = len(intersection) / len(expected_set) if expected_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'match': expected_set == actual_set,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'expected': expected_normalized,
            'actual': actual_normalized,
            'missed': list(expected_set - actual_set),
            'extra': list(actual_set - expected_set)
        }
    
    def run_single_test(self, scenario: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """Run inference test on a single scenario"""
        test_id = scenario['test_id']
        voice_transcript = scenario['voice_transcript']
        
        logger.info(f"Running test {test_id}: {scenario.get('description', '')}")
        
        try:
            # Run NLP analysis
            nlp_analysis = self.nlp_processor.process_teacher_query(voice_transcript)
            
            # Extract results
            actual_behaviors = self.extract_behaviors_from_analysis(nlp_analysis)
            actual_severity = self.normalize_severity(nlp_analysis.get('severity', 'medium'))
            actual_recommendation = self.get_recommendation_from_analysis(nlp_analysis)
            
            # Compare with expected results
            expected_behaviors = scenario['expected_behavior']
            expected_severity = self.normalize_severity(scenario['expected_severity'])
            expected_recommendation = scenario['expected_recommendation']
            
            behavior_comparison = self.compare_behaviors(expected_behaviors, actual_behaviors)
            severity_match = actual_severity == expected_severity
            
            # Check recommendation match (partial matching for protocol names)
            recommendation_match = (
                expected_recommendation.lower() in actual_recommendation.lower() or
                actual_recommendation.lower() in expected_recommendation.lower()
            )
            
            # Overall pass/fail
            overall_pass = (
                behavior_comparison['f1_score'] >= 0.7 and
                severity_match and
                recommendation_match
            )
            
            # Check for clarification requirements
            requires_clarification = scenario.get('requires_clarification', False)
            clarification_triggered = nlp_analysis.get('needs_clarification', False)
            clarification_correct = requires_clarification == clarification_triggered
            
            # Check emergency detection
            is_emergency_scenario = scenario.get('is_emergency', False)
            emergency_detected = nlp_analysis.get('is_emergency', False)
            emergency_correct = is_emergency_scenario == emergency_detected
            
            test_result = {
                'test_id': test_id,
                'description': scenario.get('description', ''),
                'voice_transcript': voice_transcript,
                'overall_pass': overall_pass,
                'behavior_analysis': behavior_comparison,
                'severity_match': severity_match,
                'recommendation_match': recommendation_match,
                'clarification_correct': clarification_correct,
                'emergency_correct': emergency_correct,
                'expected': {
                    'behaviors': expected_behaviors,
                    'severity': expected_severity,
                    'recommendation': expected_recommendation,
                    'requires_clarification': requires_clarification,
                    'is_emergency': is_emergency_scenario
                },
                'actual': {
                    'behaviors': actual_behaviors,
                    'severity': actual_severity,
                    'recommendation': actual_recommendation,
                    'requires_clarification': clarification_triggered,
                    'is_emergency': emergency_detected
                },
                'confidence_scores': nlp_analysis.get('confidence', {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if debug:
                test_result['debug_info'] = {
                    'nlp_analysis': nlp_analysis,
                    'keywords': nlp_analysis.get('keywords', []),
                    'emergency_signals': nlp_analysis.get('emergency_signals', [])
                }
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error running test {test_id}: {e}")
            return {
                'test_id': test_id,
                'description': scenario.get('description', ''),
                'voice_transcript': voice_transcript,
                'overall_pass': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def run_inference_tests(self, debug: bool = False, test_filter: Optional[str] = None) -> Dict[str, Any]:
        """Run all inference tests and return summary results"""
        logger.info("Starting inference testing mode")
        
        scenarios = self.load_test_scenarios()
        if not scenarios:
            return {
                'success': False,
                'error': 'No test scenarios loaded',
                'total_tests': 0
            }
        
        # Filter scenarios if specified
        if test_filter:
            scenarios = [s for s in scenarios if test_filter.lower() in s['test_id'].lower() or 
                        test_filter.lower() in s.get('description', '').lower()]
        
        total_tests = len(scenarios)
        passed_tests = 0
        failed_tests = []
        
        logger.info(f"Running {total_tests} inference tests")
        
        # Run each test
        for scenario in scenarios:
            result = self.run_single_test(scenario, debug)
            self.test_results.append(result)
            
            if result.get('overall_pass', False):
                passed_tests += 1
            else:
                failed_tests.append(result)
        
        # Calculate detailed metrics
        behavior_f1_scores = [r.get('behavior_analysis', {}).get('f1_score', 0) 
                             for r in self.test_results if 'behavior_analysis' in r]
        
        avg_behavior_f1 = sum(behavior_f1_scores) / len(behavior_f1_scores) if behavior_f1_scores else 0
        
        severity_matches = sum(1 for r in self.test_results if r.get('severity_match', False))
        recommendation_matches = sum(1 for r in self.test_results if r.get('recommendation_match', False))
        
        # Save results to CSV
        self.save_results_to_csv()
        
        summary = {
            'success': True,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': len(failed_tests),
            'accuracy': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_metrics': {
                'avg_behavior_f1': avg_behavior_f1,
                'severity_accuracy': severity_matches / total_tests if total_tests > 0 else 0,
                'recommendation_accuracy': recommendation_matches / total_tests if total_tests > 0 else 0
            },
            'failed_test_ids': [f['test_id'] for f in failed_tests],
            'timestamp': datetime.utcnow().isoformat(),
            'results_file': 'inference_test_results.csv'
        }
        
        if debug:
            summary['detailed_results'] = self.test_results
        
        # Log summary
        logger.info(f"Inference testing completed: {passed_tests}/{total_tests} tests passed "
                   f"({summary['accuracy']:.1%} accuracy)")
        
        if failed_tests:
            logger.warning(f"Failed tests: {[f['test_id'] for f in failed_tests]}")
        
        return summary
    
    def save_results_to_csv(self, filename: str = "inference_test_results.csv"):
        """Save test results to CSV file"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'test_id', 'description', 'overall_pass', 'behavior_f1_score',
                    'severity_match', 'recommendation_match', 'expected_behaviors',
                    'actual_behaviors', 'expected_severity', 'actual_severity',
                    'expected_recommendation', 'actual_recommendation',
                    'confidence_overall', 'timestamp', 'notes'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.test_results:
                    # Extract notes about failures
                    notes = []
                    if not result.get('severity_match', True):
                        notes.append("Severity mismatch")
                    if not result.get('recommendation_match', True):
                        notes.append("Recommendation mismatch")
                    if result.get('behavior_analysis', {}).get('f1_score', 1) < 0.7:
                        notes.append("Low behavior F1 score")
                    if result.get('error'):
                        notes.append(f"Error: {result['error']}")
                    
                    row = {
                        'test_id': result.get('test_id', ''),
                        'description': result.get('description', ''),
                        'overall_pass': 'PASS' if result.get('overall_pass', False) else 'FAIL',
                        'behavior_f1_score': result.get('behavior_analysis', {}).get('f1_score', 0),
                        'severity_match': 'YES' if result.get('severity_match', False) else 'NO',
                        'recommendation_match': 'YES' if result.get('recommendation_match', False) else 'NO',
                        'expected_behaviors': ', '.join(result.get('expected', {}).get('behaviors', [])),
                        'actual_behaviors': ', '.join(result.get('actual', {}).get('behaviors', [])),
                        'expected_severity': result.get('expected', {}).get('severity', ''),
                        'actual_severity': result.get('actual', {}).get('severity', ''),
                        'expected_recommendation': result.get('expected', {}).get('recommendation', ''),
                        'actual_recommendation': result.get('actual', {}).get('recommendation', ''),
                        'confidence_overall': result.get('confidence_scores', {}).get('overall', 0),
                        'timestamp': result.get('timestamp', ''),
                        'notes': '; '.join(notes)
                    }
                    writer.writerow(row)
            
            logger.info(f"Test results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")

# Global test runner instance
inference_test_runner = InferenceTestRunner()