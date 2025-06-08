"""
Student Management System for SereniTeach
Handles student roster, BIP awareness, and personalized behavior logging
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class StudentManager:
    """Manages student roster, BIP index, and personalized behavior logging"""
    
    def __init__(self):
        self.student_roster = {}
        self.bip_index = {}
        self.behavior_logs_dir = "behavior_logs"
        self._load_student_data()
        self._ensure_logs_directory()
        logger.info("Student manager initialized with roster and BIP data")
    
    def _load_student_data(self):
        """Load student roster and BIP index from JSON files"""
        try:
            # Load student roster
            with open('student_roster.json', 'r') as f:
                roster_data = json.load(f)
                for student in roster_data['students']:
                    # Create searchable entries by first name, last name, and full name
                    first_name = student['first_name'].lower()
                    last_name = student['last_name'].lower()
                    full_name = f"{first_name} {last_name}"
                    student_id = student['student_id']
                    
                    self.student_roster[first_name] = student
                    self.student_roster[last_name] = student
                    self.student_roster[full_name] = student
                    
            logger.info(f"Loaded {len(roster_data['students'])} students into roster")
            
            # Load BIP index
            with open('bip_index.json', 'r') as f:
                self.bip_index = json.load(f)
                
            logger.info(f"Loaded BIP data for {len(self.bip_index)} students")
            
        except FileNotFoundError as e:
            logger.error(f"Student data file not found: {e}")
            self.student_roster = {}
            self.bip_index = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing student data JSON: {e}")
            self.student_roster = {}
            self.bip_index = {}
    
    def _ensure_logs_directory(self):
        """Ensure behavior logs directory exists"""
        if not os.path.exists(self.behavior_logs_dir):
            os.makedirs(self.behavior_logs_dir)
            logger.info(f"Created behavior logs directory: {self.behavior_logs_dir}")
    
    def identify_student_from_speech(self, speech_text: str) -> Optional[Dict]:
        """
        Identify a student mentioned in speech text
        
        Parameters:
        - speech_text: The spoken text to analyze
        
        Returns:
        - Dictionary with student info and BIP status, or None if no match
        """
        speech_lower = speech_text.lower()
        
        # Look for student names in the speech
        for name_key, student_data in self.student_roster.items():
            if name_key in speech_lower:
                student_id = student_data['student_id']
                has_bip = student_id in self.bip_index
                bip_strategy = self.bip_index.get(student_id, None)
                
                result = {
                    'student_found': True,
                    'student_id': student_id,
                    'first_name': student_data['first_name'],
                    'last_name': student_data['last_name'],
                    'has_bip': has_bip,
                    'bip_strategy': bip_strategy,
                    'matched_name': name_key
                }
                
                logger.info(f"Identified student {student_data['first_name']} {student_data['last_name']} "
                          f"(ID: {student_id}) from speech. BIP: {has_bip}")
                
                return result
        
        return None
    
    def get_bip_enhanced_recommendation(self, base_recommendation: str, student_info: Dict) -> str:
        """
        Enhance a base recommendation with BIP-specific strategies
        
        Parameters:
        - base_recommendation: The original recommendation
        - student_info: Student information including BIP data
        
        Returns:
        - Enhanced recommendation with BIP strategies prioritized
        """
        if not student_info.get('has_bip', False):
            return base_recommendation
        
        bip_strategy = student_info.get('bip_strategy', '')
        student_name = f"{student_info['first_name']} {student_info['last_name']}"
        
        enhanced_recommendation = f"""
**BIP-AWARE RECOMMENDATION for {student_name}:**

Priority Strategy (from BIP): {bip_strategy}

Standard Protocol: {base_recommendation}

NOTE: This student has a Behavior Intervention Plan. The BIP strategy should be implemented first, followed by standard protocols if needed.
""".strip()
        
        logger.info(f"Enhanced recommendation with BIP strategy for {student_name}")
        return enhanced_recommendation
    
    def create_behavior_log_entry(self, student_id: str, incident_data: Dict) -> str:
        """
        Create a behavior log file for a specific student and incident
        
        Parameters:
        - student_id: The student's ID
        - incident_data: Dictionary containing incident details
        
        Returns:
        - Path to the created log file
        """
        try:
            # Create filename with student ID and date
            date_str = datetime.now().strftime("%Y-%m-%d")
            timestamp_str = datetime.now().strftime("%H%M%S")
            log_filename = f"{student_id}_{date_str}_{timestamp_str}.json"
            log_path = os.path.join(self.behavior_logs_dir, log_filename)
            
            # Prepare log entry
            log_entry = {
                'student_id': student_id,
                'incident_date': date_str,
                'incident_time': datetime.now().isoformat(),
                'has_bip': student_id in self.bip_index,
                'bip_strategy_applied': self.bip_index.get(student_id, None),
                'incident_data': incident_data,
                'behavior_timeline': [],
                'recommendations_given': [],
                'outcome': None
            }
            
            # Write log file
            with open(log_path, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
            
            logger.info(f"Created behavior log for student {student_id}: {log_path}")
            return log_path
            
        except Exception as e:
            logger.error(f"Error creating behavior log for student {student_id}: {e}")
            return None
    
    def update_behavior_log(self, student_id: str, log_path: str, update_data: Dict):
        """
        Update an existing behavior log with new information
        
        Parameters:
        - student_id: The student's ID
        - log_path: Path to the existing log file
        - update_data: Dictionary containing update information
        """
        try:
            if not os.path.exists(log_path):
                logger.warning(f"Behavior log not found: {log_path}")
                return
            
            # Read existing log
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Update with new data
            if 'behavior_update' in update_data:
                behavior_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'behavior_description': update_data['behavior_update'],
                    'context': update_data.get('context', {})
                }
                log_data['behavior_timeline'].append(behavior_entry)
            
            if 'recommendation' in update_data:
                recommendation_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': update_data['recommendation'],
                    'bip_enhanced': update_data.get('bip_enhanced', False)
                }
                log_data['recommendations_given'].append(recommendation_entry)
            
            if 'outcome' in update_data:
                log_data['outcome'] = update_data['outcome']
            
            # Write updated log
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"Updated behavior log for student {student_id}")
            
        except Exception as e:
            logger.error(f"Error updating behavior log for student {student_id}: {e}")
    
    def get_student_behavior_history(self, student_id: str, days_back: int = 30) -> List[Dict]:
        """
        Get recent behavior log history for a student
        
        Parameters:
        - student_id: The student's ID
        - days_back: Number of days to look back
        
        Returns:
        - List of behavior log entries
        """
        try:
            history = []
            
            # Look through behavior logs directory
            for filename in os.listdir(self.behavior_logs_dir):
                if filename.startswith(f"{student_id}_") and filename.endswith('.json'):
                    log_path = os.path.join(self.behavior_logs_dir, filename)
                    
                    try:
                        with open(log_path, 'r') as f:
                            log_data = json.load(f)
                            
                        # Check if log is within the time window
                        log_date = datetime.fromisoformat(log_data['incident_time'])
                        days_ago = (datetime.now() - log_date).days
                        
                        if days_ago <= days_back:
                            history.append(log_data)
                            
                    except Exception as e:
                        logger.warning(f"Error reading log file {filename}: {e}")
            
            # Sort by incident time (most recent first)
            history.sort(key=lambda x: x['incident_time'], reverse=True)
            
            logger.info(f"Retrieved {len(history)} behavior log entries for student {student_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving behavior history for student {student_id}: {e}")
            return []
    
    def get_all_students_with_bips(self) -> List[Dict]:
        """
        Get list of all students who have BIPs
        
        Returns:
        - List of student information for those with BIPs
        """
        students_with_bips = []
        
        for student_id, bip_strategy in self.bip_index.items():
            # Find student in roster
            for student in self.student_roster.values():
                if isinstance(student, dict) and student.get('student_id') == student_id:
                    student_info = {
                        'student_id': student_id,
                        'first_name': student['first_name'],
                        'last_name': student['last_name'],
                        'bip_strategy': bip_strategy
                    }
                    students_with_bips.append(student_info)
                    break
        
        return students_with_bips
    
    def get_roster_summary(self) -> Dict:
        """
        Get summary statistics about the student roster and BIPs
        
        Returns:
        - Dictionary with summary statistics
        """
        # Count unique students (avoid duplicates from name variations)
        unique_students = set()
        for student in self.student_roster.values():
            if isinstance(student, dict):
                unique_students.add(student['student_id'])
        
        return {
            'total_students': len(unique_students),
            'students_with_bips': len(self.bip_index),
            'behavior_logs_created': len([f for f in os.listdir(self.behavior_logs_dir) 
                                        if f.endswith('.json')]) if os.path.exists(self.behavior_logs_dir) else 0
        }