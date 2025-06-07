"""
SereniTeach Teacher Feedback Collection System
Collects feedback after incident reports to improve recommendation effectiveness
"""

import os
import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TeacherFeedbackSystem:
    """Handles collection and storage of teacher feedback after incidents"""
    
    def __init__(self):
        self.feedback_file = "feedback_log.csv"
        self.pending_feedback = {}  # session_id -> incident_data
        self._ensure_feedback_file_exists()
        logger.info("Teacher feedback system initialized")
    
    def _ensure_feedback_file_exists(self):
        """Create feedback log file with headers if it doesn't exist"""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Date', 'Time', 'Incident_ID', 'Feedback', 'Comment', 'Teacher_Email'])
            logger.info(f"Created feedback log file: {self.feedback_file}")
    
    def request_feedback(self, session_id, incident_id, teacher_email=None):
        """
        Request feedback from teacher after incident report is sent
        
        Parameters:
        - session_id: Session identifier
        - incident_id: Incident ID for tracking
        - teacher_email: Optional teacher email for record keeping
        
        Returns:
        - Dictionary with feedback request status
        """
        try:
            self.pending_feedback[session_id] = {
                'incident_id': incident_id,
                'teacher_email': teacher_email or 'unknown',
                'requested_at': datetime.now()
            }
            
            logger.info(f"Feedback requested for incident {incident_id}")
            
            return {
                'success': True,
                'feedback_requested': True,
                'message': 'Was this recommendation helpful?',
                'options': ['thumbs_up', 'thumbs_down'],
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error requesting feedback: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def submit_feedback(self, session_id, feedback_type, comment=None):
        """
        Submit teacher feedback for an incident
        
        Parameters:
        - session_id: Session identifier
        - feedback_type: 'thumbs_up' or 'thumbs_down'
        - comment: Optional comment text
        
        Returns:
        - Dictionary with submission status
        """
        try:
            if session_id not in self.pending_feedback:
                return {
                    'success': False,
                    'error': 'No pending feedback request found for this session'
                }
            
            incident_data = self.pending_feedback[session_id]
            
            # Convert feedback type to readable format
            feedback_value = 'positive' if feedback_type == 'thumbs_up' else 'negative'
            
            # Log feedback to CSV file
            timestamp = datetime.now()
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp.strftime('%Y-%m-%d'),
                    timestamp.strftime('%H:%M:%S'),
                    incident_data['incident_id'],
                    feedback_value,
                    comment or '',
                    incident_data['teacher_email']
                ])
            
            # Remove from pending feedback
            del self.pending_feedback[session_id]
            
            logger.info(f"Feedback submitted for incident {incident_data['incident_id']}: {feedback_value}")
            
            result = {
                'success': True,
                'feedback_recorded': True,
                'feedback_type': feedback_value,
                'message': 'Thank you for your feedback!'
            }
            
            if comment:
                result['comment_recorded'] = True
                logger.info(f"Comment recorded for incident {incident_data['incident_id']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def has_pending_feedback(self, session_id):
        """Check if there's pending feedback for a session"""
        return session_id in self.pending_feedback
    
    def get_feedback_stats(self):
        """
        Get feedback statistics from the log file
        
        Returns:
        - Dictionary with feedback statistics
        """
        try:
            if not os.path.exists(self.feedback_file):
                return {'total': 0, 'positive': 0, 'negative': 0, 'with_comments': 0}
            
            stats = {'total': 0, 'positive': 0, 'negative': 0, 'with_comments': 0}
            
            with open(self.feedback_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    stats['total'] += 1
                    if row['Feedback'] == 'positive':
                        stats['positive'] += 1
                    elif row['Feedback'] == 'negative':
                        stats['negative'] += 1
                    if row['Comment'].strip():
                        stats['with_comments'] += 1
            
            # Calculate percentages
            if stats['total'] > 0:
                stats['positive_percentage'] = round((stats['positive'] / stats['total']) * 100, 1)
                stats['negative_percentage'] = round((stats['negative'] / stats['total']) * 100, 1)
            else:
                stats['positive_percentage'] = 0
                stats['negative_percentage'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {str(e)}")
            return {'error': str(e)}
    
    def get_recent_feedback(self, limit=10):
        """
        Get recent feedback entries
        
        Parameters:
        - limit: Number of recent entries to return
        
        Returns:
        - List of recent feedback entries
        """
        try:
            if not os.path.exists(self.feedback_file):
                return []
            
            feedback_entries = []
            with open(self.feedback_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                feedback_entries = list(reader)
            
            # Return most recent entries (reverse order)
            return feedback_entries[-limit:] if feedback_entries else []
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {str(e)}")
            return []