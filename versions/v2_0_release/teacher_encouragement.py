"""
SereniTeach Teacher Encouragement System
Provides periodic supportive messages during behavioral crises to help with teacher emotional regulation
"""

import logging
import random
import threading
import time
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TeacherEncouragementSystem:
    """Handles periodic encouragement messages during active crisis situations"""
    
    def __init__(self):
        self.active_sessions = {}  # session_id -> encouragement data
        self.encouragement_phrases = [
            "You're doing great—stay calm and focused.",
            "Keep breathing—you've got this.",
            "Remember, your calm helps the student calm down.",
            "You're handling this with professionalism.",
            "Take a deep breath—you're not alone.",
            "Your steady presence makes all the difference.",
            "Trust your training—you know what to do.",
            "Stay centered, you're creating a safe space.",
            "Your compassion is helping right now.",
            "One moment at a time—you're doing well.",
            "Your voice and manner are powerful tools.",
            "Keep your energy grounded and steady.",
            "You're showing strength and wisdom.",
            "This too shall pass—stay present.",
            "Your calmness is contagious in the best way."
        ]
        self.encouragement_interval = 50  # seconds between encouragements
        logger.info("Teacher encouragement system initialized")
    
    def start_encouragement(self, session_id, enabled=True):
        """
        Start encouragement cycle for a session during crisis
        
        Parameters:
        - session_id: Session identifier
        - enabled: Whether encouragement is enabled by default
        
        Returns:
        - Dictionary with start status
        """
        try:
            if session_id in self.active_sessions:
                logger.warning(f"Encouragement already active for session {session_id}")
                return {'success': False, 'message': 'Encouragement already active'}
            
            # Initialize session data
            session_data = {
                'enabled': enabled,
                'start_time': datetime.now(),
                'last_encouragement': None,
                'encouragement_count': 0,
                'timer_thread': None,
                'stop_flag': threading.Event()
            }
            
            self.active_sessions[session_id] = session_data
            
            # Start background timer if enabled
            if enabled:
                self._start_timer_thread(session_id)
            
            logger.info(f"Started encouragement system for session {session_id} (enabled: {enabled})")
            
            return {
                'success': True,
                'encouragement_started': True,
                'enabled': enabled,
                'message': 'Encouragement system activated' if enabled else 'Encouragement system ready (disabled)'
            }
            
        except Exception as e:
            logger.error(f"Error starting encouragement: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def stop_encouragement(self, session_id):
        """
        Stop encouragement cycle for a session
        
        Parameters:
        - session_id: Session identifier
        
        Returns:
        - Dictionary with stop status and summary
        """
        try:
            if session_id not in self.active_sessions:
                return {'success': False, 'message': 'No active encouragement for this session'}
            
            session_data = self.active_sessions[session_id]
            
            # Stop the timer thread
            session_data['stop_flag'].set()
            if session_data['timer_thread'] and session_data['timer_thread'].is_alive():
                session_data['timer_thread'].join(timeout=2)
            
            # Calculate summary stats
            duration = (datetime.now() - session_data['start_time']).total_seconds()
            encouragement_count = session_data['encouragement_count']
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Stopped encouragement for session {session_id}. Sent {encouragement_count} messages over {duration:.0f} seconds")
            
            return {
                'success': True,
                'encouragement_stopped': True,
                'duration_seconds': duration,
                'total_encouragements': encouragement_count,
                'message': f'Encouragement stopped. Provided {encouragement_count} supportive messages.'
            }
            
        except Exception as e:
            logger.error(f"Error stopping encouragement: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def toggle_encouragement(self, session_id, enable=None):
        """
        Toggle encouragement on/off for an active session
        
        Parameters:
        - session_id: Session identifier
        - enable: Optional boolean to set specific state, None to toggle
        
        Returns:
        - Dictionary with toggle status
        """
        try:
            if session_id not in self.active_sessions:
                return {'success': False, 'message': 'No active encouragement session to toggle'}
            
            session_data = self.active_sessions[session_id]
            
            # Determine new state
            if enable is None:
                new_state = not session_data['enabled']
            else:
                new_state = enable
            
            # Update state
            session_data['enabled'] = new_state
            
            if new_state:
                # Start timer if not already running
                if not session_data['timer_thread'] or not session_data['timer_thread'].is_alive():
                    session_data['stop_flag'].clear()
                    self._start_timer_thread(session_id)
                message = "Encouragement turned on"
            else:
                # Stop timer
                session_data['stop_flag'].set()
                message = "Encouragement turned off"
            
            logger.info(f"Toggled encouragement for session {session_id}: {message}")
            
            return {
                'success': True,
                'encouragement_toggled': True,
                'enabled': new_state,
                'message': message
            }
            
        except Exception as e:
            logger.error(f"Error toggling encouragement: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _start_timer_thread(self, session_id):
        """Start background timer thread for encouragement messages"""
        def timer_loop():
            while session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                
                # Wait for interval or stop signal
                if session_data['stop_flag'].wait(timeout=self.encouragement_interval):
                    break  # Stop flag was set
                
                # Check if session still exists and is enabled
                if session_id not in self.active_sessions or not session_data['enabled']:
                    break
                
                # Generate and log encouragement
                encouragement = self._generate_encouragement()
                self._deliver_encouragement(session_id, encouragement)
        
        thread = threading.Thread(target=timer_loop, daemon=True)
        thread.start()
        self.active_sessions[session_id]['timer_thread'] = thread
    
    def _generate_encouragement(self):
        """Generate a random encouragement message"""
        return random.choice(self.encouragement_phrases)
    
    def _deliver_encouragement(self, session_id, message):
        """
        Deliver encouragement message to the session
        
        Parameters:
        - session_id: Session identifier
        - message: Encouragement message to deliver
        """
        try:
            if session_id not in self.active_sessions:
                return
            
            session_data = self.active_sessions[session_id]
            session_data['last_encouragement'] = datetime.now()
            session_data['encouragement_count'] += 1
            
            logger.info(f"Encouragement #{session_data['encouragement_count']} for session {session_id}: {message}")
            
            # In a real implementation, this would use text-to-speech
            # For now, we log the encouragement message
            # The frontend will display these messages to the teacher
            
        except Exception as e:
            logger.error(f"Error delivering encouragement: {str(e)}")
    
    def get_encouragement_status(self, session_id):
        """
        Get current encouragement status for a session
        
        Parameters:
        - session_id: Session identifier
        
        Returns:
        - Dictionary with current status
        """
        if session_id not in self.active_sessions:
            return {'active': False}
        
        session_data = self.active_sessions[session_id]
        duration = (datetime.now() - session_data['start_time']).total_seconds()
        
        return {
            'active': True,
            'enabled': session_data['enabled'],
            'duration_seconds': duration,
            'encouragement_count': session_data['encouragement_count'],
            'last_encouragement': session_data['last_encouragement'].isoformat() if session_data['last_encouragement'] else None
        }
    
    def get_latest_encouragement(self, session_id):
        """
        Get the latest encouragement message for display
        
        Parameters:
        - session_id: Session identifier
        
        Returns:
        - Latest encouragement data or None
        """
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        
        if session_data['last_encouragement']:
            # Check if this is a recent encouragement (within last 5 seconds)
            time_since = (datetime.now() - session_data['last_encouragement']).total_seconds()
            if time_since <= 5:
                return {
                    'message': self.encouragement_phrases[session_data['encouragement_count'] % len(self.encouragement_phrases)],
                    'timestamp': session_data['last_encouragement'].isoformat(),
                    'count': session_data['encouragement_count']
                }
        
        return None
    
    def has_active_encouragement(self, session_id):
        """Check if there's active encouragement for a session"""
        return session_id in self.active_sessions
    
    def cleanup_session(self, session_id):
        """Clean up encouragement session (called when crisis ends)"""
        if session_id in self.active_sessions:
            self.stop_encouragement(session_id)