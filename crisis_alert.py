"""
SereniTeach Crisis Alert System
Sends SMS notifications to front desk when behavioral crises are detected
"""
import os
import logging
from datetime import datetime
from twilio.rest import Client

logger = logging.getLogger(__name__)

class CrisisAlertSystem:
    """Handles crisis detection and automated alert notifications"""
    
    def __init__(self):
        self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
        self.front_desk_number = "+12792000303"  # Front desk number
        
        # Initialize Twilio client if credentials are available
        if self.twilio_account_sid and self.twilio_auth_token:
            self.client = Client(self.twilio_account_sid, self.twilio_auth_token)
            logger.info("Twilio SMS client initialized successfully")
        else:
            self.client = None
            logger.warning("Twilio credentials not found - SMS alerts disabled")
    
    def is_crisis_situation(self, analysis_data):
        """
        Determine if the situation qualifies as a crisis requiring immediate alert
        
        Parameters:
        - analysis_data: Dictionary containing behavior analysis results
        
        Returns:
        - Boolean indicating if this is a crisis situation
        """
        if not analysis_data:
            return False
        
        # Check for emergency signals
        if analysis_data.get('is_emergency', False):
            return True
        
        # Check for high-risk keywords
        keywords = analysis_data.get('keywords', [])
        crisis_keywords = [
            'weapon', 'dangerous', 'violent', 'threatening', 'harm', 'hurt', 
            'emergency', 'crisis', 'safety', 'injured', 'attack', 'fight'
        ]
        
        if any(keyword in crisis_keywords for keyword in keywords):
            return True
        
        # Check severity level if available
        severity = analysis_data.get('severity', '')
        if severity and severity.lower() in ['high', 'severe', 'critical']:
            return True
        
        # Check protocol analysis for emergency indicators
        protocol_analysis = analysis_data.get('protocol_analysis', {})
        if protocol_analysis.get('is_emergency', False):
            return True
        
        return False
    
    def generate_crisis_message(self, analysis_data, location="classroom"):
        """
        Generate appropriate crisis alert message
        
        Parameters:
        - analysis_data: Dictionary containing behavior analysis results
        - location: Physical location of the incident
        
        Returns:
        - String containing the alert message
        """
        timestamp = datetime.now().strftime("%I:%M %p")
        
        # Extract key information
        keywords = analysis_data.get('keywords', [])
        behavior_type = analysis_data.get('behavior_type', 'behavioral incident')
        
        # Create concise alert message
        message = f"🚨 BEHAVIORAL CRISIS ALERT - {timestamp}\n\n"
        message += f"Location: {location.title()}\n"
        message += f"Situation: {behavior_type.title()}\n"
        
        if keywords:
            key_indicators = ', '.join(keywords[:3])  # Show top 3 keywords
            message += f"Indicators: {key_indicators}\n"
        
        message += "\nImmediate response needed. SereniTeach alert system."
        
        return message
    
    def send_crisis_alert(self, analysis_data, location="classroom"):
        """
        Send SMS alert to front desk about crisis situation
        
        Parameters:
        - analysis_data: Dictionary containing behavior analysis results
        - location: Physical location of the incident
        
        Returns:
        - Dictionary with success status and message details
        """
        if not self.client:
            logger.error("Cannot send SMS - Twilio client not initialized")
            return {
                "success": False,
                "error": "SMS service not configured",
                "message": "Twilio credentials required for SMS alerts"
            }
        
        if not self.twilio_phone_number:
            logger.error("Cannot send SMS - Twilio phone number not configured")
            return {
                "success": False,
                "error": "Twilio phone number not set",
                "message": "TWILIO_PHONE_NUMBER environment variable required"
            }
        
        try:
            # Generate the alert message
            alert_message = self.generate_crisis_message(analysis_data, location)
            
            # Send SMS using Twilio
            message = self.client.messages.create(
                body=alert_message,
                from_=self.twilio_phone_number,
                to=self.front_desk_number
            )
            
            logger.info(f"Crisis alert SMS sent successfully. Message SID: {message.sid}")
            
            return {
                "success": True,
                "message_sid": message.sid,
                "message": "Crisis alert sent to front desk",
                "phone_number": self.front_desk_number,
                "alert_text": alert_message
            }
            
        except Exception as e:
            logger.error(f"Failed to send crisis alert SMS: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send crisis alert"
            }
    
    def process_behavior_incident(self, analysis_data, location="classroom"):
        """
        Process a behavior incident and send alert if crisis detected
        
        Parameters:
        - analysis_data: Dictionary containing behavior analysis results
        - location: Physical location of the incident
        
        Returns:
        - Dictionary with processing results and alert status
        """
        # Check if this qualifies as a crisis
        is_crisis = self.is_crisis_situation(analysis_data)
        
        if is_crisis:
            logger.warning("Crisis situation detected - sending alert")
            alert_result = self.send_crisis_alert(analysis_data, location)
            
            return {
                "crisis_detected": True,
                "alert_sent": alert_result["success"],
                "alert_details": alert_result
            }
        else:
            logger.info("Behavior incident logged - no crisis alert needed")
            return {
                "crisis_detected": False,
                "alert_sent": False,
                "alert_details": {"message": "No crisis detected"}
            }

# Create global instance
crisis_alert_system = CrisisAlertSystem()