"""
SereniTeach Incident Reporting System
Automatically generates and emails behavior incident reports when crises end
"""
import os
import json
import logging
from datetime import datetime
from flask_mail import Message
from flask import current_app
from behavior_analytics import BehaviorAnalytics

logger = logging.getLogger(__name__)

class IncidentReporter:
    """Handles incident logging and report generation"""
    
    def __init__(self):
        self.active_incidents = {}  # Dictionary to store active incident logs by session
        self.analytics = BehaviorAnalytics()  # Initialize analytics module
        logger.info("Incident reporting system initialized")
    
    def start_incident_log(self, session_id, initial_data):
        """
        Start a new incident log when crisis is detected
        
        Parameters:
        - session_id: Unique identifier for the session/user
        - initial_data: Dictionary containing initial crisis data
        
        Returns:
        - incident_id: Unique identifier for this incident
        """
        incident_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize incident log
        incident_log = {
            "incident_id": incident_id,
            "session_id": session_id,
            "start_timestamp": datetime.now(),
            "end_timestamp": None,
            "location": initial_data.get('location', 'classroom'),
            "initial_behavior": initial_data.get('behavior_description', ''),
            "keywords": initial_data.get('keywords', []),
            "severity": initial_data.get('severity', 'medium'),
            "context": {
                "time_period": initial_data.get('time_period', ''),
                "noise_level_db": initial_data.get('noise_level_db', 0),
                "is_transition": initial_data.get('is_transition', False)
            },
            "interactions": [],
            "recommendations": [],
            "behavior_updates": [],
            "peak_noise_level": initial_data.get('noise_level_db', 0),
            "crisis_detected": initial_data.get('is_crisis', False)
        }
        
        # Store the active incident
        self.active_incidents[session_id] = incident_log
        
        logger.info(f"Started incident log {incident_id} for session {session_id}")
        return incident_id
    
    def log_interaction(self, session_id, interaction_type, content, timestamp=None):
        """
        Log an interaction during the incident
        
        Parameters:
        - session_id: Session identifier
        - interaction_type: Type of interaction (voice_input, recommendation, behavior_update, etc.)
        - content: Content of the interaction
        - timestamp: Optional timestamp (defaults to now)
        """
        if session_id not in self.active_incidents:
            logger.warning(f"No active incident for session {session_id}")
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        interaction = {
            "timestamp": timestamp,
            "type": interaction_type,
            "content": content
        }
        
        incident_log = self.active_incidents[session_id]
        incident_log["interactions"].append(interaction)
        
        # Update specific lists based on interaction type
        if interaction_type == "recommendation":
            incident_log["recommendations"].append(content)
        elif interaction_type == "behavior_update":
            incident_log["behavior_updates"].append(content)
        
        # Update peak noise level if provided
        if interaction_type == "context_update" and "noise_level_db" in content:
            current_noise = content.get("noise_level_db", 0)
            if current_noise > incident_log["peak_noise_level"]:
                incident_log["peak_noise_level"] = current_noise
        
        logger.debug(f"Logged {interaction_type} for incident {session_id}")
    
    def log_recommendation(self, session_id, recommendation_text):
        """Log a recommendation given by the system"""
        self.log_interaction(session_id, "recommendation", recommendation_text)
    
    def log_behavior_update(self, session_id, behavior_description):
        """Log an update to the behavior situation"""
        self.log_interaction(session_id, "behavior_update", behavior_description)
    
    def log_voice_input(self, session_id, voice_text):
        """Log voice input from the teacher"""
        self.log_interaction(session_id, "voice_input", voice_text)
    
    def end_incident(self, session_id, outcome="resolved"):
        """
        End the incident and generate the report
        
        Parameters:
        - session_id: Session identifier
        - outcome: How the incident was resolved
        
        Returns:
        - Dictionary containing report data and email status
        """
        if session_id not in self.active_incidents:
            logger.warning(f"No active incident to end for session {session_id}")
            return {"success": False, "error": "No active incident found"}
        
        incident_log = self.active_incidents[session_id]
        incident_log["end_timestamp"] = datetime.now()
        incident_log["outcome"] = outcome
        
        # Generate the formatted report
        report_text = self._format_incident_report(incident_log)
        
        # Save report to file
        report_filename = self._save_report_to_file(incident_log, report_text)
        
        # Send email report
        email_result = self._send_report_email(incident_log, report_text)
        
        # Remove from active incidents
        del self.active_incidents[session_id]
        
        logger.info(f"Incident {incident_log['incident_id']} completed and report generated")
        
        return {
            "success": True,
            "incident_id": incident_log["incident_id"],
            "report_filename": report_filename,
            "email_sent": email_result["success"],
            "email_details": email_result,
            "report_text": report_text,
            "feedback_ready": True,
            "teacher_email": "stevenrayhinojosa@gmail.com"
        }
    
    def _format_incident_report(self, incident_log):
        """
        Format the incident data into a readable report
        
        Parameters:
        - incident_log: Complete incident log dictionary
        
        Returns:
        - Formatted report string
        """
        start_time = incident_log["start_timestamp"].strftime("%I:%M %p")
        end_time = incident_log["end_timestamp"].strftime("%I:%M %p") if incident_log["end_timestamp"] else "Ongoing"
        date = incident_log["start_timestamp"].strftime("%B %d, %Y")
        duration = ""
        
        if incident_log["end_timestamp"]:
            duration_delta = incident_log["end_timestamp"] - incident_log["start_timestamp"]
            duration_minutes = int(duration_delta.total_seconds() / 60)
            duration = f" (Duration: {duration_minutes} minutes)"
        
        report = f"""BEHAVIOR INCIDENT REPORT
Generated by SereniTeach System

Date: {date}
Time: {start_time} to {end_time}{duration}
Location: {incident_log['location'].title()}
Incident ID: {incident_log['incident_id']}

INITIAL SITUATION:
{incident_log['initial_behavior']}

"""
        
        # Add behavior updates if any
        if incident_log['behavior_updates']:
            report += "BEHAVIOR PROGRESSION:\n"
            for i, update in enumerate(incident_log['behavior_updates'], 1):
                report += f"{i}. {update}\n"
            report += "\n"
        
        # Add recommendations given
        if incident_log['recommendations']:
            report += "ACTIONS RECOMMENDED BY SYSTEM:\n"
            for i, rec in enumerate(incident_log['recommendations'], 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        # Add environmental context
        report += f"ENVIRONMENTAL CONTEXT:\n"
        report += f"Time Period: {incident_log['context']['time_period'].replace('-', ' ').title()}\n"
        report += f"Peak Noise Level: {incident_log['peak_noise_level']:.1f} dB\n"
        if incident_log['context']['is_transition']:
            report += "Note: Incident occurred during transition period\n"
        report += "\n"
        
        # Add crisis information
        if incident_log['crisis_detected']:
            report += "CRISIS STATUS: Emergency protocols activated\n"
            if incident_log['keywords']:
                keywords_str = ', '.join(incident_log['keywords'])
                report += f"Emergency Indicators: {keywords_str}\n"
            report += "\n"
        
        # Add timeline of interactions
        if incident_log['interactions']:
            report += "DETAILED TIMELINE:\n"
            for interaction in incident_log['interactions']:
                time_str = interaction['timestamp'].strftime("%I:%M:%S %p")
                report += f"{time_str} - {interaction['type'].replace('_', ' ').title()}: {interaction['content']}\n"
            report += "\n"
        
        # Add outcome
        report += f"OUTCOME: {incident_log.get('outcome', 'Not specified').title()}\n\n"
        
        report += "END OF REPORT\n"
        report += f"Report generated at {datetime.now().strftime('%I:%M %p on %B %d, %Y')}"
        
        return report
    
    def _save_report_to_file(self, incident_log, report_text):
        """
        Save the incident report to a local file
        
        Parameters:
        - incident_log: Incident log dictionary
        - report_text: Formatted report text
        
        Returns:
        - Filename of saved report
        """
        try:
            # Create reports directory if it doesn't exist
            reports_dir = "incident_reports"
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            # Generate filename with timestamp
            timestamp = incident_log["start_timestamp"].strftime("%Y%m%d_%H%M%S")
            filename = f"incident_report_{timestamp}_{incident_log['session_id']}.txt"
            filepath = os.path.join(reports_dir, filename)
            
            # Write report to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logger.info(f"Incident report saved to {filepath}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save incident report to file: {str(e)}")
            return None
    
    def _send_report_email(self, incident_log, report_text):
        """
        Send the incident report via email with trend visualizations
        
        Parameters:
        - incident_log: Incident log dictionary
        - report_text: Formatted report text
        
        Returns:
        - Dictionary with email sending results
        """
        try:
            # Generate email subject
            date_str = incident_log["start_timestamp"].strftime("%B %d, %Y")
            subject = f"Behavior Incident Report - {incident_log['location'].title()} - {date_str}"
            
            # Generate trend visualizations
            student_id = None
            if 'student_info' in incident_log and incident_log['student_info']:
                student_id = incident_log['student_info']['student_id']
            
            analytics_result = self.analytics.generate_all_visualizations(student_id)
            
            # Get email configuration
            mail_username = current_app.config.get('MAIL_USERNAME')
            teacher_email = "stevenrayhinojosa@gmail.com"  # Default teacher email
            
            if not mail_username:
                logger.warning("Email credentials not configured - cannot send incident report")
                return {
                    "success": False,
                    "error": "Email service not configured"
                }
            
            # Enhance email body with analytics summary
            enhanced_body = report_text
            
            if analytics_result['success'] and analytics_result['data_available']:
                enhanced_body += "\n" + "="*60 + "\n"
                enhanced_body += "BEHAVIOR TREND ANALYSIS & PROACTIVE PLANNING\n"
                enhanced_body += "="*60 + "\n\n"
                
                summary = analytics_result['summary']
                enhanced_body += f"Analysis Period: {summary.get('date_range', 'N/A')}\n"
                
                if analytics_result['student_specific']:
                    enhanced_body += "INDIVIDUAL STUDENT ANALYSIS:\n"
                else:
                    enhanced_body += "CLASS-WIDE BEHAVIOR PATTERNS:\n"
                
                for insight in summary.get('insights', []):
                    enhanced_body += f"• {insight}\n"
                
                enhanced_body += "\nPROACTIVE RECOMMENDATIONS:\n"
                enhanced_body += "• Review attached visualizations to identify behavior patterns\n"
                enhanced_body += "• Consider environmental modifications during peak incident times\n"
                enhanced_body += "• Implement preventive strategies based on trend analysis\n"
                enhanced_body += "• Monitor noise levels during transition periods\n"
                
                if analytics_result['student_specific']:
                    enhanced_body += "• Review individual BIP strategies for effectiveness\n"
                else:
                    enhanced_body += "• Consider class-wide behavioral support strategies\n"
                
                enhanced_body += "\nATTACHED VISUALIZATIONS:\n"
                enhanced_body += "1. Behavior Frequency Chart - Shows most common behavior patterns\n"
                enhanced_body += "2. Time Pattern Heatmap - Reveals peak incident times for planning\n"
                enhanced_body += "3. Environmental Analysis - Correlates noise levels with escalation\n"
            else:
                enhanced_body += "\n" + "="*60 + "\n"
                enhanced_body += "TREND ANALYSIS NOTE\n"
                enhanced_body += "="*60 + "\n"
                enhanced_body += "Insufficient historical data for trend analysis.\n"
                enhanced_body += "Visualizations will be available after more incidents are logged.\n"
            
            # Create email message
            msg = Message(
                subject=subject,
                sender=mail_username,
                recipients=[teacher_email],
                body=enhanced_body
            )
            
            # Attach trend visualization charts if available
            if analytics_result['success'] and analytics_result['chart_paths']:
                for chart_path in analytics_result['chart_paths']:
                    if os.path.exists(chart_path):
                        with open(chart_path, 'rb') as f:
                            chart_filename = os.path.basename(chart_path)
                            msg.attach(
                                filename=chart_filename,
                                content_type='image/png',
                                data=f.read()
                            )
                        logger.info(f"Attached visualization: {chart_filename}")
            
            # Send email using Flask-Mail
            mail = current_app.extensions.get('mail')
            if mail:
                mail.send(msg)
                logger.info(f"Incident report with visualizations emailed successfully to {teacher_email}")
                
                # Clean up chart files after sending
                if analytics_result['success'] and analytics_result['chart_paths']:
                    self.analytics.cleanup_charts()
                
                return {
                    "success": True,
                    "message": "Incident report with trend analysis sent successfully",
                    "recipient": teacher_email,
                    "subject": subject,
                    "charts_attached": len(analytics_result.get('chart_paths', [])),
                    "analytics_included": analytics_result['success']
                }
            else:
                logger.error("Mail service not initialized")
                return {
                    "success": False,
                    "error": "Mail service not initialized"
                }
                
        except Exception as e:
            logger.error(f"Failed to send incident report email: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_active_incident(self, session_id):
        """Get the active incident for a session"""
        return self.active_incidents.get(session_id)
    
    def has_active_incident(self, session_id):
        """Check if there's an active incident for a session"""
        return session_id in self.active_incidents

# Create global instance
incident_reporter = IncidentReporter()