"""
Database service layer for managing behavioral data and protocols in PostgreSQL
"""
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import joinedload
from app import db
from models import (
    Protocol, DecisionPoint, DecisionOption, BehaviorType, BehaviorProtocol,
    Recommendation, Student, BehaviorIncident, IncidentInteraction,
    ProtocolKeyword, BehaviorTrend, BehavioralData, SeverityLevel
)

logger = logging.getLogger(__name__)

class ProtocolDatabaseService:
    """Service for managing protocol data in PostgreSQL"""
    
    @staticmethod
    def get_all_protocols() -> List[Protocol]:
        """Get all protocols from database"""
        return Protocol.query.all()
    
    @staticmethod
    def get_protocol_by_id(protocol_id: int) -> Optional[Protocol]:
        """Get protocol by ID with decision points"""
        return Protocol.query.options(
            joinedload(Protocol.decision_points).joinedload(DecisionPoint.options)
        ).filter_by(id=protocol_id).first()
    
    @staticmethod
    def get_protocol_by_name(name: str) -> Optional[Protocol]:
        """Get protocol by name"""
        return Protocol.query.filter_by(name=name).first()
    
    @staticmethod
    def get_protocols_by_category(category: str) -> List[Protocol]:
        """Get protocols by category"""
        return Protocol.query.filter_by(category=category).all()
    
    @staticmethod
    def search_protocols_by_keywords(keywords: List[str]) -> List[Dict[str, Any]]:
        """Search protocols by keywords using NLP mapping"""
        if not keywords:
            return []
        
        protocol_scores = {}
        
        # Search through protocol keywords
        for keyword in keywords:
            keyword_matches = ProtocolKeyword.query.filter(
                ProtocolKeyword.keyword.ilike(f'%{keyword.lower()}%')
            ).all()
            
            for match in keyword_matches:
                protocol_id = match.protocol_id
                if protocol_id not in protocol_scores:
                    protocol_scores[protocol_id] = {
                        'protocol': match.protocol,
                        'score': 0,
                        'matching_keywords': []
                    }
                
                protocol_scores[protocol_id]['score'] += match.weight
                protocol_scores[protocol_id]['matching_keywords'].append(match.keyword)
        
        # Sort by score and return
        sorted_protocols = sorted(
            protocol_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_protocols
    
    @staticmethod
    def get_protocol_for_behavior(behavior_type: str, severity: str) -> Optional[Protocol]:
        """Get recommended protocol for specific behavior type and severity"""
        behavior = BehaviorType.query.filter_by(name=behavior_type).first()
        if not behavior:
            return None
        
        behavior_protocol = BehaviorProtocol.query.filter_by(
            behavior_type_id=behavior.id,
            severity_level=severity,
            is_primary=True
        ).first()
        
        if behavior_protocol:
            return behavior_protocol.protocol
        
        # Fallback to any protocol for this behavior
        behavior_protocol = BehaviorProtocol.query.filter_by(
            behavior_type_id=behavior.id
        ).first()
        
        return behavior_protocol.protocol if behavior_protocol else None

class BehaviorDatabaseService:
    """Service for managing behavior data in PostgreSQL"""
    
    @staticmethod
    def get_all_behavior_types() -> List[BehaviorType]:
        """Get all behavior types"""
        return BehaviorType.query.all()
    
    @staticmethod
    def get_behavior_type_by_name(name: str) -> Optional[BehaviorType]:
        """Get behavior type by name"""
        return BehaviorType.query.filter_by(name=name).first()
    
    @staticmethod
    def get_behaviors_by_category(category: str) -> List[BehaviorType]:
        """Get behaviors by category"""
        return BehaviorType.query.filter_by(category=category).all()
    
    @staticmethod
    def create_behavior_incident(incident_data: Dict[str, Any]) -> BehaviorIncident:
        """Create new behavior incident record"""
        try:
            incident = BehaviorIncident(
                incident_id=incident_data['incident_id'],
                session_id=incident_data['session_id'],
                behavior_description=incident_data['behavior_description'],
                behavior_type=incident_data.get('behavior_type'),
                severity_level=incident_data['severity_level'],
                location=incident_data['location'],
                start_time=incident_data.get('start_time', datetime.utcnow()),
                time_period=incident_data.get('time_period'),
                noise_level_db=incident_data.get('noise_level_db'),
                is_transition_period=incident_data.get('is_transition_period', False),
                is_crisis=incident_data.get('is_crisis', False),
                protocol_used_id=incident_data.get('protocol_used_id'),
                keywords=json.dumps(incident_data.get('keywords', [])),
                confidence_scores=json.dumps(incident_data.get('confidence_scores', {})),
                clarification_prompted=incident_data.get('clarification_prompted', False),
                clarification_response=incident_data.get('clarification_response')
            )
            
            # Link to student if provided
            if 'student_id' in incident_data:
                student = StudentDatabaseService.get_student_by_id(incident_data['student_id'])
                if student:
                    incident.student_id = student.id
            
            db.session.add(incident)
            db.session.commit()
            
            logger.info(f"Created behavior incident: {incident.incident_id}")
            return incident
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating behavior incident: {e}")
            raise
    
    @staticmethod
    def update_incident_outcome(incident_id: str, outcome_data: Dict[str, Any]) -> Optional[BehaviorIncident]:
        """Update incident with outcome data"""
        try:
            incident = BehaviorIncident.query.filter_by(incident_id=incident_id).first()
            if not incident:
                return None
            
            incident.end_time = outcome_data.get('end_time', datetime.utcnow())
            incident.duration_minutes = outcome_data.get('duration_minutes')
            incident.outcome = outcome_data.get('outcome')
            incident.recommendations_given = json.dumps(outcome_data.get('recommendations_given', []))
            incident.teacher_feedback = outcome_data.get('teacher_feedback')
            incident.feedback_comment = outcome_data.get('feedback_comment')
            incident.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Updated incident outcome: {incident_id}")
            return incident
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating incident outcome: {e}")
            raise
    
    @staticmethod
    def add_incident_interaction(incident_id: str, interaction_data: Dict[str, Any]) -> IncidentInteraction:
        """Add interaction to incident"""
        try:
            incident = BehaviorIncident.query.filter_by(incident_id=incident_id).first()
            if not incident:
                raise ValueError(f"Incident not found: {incident_id}")
            
            interaction = IncidentInteraction(
                incident_id=incident.id,
                interaction_type=interaction_data['interaction_type'],
                content=interaction_data['content'],
                timestamp=interaction_data.get('timestamp', datetime.utcnow()),
                interaction_metadata=json.dumps(interaction_data.get('metadata', {}))
            )
            
            db.session.add(interaction)
            db.session.commit()
            
            return interaction
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding incident interaction: {e}")
            raise
    
    @staticmethod
    def get_incident_by_id(incident_id: str) -> Optional[BehaviorIncident]:
        """Get incident by ID with all related data"""
        return BehaviorIncident.query.options(
            joinedload(BehaviorIncident.interactions),
            joinedload(BehaviorIncident.student),
            joinedload(BehaviorIncident.protocol_used)
        ).filter_by(incident_id=incident_id).first()
    
    @staticmethod
    def get_incidents_by_date_range(start_date: date, end_date: date) -> List[BehaviorIncident]:
        """Get incidents within date range"""
        return BehaviorIncident.query.filter(
            and_(
                func.date(BehaviorIncident.start_time) >= start_date,
                func.date(BehaviorIncident.start_time) <= end_date
            )
        ).order_by(desc(BehaviorIncident.start_time)).all()

class StudentDatabaseService:
    """Service for managing student data in PostgreSQL"""
    
    @staticmethod
    def get_all_students() -> List[Student]:
        """Get all students"""
        return Student.query.all()
    
    @staticmethod
    def get_student_by_id(student_id: str) -> Optional[Student]:
        """Get student by student ID"""
        return Student.query.filter_by(student_id=student_id).first()
    
    @staticmethod
    def get_student_by_name(first_name: str, last_name: str) -> Optional[Student]:
        """Get student by name"""
        return Student.query.filter_by(
            first_name=first_name,
            last_name=last_name
        ).first()
    
    @staticmethod
    def get_students_with_bip() -> List[Student]:
        """Get all students with BIP plans"""
        return Student.query.filter_by(has_bip=True).all()
    
    @staticmethod
    def get_student_incidents(student_id: str, limit: int = 50) -> List[BehaviorIncident]:
        """Get recent incidents for a student"""
        student = StudentDatabaseService.get_student_by_id(student_id)
        if not student:
            return []
        
        return BehaviorIncident.query.filter_by(
            student_id=student.id
        ).order_by(desc(BehaviorIncident.start_time)).limit(limit).all()
    
    @staticmethod
    def search_students(query: str) -> List[Student]:
        """Search students by name or ID"""
        search_term = f'%{query}%'
        return Student.query.filter(
            or_(
                Student.student_id.ilike(search_term),
                Student.first_name.ilike(search_term),
                Student.last_name.ilike(search_term)
            )
        ).all()

class AnalyticsDatabaseService:
    """Service for managing analytics and trends in PostgreSQL"""
    
    @staticmethod
    def update_behavior_trends(student_id: Optional[str] = None, target_date: Optional[date] = None):
        """Update behavior trend aggregations"""
        try:
            if target_date is None:
                target_date = date.today()
            
            # Get incidents for the date
            incidents_query = BehaviorIncident.query.filter(
                func.date(BehaviorIncident.start_time) == target_date
            )
            
            if student_id:
                student = StudentDatabaseService.get_student_by_id(student_id)
                if student:
                    incidents_query = incidents_query.filter_by(student_id=student.id)
            
            incidents = incidents_query.all()
            
            # Group by student and behavior type
            trend_data = {}
            for incident in incidents:
                key = (incident.student_id, incident.behavior_type or 'Unknown')
                if key not in trend_data:
                    trend_data[key] = {
                        'student_id': incident.student_id,
                        'behavior_type': incident.behavior_type or 'Unknown',
                        'incidents': [],
                        'crisis_count': 0,
                        'transition_count': 0
                    }
                
                trend_data[key]['incidents'].append(incident)
                if incident.is_crisis:
                    trend_data[key]['crisis_count'] += 1
                if incident.is_transition_period:
                    trend_data[key]['transition_count'] += 1
            
            # Update or create trend records
            for (student_id, behavior_type), data in trend_data.items():
                existing_trend = BehaviorTrend.query.filter_by(
                    student_id=student_id,
                    behavior_type=behavior_type,
                    date=target_date
                ).first()
                
                incidents = data['incidents']
                total_duration = sum(i.duration_minutes or 0 for i in incidents)
                
                # Calculate average severity
                severity_scores = {
                    'low': 1, 'medium': 2, 'high': 3, 'severe': 4, 'critical': 5
                }
                avg_severity = sum(
                    severity_scores.get(i.severity_level.lower(), 0) for i in incidents
                ) / len(incidents) if incidents else 0
                
                # Find peak time period
                time_periods = [i.time_period for i in incidents if i.time_period]
                peak_time_period = max(set(time_periods), key=time_periods.count) if time_periods else None
                
                # Calculate average noise level
                noise_levels = [i.noise_level_db for i in incidents if i.noise_level_db]
                avg_noise = sum(noise_levels) / len(noise_levels) if noise_levels else None
                
                if existing_trend:
                    existing_trend.incident_count = len(incidents)
                    existing_trend.total_duration_minutes = total_duration
                    existing_trend.average_severity = avg_severity
                    existing_trend.crisis_count = data['crisis_count']
                    existing_trend.peak_time_period = peak_time_period
                    existing_trend.average_noise_level = avg_noise
                    existing_trend.transition_incident_count = data['transition_count']
                    existing_trend.updated_at = datetime.utcnow()
                else:
                    trend = BehaviorTrend(
                        student_id=student_id,
                        behavior_type=behavior_type,
                        date=target_date,
                        incident_count=len(incidents),
                        total_duration_minutes=total_duration,
                        average_severity=avg_severity,
                        crisis_count=data['crisis_count'],
                        peak_time_period=peak_time_period,
                        average_noise_level=avg_noise,
                        transition_incident_count=data['transition_count']
                    )
                    db.session.add(trend)
            
            db.session.commit()
            logger.info(f"Updated behavior trends for {target_date}")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating behavior trends: {e}")
            raise
    
    @staticmethod
    def get_behavior_trends(student_id: Optional[str] = None, days: int = 30) -> List[BehaviorTrend]:
        """Get behavior trends for analysis"""
        query = BehaviorTrend.query.filter(
            BehaviorTrend.date >= date.today() - timedelta(days=days)
        )
        
        if student_id:
            student = StudentDatabaseService.get_student_by_id(student_id)
            if student:
                query = query.filter_by(student_id=student.id)
        
        return query.order_by(desc(BehaviorTrend.date)).all()
    
    @staticmethod
    def get_crisis_incidents(days: int = 7) -> List[BehaviorIncident]:
        """Get recent crisis incidents for monitoring"""
        from datetime import timedelta
        since_date = datetime.utcnow() - timedelta(days=days)
        
        return BehaviorIncident.query.filter(
            and_(
                BehaviorIncident.is_crisis == True,
                BehaviorIncident.start_time >= since_date
            )
        ).order_by(desc(BehaviorIncident.start_time)).all()

# Global service instances
protocol_service = ProtocolDatabaseService()
behavior_service = BehaviorDatabaseService()
student_service = StudentDatabaseService()
analytics_service = AnalyticsDatabaseService()