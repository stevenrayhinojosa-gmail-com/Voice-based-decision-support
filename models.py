from datetime import datetime
from enum import Enum
from app import db

# Enum for severity levels
class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"

class Protocol(db.Model):
    """Model for behavioral protocols stored in the system"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to decision points
    decision_points = db.relationship('DecisionPoint', backref='protocol', lazy=True)
    
    def __repr__(self):
        return f'<Protocol {self.name}>'

class DecisionPoint(db.Model):
    """Model for decision points within a protocol"""
    id = db.Column(db.Integer, primary_key=True)
    protocol_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to options (specify the foreign key explicitly)
    options = db.relationship(
        'DecisionOption', 
        foreign_keys='DecisionOption.decision_point_id',
        backref='decision_point', 
        lazy=True
    )
    
    def __repr__(self):
        return f'<DecisionPoint {self.question[:20]}...>'

class DecisionOption(db.Model):
    """Model for options at each decision point"""
    id = db.Column(db.Integer, primary_key=True)
    decision_point_id = db.Column(db.Integer, db.ForeignKey('decision_point.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    next_decision_id = db.Column(db.Integer, db.ForeignKey('decision_point.id'), nullable=True)
    is_terminal = db.Column(db.Boolean, default=False)
    recommendation = db.Column(db.Text, nullable=True)
    
    # Define the relationship to the next decision point with an explicit foreign key
    next_decision = db.relationship(
        'DecisionPoint',
        foreign_keys=[next_decision_id],
        backref=db.backref('previous_options', lazy=True)
    )
    
    def __repr__(self):
        return f'<DecisionOption {self.text[:20]}...>'

class BehavioralData(db.Model):
    """Model for storing behavioral data entries"""
    id = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.String(50), nullable=False)  # Anonymized identifier
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    context = db.Column(db.String(100), nullable=True)
    behavior_description = db.Column(db.Text, nullable=False)
    intensity = db.Column(db.Integer, nullable=True)  # Scale 1-10
    frequency = db.Column(db.Integer, nullable=True)  # Count
    duration = db.Column(db.Integer, nullable=True)  # In minutes
    triggers = db.Column(db.Text, nullable=True)
    consequences = db.Column(db.Text, nullable=True)
    protocol_used = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=True)
    outcome = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    time_period = db.Column(db.String(50), nullable=True)  # School time period (e.g., "pre-class", "recess")
    noise_level_db = db.Column(db.Float, nullable=True)  # Ambient noise level in decibels
    
    def __repr__(self):
        return f'<BehavioralData {self.subject_id}>'

class MLModel(db.Model):
    """Model for storing trained machine learning models"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    model_type = db.Column(db.String(50), nullable=False)  # e.g., 'decision_tree', 'random_forest'
    features = db.Column(db.Text, nullable=False)  # JSON string of feature names
    target = db.Column(db.String(50), nullable=False)
    performance_metrics = db.Column(db.Text, nullable=True)  # JSON string of metrics
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<MLModel {self.name}>'


# New models for enhanced behavioral support system

class BehaviorType(db.Model):
    """Model for categorizing different types of behaviors"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=True)  # E.g., "Verbal", "Physical", "Emotional"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    behavior_protocols = db.relationship('BehaviorProtocol', backref='behavior_type', lazy=True)
    
    def __repr__(self):
        return f'<BehaviorType {self.name}>'


class BehaviorProtocol(db.Model):
    """Model for connecting behaviors to protocols with severity levels"""
    id = db.Column(db.Integer, primary_key=True)
    behavior_type_id = db.Column(db.Integer, db.ForeignKey('behavior_type.id'), nullable=False)
    protocol_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=False)
    severity_level = db.Column(db.String(20), nullable=False)  # Uses values from SeverityLevel enum
    is_primary = db.Column(db.Boolean, default=False)  # Is this the primary protocol for this behavior/severity?
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<BehaviorProtocol {self.behavior_type_id}:{self.protocol_id}>'


class Recommendation(db.Model):
    """Model for storing additional recommendations, resources, and notes"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=True)  # E.g., "Resource", "Training", "Best Practice"
    behavior_type_id = db.Column(db.Integer, db.ForeignKey('behavior_type.id'), nullable=True)
    protocol_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=True)
    severity_level = db.Column(db.String(20), nullable=True)  # Optional severity level filter
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    behavior_type = db.relationship('BehaviorType', backref='recommendations', lazy=True)
    protocol = db.relationship('Protocol', backref='recommendations', lazy=True)
    
    def __repr__(self):
        return f'<Recommendation {self.title}>'

class Student(db.Model):
    """Model for student information and BIP tracking"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    grade_level = db.Column(db.String(20), nullable=True)
    classroom = db.Column(db.String(50), nullable=True)
    has_bip = db.Column(db.Boolean, default=False)
    bip_details = db.Column(db.Text, nullable=True)  # JSON string of BIP strategies
    emergency_contact = db.Column(db.String(200), nullable=True)
    medical_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    incidents = db.relationship('BehaviorIncident', backref='student', lazy=True)
    
    def __repr__(self):
        return f'<Student {self.first_name} {self.last_name}>'

class BehaviorIncident(db.Model):
    """Model for storing individual behavior incidents"""
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.String(100), unique=True, nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=True)
    
    # Incident details
    behavior_description = db.Column(db.Text, nullable=False)
    behavior_type = db.Column(db.String(50), nullable=True)
    severity_level = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    
    # Environmental context
    time_period = db.Column(db.String(50), nullable=True)
    noise_level_db = db.Column(db.Float, nullable=True)
    is_transition_period = db.Column(db.Boolean, default=False)
    
    # Crisis management
    is_crisis = db.Column(db.Boolean, default=False)
    crisis_alert_sent = db.Column(db.Boolean, default=False)
    protocol_used_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=True)
    
    # Timeline
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_minutes = db.Column(db.Integer, nullable=True)
    
    # Outcomes
    outcome = db.Column(db.Text, nullable=True)
    recommendations_given = db.Column(db.Text, nullable=True)  # JSON array of recommendations
    teacher_feedback = db.Column(db.String(20), nullable=True)  # thumbs_up/thumbs_down
    feedback_comment = db.Column(db.Text, nullable=True)
    
    # Analytics support
    keywords = db.Column(db.Text, nullable=True)  # JSON array of extracted keywords
    confidence_scores = db.Column(db.Text, nullable=True)  # JSON object of NLP confidence scores
    clarification_prompted = db.Column(db.Boolean, default=False)
    clarification_response = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    protocol_used = db.relationship('Protocol', backref='incidents', lazy=True)
    interactions = db.relationship('IncidentInteraction', backref='incident', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<BehaviorIncident {self.incident_id}>'

class IncidentInteraction(db.Model):
    """Model for tracking interactions during an incident"""
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('behavior_incident.id'), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)  # voice_input, recommendation, clarification, etc.
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    interaction_metadata = db.Column(db.Text, nullable=True)  # JSON for additional data
    
    def __repr__(self):
        return f'<IncidentInteraction {self.interaction_type}>'

class ProtocolKeyword(db.Model):
    """Model for mapping keywords to protocols for NLP analysis"""
    id = db.Column(db.Integer, primary_key=True)
    protocol_id = db.Column(db.Integer, db.ForeignKey('protocol.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    weight = db.Column(db.Float, default=1.0)  # Importance weight for this keyword
    language = db.Column(db.String(10), default='en')  # Language code
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    protocol = db.relationship('Protocol', backref='keywords', lazy=True)
    
    def __repr__(self):
        return f'<ProtocolKeyword {self.keyword}>'

class BehaviorTrend(db.Model):
    """Model for storing aggregated behavior trend data"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=True)
    behavior_type = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Aggregated metrics
    incident_count = db.Column(db.Integer, default=0)
    total_duration_minutes = db.Column(db.Integer, default=0)
    average_severity = db.Column(db.Float, nullable=True)
    crisis_count = db.Column(db.Integer, default=0)
    
    # Environmental patterns
    peak_time_period = db.Column(db.String(50), nullable=True)
    average_noise_level = db.Column(db.Float, nullable=True)
    transition_incident_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    student = db.relationship('Student', backref='behavior_trends', lazy=True)
    
    def __repr__(self):
        return f'<BehaviorTrend {self.behavior_type} {self.date}>'
