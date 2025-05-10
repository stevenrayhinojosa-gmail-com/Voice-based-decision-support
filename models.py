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
