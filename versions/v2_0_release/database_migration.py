"""
Database migration service for importing protocol data from CSV files to PostgreSQL
"""
import csv
import json
import logging
from datetime import datetime
from app import app, db
from models import (
    Protocol, DecisionPoint, DecisionOption, BehaviorType, BehaviorProtocol,
    Recommendation, ProtocolKeyword, Student, BehaviorIncident, SeverityLevel
)

logger = logging.getLogger(__name__)

class DatabaseMigrationService:
    """Service for migrating data from CSV files to PostgreSQL database"""
    
    def __init__(self):
        self.imported_protocols = {}
        self.imported_behavior_types = {}
        
    def import_protocols_from_csv(self, csv_file_path):
        """Import protocols from CSV file to database"""
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                protocols_imported = 0
                
                for row in reader:
                    # Check if protocol already exists
                    existing_protocol = Protocol.query.filter_by(name=row['name']).first()
                    if existing_protocol:
                        logger.info(f"Protocol '{row['name']}' already exists, skipping")
                        self.imported_protocols[row['name']] = existing_protocol
                        continue
                    
                    # Create new protocol
                    protocol = Protocol(
                        name=row['name'],
                        description=row.get('description', ''),
                        category=row.get('category', 'Behavioral')
                    )
                    
                    db.session.add(protocol)
                    db.session.flush()  # Get the ID
                    
                    self.imported_protocols[row['name']] = protocol
                    protocols_imported += 1
                    
                    # Create decision points if specified
                    if 'decision_points' in row and row['decision_points']:
                        self._create_decision_points(protocol, row['decision_points'])
                    
                    # Add keywords for NLP matching
                    if 'keywords' in row and row['keywords']:
                        self._create_protocol_keywords(protocol, row['keywords'])
                
                db.session.commit()
                logger.info(f"Imported {protocols_imported} new protocols from {csv_file_path}")
                return protocols_imported
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error importing protocols from {csv_file_path}: {e}")
            raise
    
    def _create_decision_points(self, protocol, decision_points_data):
        """Create decision points for a protocol"""
        try:
            # Parse decision points data (assuming JSON format)
            if isinstance(decision_points_data, str):
                decision_points = json.loads(decision_points_data)
            else:
                decision_points = decision_points_data
            
            for order, point_data in enumerate(decision_points):
                decision_point = DecisionPoint(
                    protocol_id=protocol.id,
                    question=point_data['question'],
                    order=order + 1
                )
                
                db.session.add(decision_point)
                db.session.flush()
                
                # Create options for this decision point
                if 'options' in point_data:
                    for option_data in point_data['options']:
                        option = DecisionOption(
                            decision_point_id=decision_point.id,
                            text=option_data['text'],
                            is_terminal=option_data.get('is_terminal', False),
                            recommendation=option_data.get('recommendation', '')
                        )
                        db.session.add(option)
                        
        except Exception as e:
            logger.error(f"Error creating decision points: {e}")
    
    def _create_protocol_keywords(self, protocol, keywords_data):
        """Create protocol keywords for NLP matching"""
        try:
            if isinstance(keywords_data, str):
                keywords = keywords_data.split(',')
            else:
                keywords = keywords_data
            
            for keyword in keywords:
                keyword = keyword.strip().lower()
                if keyword:
                    protocol_keyword = ProtocolKeyword(
                        protocol_id=protocol.id,
                        keyword=keyword,
                        weight=1.0,
                        language='en'
                    )
                    db.session.add(protocol_keyword)
                    
        except Exception as e:
            logger.error(f"Error creating protocol keywords: {e}")
    
    def import_behavior_types(self):
        """Import standard behavior types"""
        behavior_types_data = [
            {
                'name': 'Physical Aggression',
                'description': 'Hitting, kicking, pushing, or other physical contact behaviors',
                'category': 'Physical',
                'keywords': ['hit', 'kick', 'push', 'punch', 'throw', 'fight', 'attack']
            },
            {
                'name': 'Verbal Aggression',
                'description': 'Yelling, threatening, or using inappropriate language',
                'category': 'Verbal',
                'keywords': ['yell', 'scream', 'threat', 'curse', 'insult', 'shout']
            },
            {
                'name': 'Defiance',
                'description': 'Refusing to follow instructions or comply with requests',
                'category': 'Behavioral',
                'keywords': ['refuse', 'no', 'wont', 'defiant', 'ignore', 'noncompliant']
            },
            {
                'name': 'Disruption',
                'description': 'Behaviors that interrupt classroom activities',
                'category': 'Behavioral',
                'keywords': ['disrupt', 'interrupt', 'noise', 'distract', 'talk']
            },
            {
                'name': 'Self-Harm',
                'description': 'Behaviors that may cause harm to oneself',
                'category': 'Safety',
                'keywords': ['hurt', 'self', 'harm', 'bang', 'scratch', 'bite']
            },
            {
                'name': 'Property Destruction',
                'description': 'Damaging or destroying classroom materials or property',
                'category': 'Property',
                'keywords': ['break', 'destroy', 'damage', 'tear', 'rip']
            },
            {
                'name': 'Elopement',
                'description': 'Leaving designated area without permission',
                'category': 'Safety',
                'keywords': ['run', 'leave', 'escape', 'elope', 'wander']
            }
        ]
        
        imported_count = 0
        for behavior_data in behavior_types_data:
            existing_behavior = BehaviorType.query.filter_by(name=behavior_data['name']).first()
            if existing_behavior:
                self.imported_behavior_types[behavior_data['name']] = existing_behavior
                continue
                
            behavior_type = BehaviorType(
                name=behavior_data['name'],
                description=behavior_data['description'],
                category=behavior_data['category']
            )
            
            db.session.add(behavior_type)
            db.session.flush()
            
            self.imported_behavior_types[behavior_data['name']] = behavior_type
            imported_count += 1
            
        db.session.commit()
        logger.info(f"Imported {imported_count} new behavior types")
        return imported_count
    
    def create_behavior_protocol_mappings(self):
        """Create mappings between behavior types and protocols"""
        mappings = [
            # Physical Aggression mappings
            {
                'behavior_name': 'Physical Aggression',
                'protocol_name': 'Physical Intervention Protocol',
                'severity': SeverityLevel.HIGH.value,
                'is_primary': True
            },
            {
                'behavior_name': 'Physical Aggression',
                'protocol_name': 'De-escalation Protocol',
                'severity': SeverityLevel.MEDIUM.value,
                'is_primary': True
            },
            # Verbal Aggression mappings
            {
                'behavior_name': 'Verbal Aggression',
                'protocol_name': 'De-escalation Protocol',
                'severity': SeverityLevel.MEDIUM.value,
                'is_primary': True
            },
            {
                'behavior_name': 'Verbal Aggression',
                'protocol_name': 'Redirection Protocol',
                'severity': SeverityLevel.LOW.value,
                'is_primary': True
            },
            # Defiance mappings
            {
                'behavior_name': 'Defiance',
                'protocol_name': 'Choice-making Protocol',
                'severity': SeverityLevel.MEDIUM.value,
                'is_primary': True
            },
            # Self-Harm mappings
            {
                'behavior_name': 'Self-Harm',
                'protocol_name': 'Safety Protocol',
                'severity': SeverityLevel.CRITICAL.value,
                'is_primary': True
            },
            # Elopement mappings
            {
                'behavior_name': 'Elopement',
                'protocol_name': 'Safety Protocol',
                'severity': SeverityLevel.SEVERE.value,
                'is_primary': True
            }
        ]
        
        created_count = 0
        for mapping in mappings:
            behavior_type = self.imported_behavior_types.get(mapping['behavior_name'])
            protocol = self.imported_protocols.get(mapping['protocol_name'])
            
            if not behavior_type or not protocol:
                logger.warning(f"Skipping mapping: behavior '{mapping['behavior_name']}' or protocol '{mapping['protocol_name']}' not found")
                continue
            
            # Check if mapping already exists
            existing_mapping = BehaviorProtocol.query.filter_by(
                behavior_type_id=behavior_type.id,
                protocol_id=protocol.id,
                severity_level=mapping['severity']
            ).first()
            
            if existing_mapping:
                continue
                
            behavior_protocol = BehaviorProtocol(
                behavior_type_id=behavior_type.id,
                protocol_id=protocol.id,
                severity_level=mapping['severity'],
                is_primary=mapping['is_primary']
            )
            
            db.session.add(behavior_protocol)
            created_count += 1
        
        db.session.commit()
        logger.info(f"Created {created_count} behavior-protocol mappings")
        return created_count
    
    def import_student_data_from_json(self, student_roster_path, bip_index_path):
        """Import student data from JSON files"""
        try:
            # Import student roster
            with open(student_roster_path, 'r', encoding='utf-8') as file:
                student_data = json.load(file)
            
            # Import BIP index
            bip_data = {}
            try:
                with open(bip_index_path, 'r', encoding='utf-8') as file:
                    bip_data = json.load(file)
            except FileNotFoundError:
                logger.warning(f"BIP index file not found: {bip_index_path}")
            
            imported_count = 0
            for student_info in student_data.get('students', []):
                # Check if student already exists
                existing_student = Student.query.filter_by(student_id=student_info['student_id']).first()
                if existing_student:
                    continue
                
                student = Student(
                    student_id=student_info['student_id'],
                    first_name=student_info['first_name'],
                    last_name=student_info['last_name'],
                    grade_level=student_info.get('grade_level', ''),
                    classroom=student_info.get('classroom', ''),
                    has_bip=student_info['student_id'] in bip_data,
                    bip_details=json.dumps(bip_data.get(student_info['student_id'], {})),
                    emergency_contact=student_info.get('emergency_contact', ''),
                    medical_notes=student_info.get('medical_notes', '')
                )
                
                db.session.add(student)
                imported_count += 1
            
            db.session.commit()
            logger.info(f"Imported {imported_count} new students")
            return imported_count
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error importing student data: {e}")
            raise
    
    def run_full_migration(self):
        """Run complete database migration"""
        logger.info("Starting full database migration...")
        
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Import protocols from CSV files
            try:
                protocols_imported = 0
                protocols_imported += self.import_protocols_from_csv('attached_assets/pfisd_protocols.csv')
                protocols_imported += self.import_protocols_from_csv('attached_assets/sama_protocols.csv')
                logger.info(f"Total protocols imported: {protocols_imported}")
            except Exception as e:
                logger.error(f"Error importing protocols: {e}")
            
            # Import behavior types
            try:
                behavior_types_imported = self.import_behavior_types()
                logger.info(f"Behavior types imported: {behavior_types_imported}")
            except Exception as e:
                logger.error(f"Error importing behavior types: {e}")
            
            # Create behavior-protocol mappings
            try:
                mappings_created = self.create_behavior_protocol_mappings()
                logger.info(f"Behavior-protocol mappings created: {mappings_created}")
            except Exception as e:
                logger.error(f"Error creating mappings: {e}")
            
            # Import student data
            try:
                students_imported = self.import_student_data_from_json(
                    'student_roster.json',
                    'bip_index.json'
                )
                logger.info(f"Students imported: {students_imported}")
            except Exception as e:
                logger.error(f"Error importing student data: {e}")
        
        logger.info("Database migration completed successfully")

# Convenience function for running migration
def run_database_migration():
    """Run the database migration"""
    migration_service = DatabaseMigrationService()
    migration_service.run_full_migration()

if __name__ == "__main__":
    run_database_migration()