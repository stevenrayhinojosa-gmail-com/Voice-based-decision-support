"""
Script to import behavioral protocols and data from the provided CSV files
"""
import csv
import logging
from app import app, db
from models import Protocol, DecisionPoint, DecisionOption, BehaviorType, BehaviorProtocol, SeverityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_sama_protocols():
    """Import SAMA protocol data from CSV"""
    try:
        with open('attached_assets/sama_protocols.csv', 'r') as file:
            reader = csv.DictReader(file)
            
            # Create a SAMA protocol category
            protocol = Protocol.query.filter_by(name="SAMA Crisis Response Protocol").first()
            if not protocol:
                protocol = Protocol(
                    name="SAMA Crisis Response Protocol",
                    description="Systematic crisis management protocol for behavior escalation",
                    category="Crisis Management"
                )
                db.session.add(protocol)
                db.session.commit()
                logger.info(f"Created SAMA protocol: {protocol.name}")
            
            # Process rows
            for idx, row in enumerate(reader):
                # Create a decision point for each SAMA stage
                dp = DecisionPoint.query.filter_by(
                    protocol_id=protocol.id,
                    question=f"Is the student in {row['SAMA Stage']} stage?"
                ).first()
                
                if not dp:
                    dp = DecisionPoint(
                        protocol_id=protocol.id,
                        question=f"Is the student in {row['SAMA Stage']} stage?",
                        order=idx + 1
                    )
                    db.session.add(dp)
                    db.session.commit()
                    logger.info(f"Created decision point: {dp.question}")
                
                # Create yes option with recommendations
                yes_option = DecisionOption.query.filter_by(
                    decision_point_id=dp.id,
                    text=f"Yes - Indicators: {row['Indicators']}"
                ).first()
                
                if not yes_option:
                    yes_option = DecisionOption(
                        decision_point_id=dp.id,
                        text=f"Yes - Indicators: {row['Indicators']}",
                        is_terminal=True,
                        recommendation=f"SAMA {row['SAMA Stage']} Response: {row['Staff Response']}. Goal: {row['Goal of Action']}. Notes: {row['Notes']}"
                    )
                    db.session.add(yes_option)
                    
                # Create no option that points to next stage if available
                next_idx = idx + 1
                next_dp_id = None
                if next_idx < sum(1 for _ in reader):  # Count remaining rows
                    next_stage_dp = DecisionPoint.query.filter_by(protocol_id=protocol.id, order=next_idx + 1).first()
                    if next_stage_dp:
                        next_dp_id = next_stage_dp.id
                
                no_option = DecisionOption.query.filter_by(
                    decision_point_id=dp.id,
                    text="No - Not showing these indicators"
                ).first()
                
                if not no_option:
                    no_option = DecisionOption(
                        decision_point_id=dp.id,
                        text="No - Not showing these indicators",
                        is_terminal=next_dp_id is None,
                        next_decision_id=next_dp_id,
                        recommendation=None if next_dp_id else "Student does not appear to be in crisis at this time."
                    )
                    db.session.add(no_option)
            
            db.session.commit()
            logger.info("SAMA protocols imported successfully")
            return True
    except Exception as e:
        logger.error(f"Error importing SAMA protocols: {str(e)}")
        return False

def import_pfisd_protocols():
    """Import PFISD protocol data from CSV"""
    try:
        with open('attached_assets/pfisd_protocols.csv', 'r') as file:
            reader = csv.DictReader(file)
            
            protocols_created = set()
            
            for row in reader:
                # Create behavior type if it doesn't exist
                behavior_type = BehaviorType.query.filter_by(name=row['behavior_type']).first()
                if not behavior_type:
                    behavior_type = BehaviorType(
                        name=row['behavior_type'],
                        description=f"{row['behavior_type']} behaviors in {row['setting']}",
                        category=row['tier']
                    )
                    db.session.add(behavior_type)
                    db.session.commit()
                    logger.info(f"Created behavior type: {behavior_type.name}")
                
                # Create or get protocol
                protocol_name = f"{row['protocol_source']} - {row['tier']}"
                if protocol_name not in protocols_created:
                    protocol = Protocol.query.filter_by(name=protocol_name).first()
                    if not protocol:
                        protocol = Protocol(
                            name=protocol_name,
                            description=f"{row['protocol_source']} protocols for {row['tier']} interventions",
                            category=row['protocol_source']
                        )
                        db.session.add(protocol)
                        db.session.commit()
                        protocols_created.add(protocol_name)
                        logger.info(f"Created protocol: {protocol.name}")
                else:
                    protocol = Protocol.query.filter_by(name=protocol_name).first()
                
                # Determine severity based on tier
                severity = "LOW"
                if row['tier'] == "Tier 2":
                    severity = "MEDIUM"
                elif row['tier'] == "Tier 3":
                    severity = "HIGH"
                
                # Create behavior protocol connection
                behavior_protocol = BehaviorProtocol.query.filter_by(
                    behavior_type_id=behavior_type.id,
                    protocol_id=protocol.id,
                    severity_level=severity
                ).first()
                
                if not behavior_protocol:
                    behavior_protocol = BehaviorProtocol(
                        behavior_type_id=behavior_type.id,
                        protocol_id=protocol.id,
                        severity_level=severity,
                        is_primary=True
                    )
                    db.session.add(behavior_protocol)
                
                # Create decision point for this specific behavior/setting
                dp_name = f"Is this {row['behavior_type']} occurring in the {row['setting']}?"
                dp = DecisionPoint.query.filter_by(
                    protocol_id=protocol.id,
                    question=dp_name
                ).first()
                
                if not dp:
                    dp = DecisionPoint(
                        protocol_id=protocol.id,
                        question=dp_name,
                        order=int(row['protocol_id'])  # Use protocol_id as order
                    )
                    db.session.add(dp)
                    db.session.commit()
                    
                    # Add yes option with recommendation
                    yes_option = DecisionOption(
                        decision_point_id=dp.id,
                        text=f"Yes - {row['behavior_type']} in {row['setting']}",
                        is_terminal=True,
                        recommendation=row['action_steps']
                    )
                    db.session.add(yes_option)
                    
                    # Add no option
                    no_option = DecisionOption(
                        decision_point_id=dp.id,
                        text=f"No - Not {row['behavior_type']} or not in {row['setting']}",
                        is_terminal=True,
                        recommendation=f"Assess whether this is a different behavior type or setting. Consider general {row['tier']} support strategies."
                    )
                    db.session.add(no_option)
            
            db.session.commit()
            logger.info("PFISD protocols imported successfully")
            return True
    except Exception as e:
        logger.error(f"Error importing PFISD protocols: {str(e)}")
        return False

def import_behavioral_data():
    """Import behavioral data from CSV"""
    try:
        with open('attached_assets/behavioral_data.csv', 'r') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # We'll use this data to create keyword mappings and recommendations
                behavior_type = row['behavior_type']
                severity_level = row['severity_level'].upper()
                setting = row['setting']
                action = row['recommended_action']
                
                # Create behavior type if it doesn't exist
                bt = BehaviorType.query.filter_by(name=behavior_type).first()
                if not bt:
                    bt = BehaviorType(
                        name=behavior_type,
                        description=f"{behavior_type} behaviors requiring intervention",
                        category="Student Behavior"
                    )
                    db.session.add(bt)
                    db.session.commit()
                
                # Find any protocol to associate with this behavior
                protocol = Protocol.query.filter_by(name=row['protocol_source']).first()
                if not protocol:
                    # Create a new protocol based on the source
                    protocol = Protocol(
                        name=row['protocol_source'],
                        description=f"Behavior protocols from {row['protocol_source']}",
                        category="Behavior Management"
                    )
                    db.session.add(protocol)
                    db.session.commit()
                
                # Create behavior protocol connection if it doesn't exist
                behavior_protocol = BehaviorProtocol.query.filter_by(
                    behavior_type_id=bt.id,
                    protocol_id=protocol.id,
                    severity_level=severity_level
                ).first()
                
                if not behavior_protocol:
                    behavior_protocol = BehaviorProtocol(
                        behavior_type_id=bt.id,
                        protocol_id=protocol.id,
                        severity_level=severity_level,
                        is_primary=True
                    )
                    db.session.add(behavior_protocol)
            
            db.session.commit()
            logger.info("Behavioral data imported successfully")
            return True
    except Exception as e:
        logger.error(f"Error importing behavioral data: {str(e)}")
        return False

def run_imports():
    with app.app_context():
        import_sama_protocols()
        import_pfisd_protocols()
        import_behavioral_data()
        
if __name__ == "__main__":
    run_imports()