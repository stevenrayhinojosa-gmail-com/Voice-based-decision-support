import pandas as pd
import numpy as np
import json
import logging
from models import BehavioralData, Protocol, DecisionPoint, DecisionOption
from app import db

logger = logging.getLogger(__name__)

def load_behavioral_data_to_dataframe():
    """
    Load behavioral data from the database into a pandas DataFrame
    """
    try:
        # Query all data from the BehavioralData table
        data_entries = BehavioralData.query.all()
        
        if not data_entries:
            logger.info("No behavioral data found in database")
            return pd.DataFrame()
        
        # Convert to list of dictionaries
        data_dicts = []
        for entry in data_entries:
            data_dict = {
                'id': entry.id,
                'subject_id': entry.subject_id,
                'age': entry.age,
                'gender': entry.gender,
                'context': entry.context,
                'behavior_description': entry.behavior_description,
                'intensity': entry.intensity,
                'frequency': entry.frequency,
                'duration': entry.duration,
                'triggers': entry.triggers,
                'consequences': entry.consequences,
                'protocol_used': entry.protocol_used,
                'outcome': entry.outcome,
                'created_at': entry.created_at
            }
            data_dicts.append(data_dict)
        
        # Create DataFrame
        df = pd.DataFrame(data_dicts)
        logger.info(f"Loaded {len(df)} behavioral data entries into DataFrame")
        return df
    
    except Exception as e:
        logger.error(f"Error loading behavioral data: {str(e)}")
        return pd.DataFrame()

def preprocess_behavioral_data(df):
    """
    Preprocess behavioral data for machine learning algorithms
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for preprocessing")
        return df
    
    try:
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle missing values
        numeric_cols = ['age', 'intensity', 'frequency', 'duration']
        for col in numeric_cols:
            if col in processed_df.columns:
                # Fill missing numeric values with median
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # Fill missing categorical values with 'unknown'
        categorical_cols = ['gender', 'context']
        for col in categorical_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna('unknown')
        
        # Convert text fields to length features if they exist
        text_cols = ['behavior_description', 'triggers', 'consequences', 'outcome']
        for col in text_cols:
            if col in processed_df.columns:
                # Create a feature that is the length of the text
                processed_df[f'{col}_length'] = processed_df[col].fillna('').apply(len)
                
                # Count word occurrence for common words if needed
                # This would be more complex NLP preprocessing
        
        # One-hot encode categorical variables
        if 'gender' in processed_df.columns:
            gender_dummies = pd.get_dummies(processed_df['gender'], prefix='gender')
            processed_df = pd.concat([processed_df, gender_dummies], axis=1)
            processed_df.drop('gender', axis=1, inplace=True)
            
        if 'context' in processed_df.columns:
            context_dummies = pd.get_dummies(processed_df['context'], prefix='context')
            processed_df = pd.concat([processed_df, context_dummies], axis=1)
            processed_df.drop('context', axis=1, inplace=True)
        
        # Drop non-numeric columns for ML
        drop_cols = ['id', 'subject_id', 'behavior_description', 'triggers', 
                    'consequences', 'outcome', 'created_at']
        for col in drop_cols:
            if col in processed_df.columns:
                processed_df.drop(col, axis=1, inplace=True)
        
        logger.info("Data preprocessing completed successfully")
        return processed_df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return df

def format_decision_tree_path(path, feature_names):
    """
    Format a decision tree path into human-readable format
    """
    if not path:
        return "No path available"
    
    formatted_path = []
    for node_id, condition in path:
        if node_id == 0:  # Root node
            formatted_path.append("Starting point")
        else:
            # Condition will be True for left branch, False for right branch
            direction = "<=" if condition else ">"
            feature_index = node_id % len(feature_names)
            feature_name = feature_names[feature_index]
            threshold = round(path[node_id][1], 2)
            formatted_path.append(f"{feature_name} {direction} {threshold}")
    
    return " -> ".join(formatted_path)

def get_protocol_tree(protocol_id):
    """
    Get the complete decision tree for a protocol
    """
    try:
        protocol = Protocol.query.get(protocol_id)
        if not protocol:
            return None
        
        # Get all decision points for this protocol
        decision_points = DecisionPoint.query.filter_by(protocol_id=protocol_id).order_by(DecisionPoint.order).all()
        if not decision_points:
            return None
        
        # Structure the decision tree
        tree = {
            'protocol': {
                'id': protocol.id,
                'name': protocol.name,
                'description': protocol.description
            },
            'decision_points': {}
        }
        
        # Add all decision points
        for dp in decision_points:
            options = DecisionOption.query.filter_by(decision_point_id=dp.id).all()
            tree['decision_points'][dp.id] = {
                'id': dp.id,
                'question': dp.question,
                'order': dp.order,
                'options': []
            }
            
            # Add options for each decision point
            for option in options:
                tree['decision_points'][dp.id]['options'].append({
                    'id': option.id,
                    'text': option.text,
                    'next_decision_id': option.next_decision_id,
                    'is_terminal': option.is_terminal,
                    'recommendation': option.recommendation
                })
        
        return tree
    
    except Exception as e:
        logger.error(f"Error getting protocol tree: {str(e)}")
        return None

def add_sample_protocol():
    """
    Add a sample behavioral protocol to the database for testing
    """
    try:
        # Check if we already have protocols
        if Protocol.query.count() > 0:
            logger.info("Sample protocol not added - protocols already exist")
            return
        
        # Create a new protocol
        protocol = Protocol(
            name="Classroom Disruption Protocol",
            description="A protocol for addressing disruptive behavior in classroom settings",
            category="Education"
        )
        db.session.add(protocol)
        db.session.commit()
        
        # Create decision points
        dp1 = DecisionPoint(
            protocol_id=protocol.id,
            question="Is the behavior endangering the student or others?",
            order=1
        )
        db.session.add(dp1)
        db.session.commit()
        
        dp2 = DecisionPoint(
            protocol_id=protocol.id,
            question="Is this the first occurrence of the behavior today?",
            order=2
        )
        db.session.add(dp2)
        db.session.commit()
        
        dp3 = DecisionPoint(
            protocol_id=protocol.id,
            question="Has the student been prompted about expectations already?",
            order=3
        )
        db.session.add(dp3)
        db.session.commit()
        
        dp4 = DecisionPoint(
            protocol_id=protocol.id,
            question="Does the behavior continue after intervention?",
            order=4
        )
        db.session.add(dp4)
        db.session.commit()
        
        # Create options for decision point 1
        opt1_dp1 = DecisionOption(
            decision_point_id=dp1.id,
            text="Yes",
            is_terminal=True,
            recommendation="Implement emergency safety procedures and contact administration immediately."
        )
        db.session.add(opt1_dp1)
        
        opt2_dp1 = DecisionOption(
            decision_point_id=dp1.id,
            text="No",
            next_decision_id=dp2.id,
            is_terminal=False
        )
        db.session.add(opt2_dp1)
        
        # Create options for decision point 2
        opt1_dp2 = DecisionOption(
            decision_point_id=dp2.id,
            text="Yes",
            next_decision_id=dp3.id,
            is_terminal=False
        )
        db.session.add(opt1_dp2)
        
        opt2_dp2 = DecisionOption(
            decision_point_id=dp2.id,
            text="No",
            next_decision_id=dp4.id,
            is_terminal=False
        )
        db.session.add(opt2_dp2)
        
        # Create options for decision point 3
        opt1_dp3 = DecisionOption(
            decision_point_id=dp3.id,
            text="Yes",
            next_decision_id=dp4.id,
            is_terminal=False
        )
        db.session.add(opt1_dp3)
        
        opt2_dp3 = DecisionOption(
            decision_point_id=dp3.id,
            text="No",
            is_terminal=True,
            recommendation="Provide clear behavioral expectations and a warning. Use proximity control and nonverbal cues."
        )
        db.session.add(opt2_dp3)
        
        # Create options for decision point 4
        opt1_dp4 = DecisionOption(
            decision_point_id=dp4.id,
            text="Yes",
            is_terminal=True,
            recommendation="Implement planned consequences, document the incident, and contact support staff or administration as needed."
        )
        db.session.add(opt1_dp4)
        
        opt2_dp4 = DecisionOption(
            decision_point_id=dp4.id,
            text="No",
            is_terminal=True,
            recommendation="Provide positive reinforcement for compliance and continue with classroom activities."
        )
        db.session.add(opt2_dp4)
        
        db.session.commit()
        logger.info("Sample protocol added successfully")
        
    except Exception as e:
        logger.error(f"Error adding sample protocol: {str(e)}")
        db.session.rollback()
