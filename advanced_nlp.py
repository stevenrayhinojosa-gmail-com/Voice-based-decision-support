"""
Advanced Natural Language Processing module for teacher query understanding
"""
import nltk
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize logging
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")

# Initialize resources
download_nltk_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess text by removing special characters, converting to lowercase,
    tokenizing, removing stop words, and lemmatizing
    """
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    preprocessed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(preprocessed)

class BehaviorQueryProcessor:
    """
    Process teacher queries about student behavior and match to appropriate 
    behavior types and protocols
    """
    def __init__(self, db=None):
        self.db = db
        self.vectorizer = TfidfVectorizer()
        self.behavior_vectors = None
        self.behavior_types = []
        self.behavior_keywords = {
            'verbal_disruption': [
                'talk back', 'yell', 'shout', 'scream', 'argue', 'rude', 'disrespectful', 
                'interrupted', 'cursing', 'swearing', 'speaking out', 'verbal', 'loud',
                'refuses to talk', 'inappropriate language', 'verbal disruption'
            ],
            'physical_aggression': [
                'hit', 'kick', 'punch', 'throw', 'push', 'shove', 'physical', 'aggressive',
                'violent', 'fighting', 'grabbing', 'strikes', 'hurting', 'hitting', 'attacking',
                'physical aggression', 'harm', 'harming'
            ],
            'emotional_outburst': [
                'cry', 'crying', 'scream', 'upset', 'tantrum', 'meltdown', 'emotional',
                'outburst', 'sobbing', 'tears', 'hysterical', 'overwhelmed', 'emotional outburst'
            ],
            'elopement': [
                'run', 'escape', 'leave', 'fled', 'exit', 'bolt', 'elope', 'runaway',
                'left class', 'left room', 'ran away', 'wandering', 'elopement', 'leaving'
            ],
            'self_injurious': [
                'hit self', 'hurt self', 'self-harm', 'banging head', 'biting self',
                'injuring self', 'self-injurious', 'self-injury', 'harm self', 'self-injurious behavior'
            ],
            'property_destruction': [
                'break', 'destroy', 'damage', 'throw', 'tear', 'vandalize', 'wreck',
                'broke', 'destroyed', 'damaged', 'throwing', 'tearing', 'property', 'destruction',
                'property destruction'
            ],
            'non_compliance': [
                'refuse', 'defy', 'disobey', 'ignore', 'won\'t follow', 'defiant',
                'non-compliant', 'won\'t listen', 'not following', 'refusing', 'refuses',
                'won\'t do', 'non compliance', 'noncompliance', 'instruction'
            ],
            'withdrawal': [
                'withdrawn', 'shut down', 'isolate', 'isolation', 'quiet', 'unresponsive',
                'non-responsive', 'withdraw', 'withdrawing', 'not participating', 'disengaged',
                'withdrawal', 'disengagement'
            ]
        }
        self.severity_keywords = {
            'low': [
                'minor', 'small', 'slight', 'minimal', 'brief', 'low', 'non-disruptive',
                'little', 'short', 'infrequent'
            ],
            'medium': [
                'moderate', 'noticeable', 'disruptive', 'concerning', 'recurring',
                'medium', 'partial', 'intermittent', 'occasional', 'somewhat'
            ],
            'high': [
                'significant', 'serious', 'alarming', 'major', 'highly', 'very', 'repeated',
                'persistent', 'frequent', 'high', 'substantial', 'extensive'
            ],
            'severe': [
                'severe', 'extreme', 'dangerous', 'harmful', 'crisis', 'emergency',
                'critical', 'unsafe', 'uncontrollable', 'constant'
            ],
            'critical': [
                'emergency', 'critical', 'immediate', 'life-threatening', 'crisis',
                'urgent', 'evacuate', 'evacuation', 'code red', 'code'
            ]
        }
        self.initialize_behavior_vectors()
    
    def initialize_behavior_vectors(self):
        """Initialize vectorizer with behavior type keywords"""
        all_behavior_descriptions = []
        
        # Process each behavior type's keywords
        for behavior_type, keywords in self.behavior_keywords.items():
            # Add the behavior type itself as a document
            all_behavior_descriptions.append(" ".join(keywords))
            self.behavior_types.append(behavior_type)
        
        # Fit the vectorizer
        if all_behavior_descriptions:
            self.behavior_vectors = self.vectorizer.fit_transform(all_behavior_descriptions)
            logger.info(f"Initialized {len(self.behavior_types)} behavior type vectors")
    
    def update_from_database(self):
        """Update behavior types and keywords from database if available"""
        if self.db is None:
            logger.warning("Database not available for updating behavior types")
            return
        
        try:
            from models import BehaviorType
            
            # Query all behavior types
            behavior_types = BehaviorType.query.all()
            
            # Clear existing behavior types
            self.behavior_types = []
            all_behavior_descriptions = []
            
            # Create new behavior keywords dictionary
            new_behavior_keywords = {}
            
            for bt in behavior_types:
                # Use name and category as keywords
                keywords = f"{bt.name} {bt.category} {bt.description or ''}"
                # Convert to snake_case for key
                key = bt.name.lower().replace(' ', '_')
                
                # Add to behavior types list
                self.behavior_types.append(key)
                
                # Add keywords to dictionary
                new_behavior_keywords[key] = keywords.split()
                
                # Add to all descriptions for vectorizer
                all_behavior_descriptions.append(keywords)
            
            # Update behavior keywords
            if new_behavior_keywords:
                self.behavior_keywords.update(new_behavior_keywords)
            
            # Refit the vectorizer
            if all_behavior_descriptions:
                self.behavior_vectors = self.vectorizer.fit_transform(all_behavior_descriptions)
                logger.info(f"Updated {len(self.behavior_types)} behavior type vectors from database")
        
        except Exception as e:
            logger.error(f"Error updating behavior types from database: {str(e)}")
    
    def identify_behavior_type(self, query):
        """
        Identify the behavior type from a query
        
        Parameters:
        - query: String containing the teacher's question or description
        
        Returns:
        - behavior_type: The identified behavior type key
        - confidence: Confidence score for the match
        """
        # Preprocess the query
        preprocessed_query = preprocess_text(query)
        
        # Transform the query using the fitted vectorizer
        query_vector = self.vectorizer.transform([preprocessed_query])
        
        # Calculate cosine similarity between query and behavior types
        similarities = cosine_similarity(query_vector, self.behavior_vectors).flatten()
        
        # Find best match
        max_index = np.argmax(similarities)
        best_match = self.behavior_types[max_index]
        confidence = similarities[max_index]
        
        return best_match, confidence
    
    def identify_severity(self, query):
        """
        Identify the severity level from a query
        
        Parameters:
        - query: String containing the teacher's question or description
        
        Returns:
        - severity: The identified severity level
        - confidence: Confidence score for the match
        """
        preprocessed_query = preprocess_text(query)
        
        # Check for matches with severity keywords
        max_count = 0
        best_severity = 'medium'  # Default to medium if no clear match
        
        for severity, keywords in self.severity_keywords.items():
            # Count how many severity keywords appear in the query
            count = sum(1 for keyword in keywords if keyword in preprocessed_query)
            
            if count > max_count:
                max_count = count
                best_severity = severity
        
        # Calculate a simple confidence score
        confidence = max_count / len(self.severity_keywords[best_severity]) if max_count > 0 else 0.5
        
        return best_severity, confidence
    
    def extract_emergency_signals(self, query):
        """
        Check if the query contains emergency signals
        
        Parameters:
        - query: String containing the teacher's question or description
        
        Returns:
        - is_emergency: Boolean indicating if emergency signals were detected
        - signals: List of emergency signals found
        """
        # Emergency keywords to check for
        emergency_keywords = [
            'emergency', 'danger', 'unsafe', 'immediate', 'help', 'urgent',
            'critical', 'life-threatening', 'severe', 'crisis', 'evacuation',
            'evacuate', 'lockdown', 'violent', 'weapon', 'injury', 'blood',
            'medical', 'ambulance', 'police', 'safety', 'threat', 'harm'
        ]
        
        # Check for emergency keywords
        preprocessed_query = preprocess_text(query)
        query_tokens = preprocessed_query.split()
        
        found_signals = []
        for keyword in emergency_keywords:
            if keyword in query_tokens:
                found_signals.append(keyword)
        
        is_emergency = len(found_signals) > 0
        
        return is_emergency, found_signals
    
    def process_teacher_query(self, query):
        """
        Process a teacher's query to identify behavior type, severity,
        and emergency signals
        
        Parameters:
        - query: String containing the teacher's question or description
        
        Returns:
        - Dictionary containing behavior type, severity, and emergency information
        """
        # Extract behavior type
        behavior_type, behavior_confidence = self.identify_behavior_type(query)
        
        # Extract severity
        severity, severity_confidence = self.identify_severity(query)
        
        # Check for emergency signals
        is_emergency, emergency_signals = self.extract_emergency_signals(query)
        
        # If emergency signals found, upgrade severity if needed
        if is_emergency and severity not in ['severe', 'critical']:
            severity = 'severe'
            severity_confidence = max(severity_confidence, 0.8)
        
        # Construct result
        result = {
            'query': query,
            'behavior_type': behavior_type,
            'behavior_confidence': float(behavior_confidence),
            'severity': severity,
            'severity_confidence': float(severity_confidence),
            'is_emergency': is_emergency,
            'emergency_signals': emergency_signals,
            'processed_query': preprocess_text(query)
        }
        
        return result
    
    def get_protocol_for_behavior(self, behavior_type, severity, setting=None, time_period=None, noise_level_db=None):
        """
        Get appropriate protocol ID for the identified behavior type and severity,
        with optional context parameters
        
        Parameters:
        - behavior_type: The identified behavior type key
        - severity: The identified severity level
        - setting: Optional classroom setting (e.g., 'classroom', 'hallway')
        - time_period: Optional time period from context sensor
        - noise_level_db: Optional noise level from context sensor
        
        Returns:
        - protocol_id: ID of the recommended protocol
        - protocol_name: Name of the protocol
        """
        if self.db is None:
            logger.warning("Database not available for finding protocol")
            return None, None
        
        try:
            from models import BehaviorType, BehaviorProtocol, Protocol
            
            # Context-aware protocol selection
            if time_period and noise_level_db:
                # Special case for anxiety during quiet morning periods
                if behavior_type.lower() == 'anxiety' and noise_level_db < -70 and 'morning' in time_period:
                    # Look for SEL anxiety protocol
                    sel_protocol = Protocol.query.filter(Protocol.name.like('%SEL%')).first()
                    if sel_protocol:
                        return sel_protocol.id, f"{sel_protocol.name} (Quiet Morning Context)"
                
                # Special case for disruption during noisy post-lunch periods
                if behavior_type.lower() == 'disruption' and noise_level_db > -60 and 'lunch' in time_period:
                    # Look for PBIS disruption protocol
                    pbis_protocol = Protocol.query.filter(Protocol.name.like('%PBIS%')).first()
                    if pbis_protocol:
                        return pbis_protocol.id, f"{pbis_protocol.name} (Noisy Lunch Context)"
                
                # Use SAMA for risk behaviors and high noise levels
                if behavior_type.lower() in ['aggression', 'risk behavior'] and noise_level_db > -65:
                    sama_protocol = Protocol.query.filter(Protocol.name.like('%SAMA%')).first()
                    if sama_protocol:
                        return sama_protocol.id, f"{sama_protocol.name} (High Noise Context)"
            
            # Setting-specific protocol matching
            if setting:
                # Try to find a protocol specifically for this setting
                behavior_type_db = BehaviorType.query.filter(BehaviorType.name.ilike(f"%{behavior_type}%")).first()
                if behavior_type_db:
                    # Check if we have a setting-specific protocol from PFISD data
                    custom_protocol = self.db.session.execute(
                        "SELECT p.id, p.name FROM protocols p "
                        "JOIN decision_points dp ON p.id = dp.protocol_id "
                        f"WHERE dp.question LIKE '%{setting}%' AND "
                        f"dp.question LIKE '%{behavior_type}%' "
                        "LIMIT 1"
                    ).fetchone()
                    
                    if custom_protocol:
                        return custom_protocol[0], f"{custom_protocol[1]} (Setting: {setting})"
            
            # Convert severity to database format
            db_severity = severity.upper()
            behavior_name = behavior_type.replace('_', ' ')
            
            # Find the behavior type ID
            bt = BehaviorType.query.filter(BehaviorType.name.ilike(f"%{behavior_name}%")).first()
            if not bt:
                logger.warning(f"Behavior type not found: {behavior_type}")
                return None, None
            
            # Find the protocol for this behavior type and severity
            protocol_link = BehaviorProtocol.query.filter_by(
                behavior_type_id=bt.id,
                severity_level=severity,
                is_primary=True
            ).first()
            
            # If no protocol found with primary flag, try any protocol
            if not protocol_link:
                protocol_link = BehaviorProtocol.query.filter_by(
                    behavior_type_id=bt.id,
                    severity_level=severity
                ).first()
            
            # If still no protocol, try with default severity
            if not protocol_link:
                protocol_link = BehaviorProtocol.query.filter_by(
                    behavior_type_id=bt.id,
                    severity_level='medium',
                ).first()
            
            # If still nothing, return None
            if not protocol_link:
                return None, None
            
            # Get protocol details
            protocol = Protocol.query.get(protocol_link.protocol_id)
            if not protocol:
                return None, None
            
            return protocol.id, protocol.name
            
        except Exception as e:
            logger.error(f"Error finding protocol for behavior: {str(e)}")
            return None, None
    
    def get_recommendation_for_behavior(self, behavior_type, severity, setting=None, time_period=None, noise_level_db=None):
        """
        Get appropriate recommendation for the identified behavior type and severity,
        with optional context parameters
        
        Parameters:
        - behavior_type: The identified behavior type key
        - severity: The identified severity level
        - setting: Optional classroom setting (e.g., 'classroom', 'hallway')
        - time_period: Optional time period from context sensor
        - noise_level_db: Optional noise level from context sensor
        
        Returns:
        - recommendation: Dictionary containing recommendation details
        """
        if self.db is None:
            logger.warning("Database not available for finding recommendation")
            return None
        
        try:
            from models import BehaviorType, Recommendation, Protocol, DecisionPoint, DecisionOption
            
            # Convert snake_case to database format (if needed)
            behavior_name = behavior_type.replace('_', ' ')
            
            # Find the behavior type ID
            bt = BehaviorType.query.filter(BehaviorType.name.ilike(f"%{behavior_name}%")).first()
            if not bt:
                logger.warning(f"Behavior type not found: {behavior_type}")
                return None
            
            # Find the recommendation for this behavior type and severity
            recommendation = Recommendation.query.filter_by(
                behavior_type_id=bt.id,
                severity_level=severity
            ).first()
            
            # If no recommendation found, try with default severity
            if not recommendation:
                recommendation = Recommendation.query.filter_by(
                    behavior_type_id=bt.id
                ).first()
            
            # If still no recommendation, try general recommendations
            if not recommendation:
                recommendation = Recommendation.query.filter_by(
                    behavior_type_id=None
                ).first()
            
            # If still nothing, return None
            if not recommendation:
                return None
            
            return {
                'id': recommendation.id,
                'title': recommendation.title,
                'content': recommendation.content,
                'category': recommendation.category
            }
            
        except Exception as e:
            logger.error(f"Error finding recommendation for behavior: {str(e)}")
            return None
    
    def get_response_for_query(self, query):
        """
        Generate a complete response for a teacher query
        
        Parameters:
        - query: String containing the teacher's question or description
        
        Returns:
        - Dictionary containing response details
        """
        # Process the query
        query_analysis = self.process_teacher_query(query)
        
        # Get behavior type and severity
        behavior_type = query_analysis['behavior_type']
        severity = query_analysis['severity']
        
        # Get protocol and recommendation
        protocol_id, protocol_name = self.get_protocol_for_behavior(behavior_type, severity)
        recommendation = self.get_recommendation_for_behavior(behavior_type, severity)
        
        # Construct response
        response = {
            'query': query,
            'analysis': query_analysis,
            'protocol_id': protocol_id,
            'protocol_name': protocol_name,
            'recommendation': recommendation,
            'success': protocol_id is not None or recommendation is not None
        }
        
        return response